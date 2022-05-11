#!/usr/bin/env python
# coding: utf-8

# ### A neural network that uses patient health data to determine the risk of heart attack. The features consist of various aspects of patients' physiology and the label is a determination (by a doctor) of a high or low risk of heart attack.

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from sklearn.model_selection import train_test_split

cd = Path.cwd()
filename = 'heart'
filename += '.csv'

path = os.path.join(cd,'Downloads',filename)

df = pd.read_csv(path)
df.rename(columns={'age':'Age',
                   'cp':'Chest Pain Type',
                  'trestbps':'Resting Blood Pressure',
                  'chol':'Serum Cholestoral in mg/dl',
                  'fbs':'Fasting Blood Sugar > 120 mg/dl',
                  'restecg':'Resting ECG Results',
                  'thalach':'Maximum Heart Rate',
                  'exang':'Exercise Induced Angina',
                  'oldpeak':'ST Depression Induced by Exercise',
                  'slope':'Slope of Peak Exercise ST Segment',
                  'ca':'Major Vessels Colored by Flourosopy',
                  'thal':'Defect (Normal/Fixed/Reversable)',
                  'target':'High Risk of Heart Attack'},
         inplace=True)

df


# ### All columns are scaled to values between 0 and 1 using the maximum and minimum of each column in the training data.

# In[2]:


X = df.drop(['High Risk of Heart Attack'],axis=1)
y = df['High Risk of Heart Attack']
batch_size = 24

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=batch_size)
test_df = X_test.reset_index().drop(['index'],axis=1).astype('int32')
X_train = np.array(X_train)
X_test = np.array(X_test)

for col in range(X.shape[1]):
    col_max = X_train[:,col].max()
    col_min = X_train[:,col].min()
    d = col_max - col_min
    X_train[:,col] = (X_train[:,col] - col_min)/d
    X_test[:,col] = (X_test[:,col] - col_min)/d


# In[3]:


train_dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
train_dataset = train_dataset.batch(batch_size).shuffle(batch_size*2).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test,y_test))
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# In[4]:


l2 = keras.regularizers.L2(2e-3)

model = keras.Sequential([
    keras.layers.Dense(55,activation='relu',input_shape=(X_train.shape[1],),kernel_regularizer='l1'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(55,activation='relu',kernel_regularizer=l2),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(55,activation='relu',kernel_regularizer=l2),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1,activation='sigmoid')
])

optimizer = keras.optimizers.Adam(learning_rate=2e-3)

model.compile(loss='binary_crossentropy',
             optimizer=optimizer,
             metrics=['accuracy',keras.metrics.Recall(name='recall')])

model.summary()


# ### Since training can be eratic (with the model sometimes worsening over more epochs), the weights with the best accuracy on the test data are saved and used for the final model.

# In[5]:


epochs = 112

weights_path = 'heart-attack-model-weights.h5'

checkpoint = keras.callbacks.ModelCheckpoint(weights_path,
                                            verbose=1,
                                            save_best_only=True,
                                            save_weights_only=True,
                                            monitor='val_accuracy')

history = model.fit(train_dataset,
                   validation_data=test_dataset,
                   epochs=epochs,
                   callbacks=[checkpoint])


# ### The following charts display the loss, accuracy, and recall of the model through all the epochs. Practically, the most important metric is recall, because it's better the catch all instances of heart disease than to miss even a few, and so the model was optimized for recall.

# In[6]:


hist = history.history

sns.set_style('darkgrid')
def plot_metric(metric,grid_start=0):
    train_metric = hist[metric][grid_start:]
    val_metric = hist['val_' + metric][grid_start:]
    epoch = (np.arange(epochs) + 1)[grid_start:]
    plt.figure(figsize=(14,7))
    plt.plot(epoch,train_metric)
    plt.plot(epoch,val_metric)
    title = 'Training & Validation ' + metric.title()
    plt.title(title,fontdict={'fontsize':20})
    plt.xlabel('Epoch',fontdict={'fontsize':16})
    plt.ylabel(metric.title(),fontdict={'fontsize':16})
    plt.legend(['Training '+metric.title(),
               'Validation '+metric.title()],prop={'size':18})
    plt.show()
    
plot_metric('loss')
plot_metric('accuracy')
plot_metric('recall')

model.load_weights(weights_path)
val_loss,val_accuracy,val_recall = model.evaluate(test_dataset)

print('\nFinal loss on test data: ' + str(round(val_loss,2)))
print('Final accuracy on test data: ' + str(round(val_accuracy,2)))
print('Final recall on test data: ' + str(round(val_recall,2)))


# ### The table below shows the test data with the correct labels, as well as the model's predictions of heart attack risk in terms of percentage (i.e. the model predicts how likely it is, based on a patient's health data, for a doctor to declare that patient as having a high risk of a heart attack.)

# In[7]:


predictions = model.predict(X_test).squeeze()
percentage = (predictions*100).astype('int').astype('str')
percentage = pd.Series(percentage) + '%'

test_df['High Risk of Heart Attack'] = np.array(y_test)
test_df['Prediction of Heart Attack Risk'] = percentage
test_df.style.hide_index()

