#!/usr/bin/env python
# coding: utf-8

# # 𝐼𝓂𝓅𝑜𝓇𝓉 𝓁𝒾𝒷𝓇𝒶𝓇𝒾𝑒𝓈

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# # LOAD DATASET
# 

# In[2]:


dataset = pd.read_csv('11Absenteeism_at_work.csv')
dataset.head(10)


# In[3]:


dataset.columns


# In[4]:


#shape and type of dataset

print('Shape of dataset is:{}'.format(dataset.shape))
print('Type of features is:{}'.format(dataset.dtypes))


# In[5]:


#mean of column 'Absenteeism time in hours'
dataset['Absenteeism time in hours'].mean()


# # 𝕯𝖎𝖋𝖋𝖊𝖗𝖊𝖓𝖙 𝖛𝖎𝖊𝖜𝖘 𝖔𝖋 𝖉𝖆𝖙𝖆 𝖆𝖓𝖉 𝖜𝖎𝖙𝖍 𝖗𝖊𝖘𝖕𝖊𝖈𝖙𝖎𝖛𝖊 𝖙𝖔 𝖙𝖆𝖗𝖌𝖊𝖙 𝖈𝖔𝖑𝖚𝖒𝖓

# In[6]:


#histogram view of Absenteeism time in hours
plt.hist(dataset["Absenteeism time in hours"])


# In[7]:


sns.jointplot(y='Transportation expense',x='Month of absence',data=dataset,kind='scatter',color='green')


# In[8]:


sns.jointplot(x='Absenteeism time in hours',y='Seasons',data=dataset)


# In[9]:


sns.jointplot(y='Age',x='Absenteeism time in hours',data=dataset)


# The above code shows most of the employees are between 35-40 and most of the absence records is between 0-20 hours.
# so the employees between 30-35 has most absence records followed by age group of 50, and those who are below 30.

# In[10]:


plt.hist(dataset["Age"])


# In[11]:


sns.jointplot(y='Body mass index',x='Absenteeism time in hours',data=dataset)


# In the above graphs we are checking the relation tht age & body mass index play a role in absenteesim

# In[12]:


plt.figure(figsize=(14,7))
sns.lmplot(x='Age',y='Absenteeism time in hours',data=dataset,hue='Day of the week',height=4,aspect=3)


# We are able to see tht 5th day of the week has a lot of absenteesim followed by 6th day through all age factors in the above graph 

# In[13]:


plt.figure(figsize=(12,6))
dataset[dataset['Son']!=0]['Absenteeism time in hours'].plot.hist(bins=30)


# In[14]:


plt.figure(figsize=(12,6))
dataset[dataset['Son']==0]['Absenteeism time in hours'].plot.hist(bins=30)


# In[15]:


g = sns.FacetGrid(data=dataset,col='Son')
g.map(plt.hist,'Absenteeism time in hours')


# So from the above graphs we can see tht people without children tend to work more and people with 1 or 2 child tend to have more absenteesim in hours 

# In[16]:


plt.figure(figsize=(10,5))
sns.displot(dataset['Reason for absence'])


# # Machine Learning Model Training 

# In[17]:


#count of entries in column 
dataset[dataset['Absenteeism time in hours']==0].count()


# In[18]:


dataset.head(740)


# We see tht out of 740 workers only 44 have not been absent till now 

# In[19]:


#dropping unecessary columns

dataset.drop(['Work load Average/day '], axis=1, inplace=True)
dataset.head()


# In[20]:


#dividing dataset inorder to test and train the model

X = dataset.iloc[:,:20]
Y = dataset.iloc[:,19:]


# In[21]:


#Dividing data inorder to train and predict 
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(X, Y, test_size= 0.3,shuffle=False)  


# In[22]:


from sklearn.ensemble import RandomForestRegressor


# In[23]:


#Passing parameters so that to train the model
regressor = RandomForestRegressor(n_estimators= 100, max_features= 'auto',max_depth=None,min_samples_leaf=1)


# In[24]:


#predicting the Target Variable
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)


# In[25]:


y_pred


# In[26]:


dataset.tail(6)


# In[27]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[28]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# r2_score - to check accuracy of the model

# In[ ]:





# In[ ]:




