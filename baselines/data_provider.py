import numpy as np
import pandas as pd
import re
import os
def preprocess_train():
    set_name="train"    
    #load train data
    lists = os.listdir(r'{}'.format(set_name))
    for i in range(len(lists)):
        lists[i] = int(lists[i][0:-4])
    lists.sort()

    samples = []

    for num in lists:

        filepath = r'{}/{}.csv'.format(set_name,num)
        content = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        if len(content.shape)==1:
            content = content[np.newaxis,:]
        data = content[:,1:]
        min_v = np.nanmin(data,axis=0)
        max_v = np.nanmax(data,axis=0)
        mean_v = np.nanmean(data,axis=0)
        statistics = np.zeros((3,data.shape[1]))

        statistics[0] = min_v
        statistics[1] = max_v
        statistics[2] = mean_v

        sample = np.nan_to_num(statistics)
        samples.append(sample)

    train_data = np.stack(samples,axis=0)

    min_v = train_data[:,0,:].min(axis=0)
    max_v = train_data[:,1,:].max(axis=0)



    #normalize
    s = max_v - min_v
    s = s + (s==0).astype(np.float)

    train_data = (train_data - min_v)/s

    np.save('{}_data.npy'.format(set_name),train_data)
    
    
    

    #label read
    filepath = r'{}_label.csv'.format(set_name)

    df = pd.read_csv("train_label.csv")
    c = df.values
    new=np.zeros([c.shape[0],7])
    new[:,0]=c[:,0]
    for i in range(c.shape[0]):
        if pd.isna(df).values[i,1] == False:
            new[i,1:]=sum(map(lambda x: np.eye(6)[int(x)-1],(re.findall("\d",c[i,1])))) 

    labels = np.zeros((len(samples),3))

    for i in range(new.shape[0]):
        ind = new[i][0].astype(np.int)
        labels[ind] = new[i,1:4]



    np.save('train_label.npy',labels)


def preprocess_test():
    set_name="train"
    #load train data
    lists = os.listdir(r'{}'.format(set_name))
    for i in range(len(lists)):
        lists[i] = int(lists[i][0:-4])
    lists.sort()

    samples = []

    for num in lists:

        filepath = r'{}/{}.csv'.format(set_name,num)
        content = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        if len(content.shape)==1:
            content = content[np.newaxis,:]
        data = content[:,1:]
        min_v = np.nanmin(data,axis=0)
        max_v = np.nanmax(data,axis=0)
        mean_v = np.nanmean(data,axis=0)
        statistics = np.zeros((3,data.shape[1]))

        statistics[0] = min_v
        statistics[1] = max_v
        statistics[2] = mean_v

        sample = np.nan_to_num(statistics)
        samples.append(sample)

    train_data = np.stack(samples,axis=0)
    tr_min_v = train_data[:,0,:].min(axis=0)
    tr_max_v = train_data[:,1,:].max(axis=0)



    set_name="test"
    #load test data
    lists = os.listdir(r'{}'.format(set_name))
    for i in range(len(lists)):
        lists[i] = int(lists[i][0:-4])
    lists.sort()

    samples = []

    for num in lists:

        filepath = r'{}/{}.csv'.format(set_name,num)
        content = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        if len(content.shape)==1:
            content = content[np.newaxis,:]
        data = content[:,1:]
        min_v = np.nanmin(data,axis=0)
        max_v = np.nanmax(data,axis=0)
        mean_v = np.nanmean(data,axis=0)
        statistics = np.zeros((3,data.shape[1]))

        statistics[0] = min_v
        statistics[1] = max_v
        statistics[2] = mean_v

        sample = np.nan_to_num(statistics)
        samples.append(sample)

    test_data = np.stack(samples,axis=0)



    #normalize
    s = tr_max_v - tr_min_v
    s = s + (s==0).astype(np.float)

    test_data = (test_data - tr_min_v)/s

    np.save('{}_data.npy'.format(set_name),test_data)
    
    
    

    #label read
    df = pd.read_csv("test_label.csv")
    c = df.values
    labels=c[:,1:4]

    np.save('test_label.npy',labels)

preprocess_test()
preprocess_train()

print('done')