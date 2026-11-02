import numpy as np

def construct():
       D = 11
       vectors = []
       for i in range(D):
           for j in range(i+1, D):
               for s1 in [1, -1]:
                   for s2 in [1, -1]:
                       v = [0] * D
                       v[i] = s1
                       v[j] = s2
                       vectors.append(v)
       
       arr = np.array(vectors, dtype=np.int64)
       return arr
