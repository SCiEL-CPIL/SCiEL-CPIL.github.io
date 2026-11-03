import numpy as np
from itertools import product

def construct():
       D = 11
       
       # Type 1: one entry +/-2
       type1 = []
       for i in range(D):
           for s in (2, -2):
               v = np.zeros(D, dtype=np.int64)
               v[i] = s
               type1.append(v)
               
       # Type 3: four entries +/-1
       type3 = []
       for i in range(D):
           for j in range(i+1, D):
               for k in range(j+1, D):
                   for l in range(k+1, D):
                       for signs in product((1, -1), repeat=4):
                           v = np.zeros(D, dtype=np.int64)
                           v[i], v[j], v[k], v[l] = signs
                           type3.append(v)
                           
       type3_array = np.array(type3)
       
       best_selected = []
       
       # Try multiple random orders
       for trial in range(20):
           indices = np.random.permutation(len(type3))
           selected = []
           selected_array = np.empty((0, D), dtype=np.int64)
           
           for idx in indices:
               v = type3_array[idx]
               if len(selected_array) == 0:
                   selected.append(v)
                   selected_array = v.reshape(1, D)
               else:
                   # Check dot products
                   dots = selected_array @ v
                   if np.all(dots <= 2):
                       selected.append(v)
                       selected_array = np.vstack([selected_array, v])
                       
           if len(selected) > len(best_selected):
               best_selected = selected
               
       # Add Type 1 vectors
       # They are compatible with all Type 3
       final_vectors = best_selected + type1
       
       return np.array(final_vectors, dtype=np.int64)
