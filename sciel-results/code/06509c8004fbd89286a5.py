import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115

    vectors = []
    
    # Generate all vectors of the form +/- e_i +/- e_j for i < j
    # These have squared norm 2.
    # There are 11*10 = 110 such vectors without the global sign flip?
    # Actually, for each pair (i, j) with i < j, we have:
    # e_i + e_j
    # e_i - e_j
    # -e_i + e_j
    # -e_i - e_j
    # Total 4 per pair.
    # Number of pairs is C(11, 2) = 55.
    # Total vectors = 55 * 4 = 220.
    
    for i in range(D):
        for j in range(i + 1, D):
            # e_i + e_j
            v = [0] * D
            v[i] = 1
            v[j] = 1
            vectors.append(v)
            
            # e_i - e_j
            v = [0] * D
            v[i] = 1
            v[j] = -1
            vectors.append(v)
            
            # -e_i + e_j
            v = [0] * D
            v[i] = -1
            v[j] = 1
            vectors.append(v)
            
            # -e_i - e_j
            v = [0] * D
            v[i] = -1
            v[j] = -1
            vectors.append(v)
            
    vecs = np.array(vectors, dtype=np.int64)
    
    return vecs

# EVOLVE-BLOCK-END
