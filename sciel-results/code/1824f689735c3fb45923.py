import numpy as np
import itertools

def construct():
    D = 11
    vectors = []
    # Generate all vectors of the form +/- e_i +/- e_j
    # These have squared norm 2.
    # Minimum squared distance between any two distinct such vectors is 2.
    # Max squared norm is 2.
    # Condition: min_dist >= max_norm => sqrt(2) >= sqrt(2). True.
    
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
                
    return np.array(vectors, dtype=np.int64)
