import numpy as np
import itertools

D = 11

def construct():
    vectors = []
    
    # Type A: One entry is +/- 2
    for i in range(D):
        for s in (2, -2):
            v = [0] * D
            v[i] = s
            vectors.append(v)
            
    # Type B: Two entries are +/- 1
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
                
    return np.array(vectors, dtype=np.int64)
