import numpy as np
import itertools

D = 11

def construct():
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = [0] * D
                v[i] = s1
                v[j] = s2
                vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
