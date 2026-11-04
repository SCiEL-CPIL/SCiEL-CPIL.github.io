import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                if time.time() > deadline:
                    break
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
        if time.time() > deadline:
            break
    if time.time() > deadline:
        break
        
    return np.array(vectors, dtype=np.int64)
