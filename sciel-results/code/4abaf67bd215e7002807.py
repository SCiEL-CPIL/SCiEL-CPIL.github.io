import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = 120.0
    
    vectors = []
    # Generate all vectors with exactly two non-zero entries of value +/-1
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    if time.time() > deadline:
                        break
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
        
    if not vectors:
        return np.empty((0, D), dtype=np.int64)
        
    return np.array(vectors, dtype=np.int64)
