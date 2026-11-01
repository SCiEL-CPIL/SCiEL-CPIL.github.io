import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Generate all integer vectors with exactly two non-zero entries of value +/- 1
    # These have squared norm 2.
    vectors = []
    
    # Iterate over all pairs of indices
    for i in range(D):
        for j in range(i + 1, D):
            # Signs for i and j
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
                    
                    # Check time
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    if not vectors:
        # Fallback to a single vector if something went wrong or time out immediately
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        return np.array([v])
        
    return np.array(vectors, dtype=np.int64)
