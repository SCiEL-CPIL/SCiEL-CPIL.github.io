import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    vectors = []
    
    # Generate all vectors of the form e_i + e_j, e_i - e_j, -e_i + e_j, -e_i - e_j
    # These are the roots of the D_11 lattice.
    # There are 4 * C(11, 2) = 220 such vectors.
    
    for i, j in itertools.combinations(range(D), 2):
        # si, sj in {1, -1}
        # v[i] = si, v[j] = sj
        for si in (1, -1):
            for sj in (1, -1):
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
        return np.array([], dtype=np.int64).reshape(0, D)
        
    return np.array(vectors, dtype=np.int64)
