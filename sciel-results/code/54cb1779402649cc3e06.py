import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    vectors = []
    
    # Type 3: Two 2s, rest 0: ±2e_i ± 2e_j
    # Norm squared = 8
    # All such vectors are mutually compatible.
    # Count: C(11, 2) * 4 = 55 * 4 = 220
    
    for i, j in itertools.combinations(range(D), 2):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = [0] * D
                v[i] = 2 * s1
                v[j] = 2 * s2
                vectors.append(v)
                
    if time.time() > deadline:
        # Fallback, though this is fast
        pass

    return np.array(vectors, dtype=np.int64)
