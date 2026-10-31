import numpy as np
import itertools

def construct():
    D = 11
    vectors = []
    
    # Part 1: Scaled D11 minimal vectors
    # Two non-zeros +/- 2 in any 2 of 11 dimensions
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
                
    # Part 2: E8 Type 2 vectors (embedded in first 8 dims)
    # Eight non-zeros +/- 1 in first 8 dims, even number of -1s
    for mask in range(256):
        v = [0] * D
        minus_count = 0
        for k in range(8):
            if (mask >> k) & 1:
                v[k] = -1
                minus_count += 1
            else:
                v[k] = 1
        if minus_count % 2 == 0:
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
