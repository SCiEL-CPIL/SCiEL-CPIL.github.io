import numpy as np
import itertools
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 140s limit
    
    D = 11
    
    vectors = []
    
    # Generate 2 * E8 minimum norm vectors (first 8 dims)
    # E8 min norm vectors are of two types:
    # 1. Two +/-1, rest 0. (112 vectors)
    # 2. All +/-0.5, even number of minuses. (128 vectors)
    # Scaled by 2:
    # 1. Two +/-2, rest 0.
    # 2. All +/-1, even number of minuses.
    
    # Type 1: Two +/-2
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
                
    if time.time() > deadline:
        return np.array(vectors, dtype=np.int64)

    # Type 2: All +/-1, even number of minuses
    # 2^8 = 256 combinations. Half have even number of minuses.
    for mask in range(256):
        # Count minuses
        minuses = 0
        v = [0] * D
        for i in range(8):
            if (mask >> i) & 1:
                v[i] = -1
                minuses += 1
            else:
                v[i] = 1
        if minuses % 2 == 0:
            vectors.append(v)
            
    if time.time() > deadline:
        return np.array(vectors, dtype=np.int64)

    # Generate 2 * D3 minimum norm vectors (last 3 dims)
    # D3 min norm vectors: Two +/-1, one 0. (12 vectors)
    # Scaled by 2: Two +/-2, one 0.
    
    for i, j in itertools.combinations(range(3), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[8 + i] = 2 * si
                v[8 + j] = 2 * sj
                vectors.append(v)

    return np.array(vectors, dtype=np.int64)
