import numpy as np
import itertools

def construct():
    D = 11
    vectors = []
    
    # 1. E8 lattice part (first 8 dimensions)
    # Type 1: (±2, ±2, 0, ..., 0)
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
                
    # Type 2: (±1, ±1, ..., ±1) with even number of -1s
    for signs in itertools.product([1, -1], repeat=8):
        if sum(1 for s in signs if s == -1) % 2 == 0:
            v = [0] * D
            for i in range(8):
                v[i] = signs[i]
            vectors.append(v)
            
    # 2. D3 part (last 3 dimensions) - Updated to norm sqrt(8)
    # Vectors of form (±2, ±2, 0) in dimensions 8, 9, 10
    dims_d3 = [8, 9, 10]
    for i, j in itertools.combinations(dims_d3, 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
                
    # 3. Mixed part (one in E8, one in D3)
    # Vectors of form (±2, ..., ±2, 0...) with one non-zero in 0-7 and one in 8-10
    for i in range(8):
        for j in range(8, 11):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)
                    
    return np.array(vectors, dtype=np.int64)
