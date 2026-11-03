import numpy as np
import itertools

D = 11

def construct():
    vectors = []
    
    # Part 1: E8 lattice minimal vectors (scaled by 2) in first 8 dimensions
    # Type 1: 112 vectors with two +/-2 and rest 0
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
                
    # Type 2: 128 vectors with eight +/-1 (even number of -1s)
    # Signs are +1 or -1. 
    # Condition: even number of -1s in the original half-integer representation.
    # Original: x_i in {0, -1}. -1 corresponds to -1/2.
    # So we need even number of -1/2s.
    # In scaled integer version, -1/2 becomes -1.
    # So we need even number of -1s in the vector.
    for signs in itertools.product([1, -1], repeat=8):
        if signs.count(-1) % 2 == 0:
            v = [0] * D
            for i in range(8):
                v[i] = signs[i]
            vectors.append(v)
            
    # Part 2: Kissing configuration in last 3 dimensions (Norm sqrt(2) vectors)
    # 12 vectors: pairs (8,9), (8,10), (9,10) with 4 signs each
    tail_indices = [8, 9, 10]
    for i, j in itertools.combinations(tail_indices, 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
                
    return np.array(vectors, dtype=np.int64)
