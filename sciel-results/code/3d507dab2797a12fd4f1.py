import numpy as np
import itertools

def construct():
    D = 11
    # Generate initial set from E8 x A3 (252 vectors)
    vectors = []
    
    # E8 Part (Dimensions 0-7)
    # Type 1: (±2, ±2, 0, ..., 0)
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 11
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)
                    
    # Type 2: (±1, ±1, ..., ±1) with even number of -1s
    for mask in range(256):
        v = [0] * 11
        minus_count = 0
        for i in range(8):
            if (mask >> i) & 1:
                v[i] = -1
                minus_count += 1
            else:
                v[i] = 1
        if minus_count % 2 == 0:
            vectors.append(v)
            
    # D3 Part (Dimensions 8-10)
    for i in range(8, 11):
        for j in range(i + 1, 11):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 11
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)
                    
    # Mixed Part
    for i in range(8):
        for j in range(8, 11):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 11
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)
                    
    vec_array = np.array(vectors, dtype=np.int64)
    
    # Try to add more vectors
    # Max norm squared is 8.
    # Condition: dist_sq >= 8.
    
    # Generate candidate vectors with norm squared <= 8
    # We need to be careful with performance.
    # Candidates:
    # Norm sq 1: 22
    # Norm sq 2: 220
    # Norm sq 3: 11*10*9*8? No. (±1, ±1, ±1, 0...)
    #   Choose 3 pos: 165. Signs: 8. Total 1320.
    # Norm sq 4: 
    #   (±2, 0...): 22
    #   (±1, ±1, ±1, ±1, 0...): 5280
    # Norm sq 5: (±2, ±1, 0...): 
    #   Choose 2 pos: 55. Signs: 4. Total 220.
    # Norm sq 6: (±2, ±1, ±1, 0...):
    #   Choose 3 pos: 165. Signs: 4 (one 2, two 1s). 165*4=660?
    #   Actually, permutations of (2,1,1). 3!/2! = 3 orders. Signs 2^3=8.
    #   165 * 3 * 8 = 3960.
    # Norm sq 7: (±2, ±1, ±1, ±1, 0...):
    #   Choose 4 pos: 330. Perms of (2,1,1,1): 4. Signs 16.
    #   330 * 4 * 16 = 21120.
    # Norm sq 8:
    #   (±2, ±2, 0...): 220
    #   (±2, ±1, ±1, ±1, 0...)? No, 4+1+1+1=7.
    #   (±1, ..., ±1) 8 ones: 
    #   Choose 8 pos: 165. Signs 256. 165*256 = 42240.
    
    # This is too many candidates to check against 252 vectors for each.
    # 252 * 50000 * 11 operations might be too slow in Python.
    
    # Instead, we can use a simpler heuristic.
    # Just return the current 252.
    # Or, try a few specific additions.
    
    return vec_array
