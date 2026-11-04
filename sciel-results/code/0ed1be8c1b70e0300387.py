"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Constructs a kissing configuration in R^11 using the direct sum of E8 and D3 lattices.
    
    E8 has 240 minimal vectors of norm sqrt(2).
    D3 (isomorphic to A3) has 12 minimal vectors of norm sqrt(2).
    The direct sum E8 x D3 is a lattice in R^11.
    The minimal vectors of E8 x D3 are:
    1. (v, 0) where v is a minimal vector of E8. (240 vectors)
    2. (0, w) where w is a minimal vector of D3. (12 vectors)
    
    Total vectors: 252.
    Norm of each vector is sqrt(2).
    Minimum distance within E8 part is sqrt(2).
    Minimum distance within D3 part is sqrt(2).
    Distance between E8 part and D3 part is sqrt(2^2 + 2^2) = 2.
    Since 2 > sqrt(2), the minimum pairwise distance is sqrt(2).
    The maximum norm is sqrt(2).
    Condition: min_dist >= max_norm is satisfied (sqrt(2) >= sqrt(2)).
    
    We return integer vectors by scaling? 
    The task asks for integer vectors.
    E8 minimal vectors are integers (Type 1: +/-1, +/-1, 0...; Type 2: +/-1/2... wait).
    Standard E8 minimal vectors:
    1. Permutations of (+/-1, +/-1, 0, ..., 0). These are integers.
    2. Permutations of (+/-1/2, ..., +/-1/2) with even number of minuses. These are NOT integers.
    
    The current program only used Type 1?
    Let's check the current program.
    Type 1: (±2, ±2, 0...) scaled by 2? 
    The current program says: "Scaled by 2... (±2, ±2, 0...)".
    And "Type 2... Scaled by 2... (±1, ±1, ...)".
    So the current program uses scaled vectors.
    If we scale E8 minimal vectors by 2:
    Type 1 becomes (±2, ±2, 0...). Norm sqrt(8).
    Type 2 becomes (±1, ±1, ...). Norm sqrt(8).
    
    If we use E8 x D3:
    We should scale the whole lattice by 2 to ensure integers if the base lattice has half-integers.
    E8 has half-integers. D3 has integers.
    So scaling by 2 makes everything integers.
    
    Vectors in E8 part (scaled by 2):
    1. (±2, ±2, 0, ..., 0) in first 8 dims. (240/2? No, 240 vectors in E8. 
       Wait, E8 minimal vectors: 112 of Type 1, 128 of Type 2. Total 240.
       Type 1: 8*7*4 = 224? No.
       Permutations of (±1, ±1, 0...): Choose 2 pos (28), 4 signs (4). 28*4=112.
       Type 2: (±1/2...): 2^7 = 128.
       Total 240.
       
       Scaled by 2:
       Type 1: (±2, ±2, 0...). 112 vectors.
       Type 2: (±1, ±1, ...). 128 vectors.
       
    Vectors in D3 part (scaled by 2):
    D3 minimal vectors: (±1, ±1, 0) permutations. 12 vectors.
    Scaled by 2: (±2, ±2, 0) permutations. 12 vectors.
    
    Total 112 + 128 + 12 = 252 vectors.
    
    Norm of all these vectors:
    E8 Type 1 scaled: sqrt(4+4) = sqrt(8).
    E8 Type 2 scaled: sqrt(1+...+1) = sqrt(8).
    D3 scaled: sqrt(4+4) = sqrt(8).
    
    Max norm is sqrt(8).
    Min distance:
    Within E8 Type 1: Min dist sqrt(8)?
    Original min dist sqrt(2). Scaled by 2 -> sqrt(8).
    So min dist is sqrt(8).
    Condition: sqrt(8) >= sqrt(8). OK.
    """
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
    # Minimal vectors of D3: (±1, ±1, 0) permutations.
    # Scaled by 2: (±2, ±2, 0) permutations.
    # Positions in 11D: 8, 9, 10.
    for i in range(8, 11):
        for j in range(i + 1, 11):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 11
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)
                    
    return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
