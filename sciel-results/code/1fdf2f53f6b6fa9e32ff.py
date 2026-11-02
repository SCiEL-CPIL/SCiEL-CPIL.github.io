import numpy as np
import itertools

D = 11

def construct():
    # Construct the roots of the D_11 lattice
    # These are vectors with exactly two non-zero entries of +/- 1
    # Number of such vectors: C(11, 2) * 2^2 = 55 * 4 = 220
    
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
                
    result = np.array(vectors, dtype=np.int64)
    
    # Verify validity just in case (should be 220 vectors, all norm^2 = 2)
    # Min distance squared should be >= Max norm squared (which is 2)
    # For D_n roots, min dist^2 is 2 (e.g., e_i+e_j and e_i-e_k? No, e_i+e_j and e_i+e_k dist^2 = 2. e_i+e_j and e_i-e_j dist^2 = 4.)
    # Wait, e_i+e_j and e_i+e_k (i!=j, i!=k, j!=k).
    # v1 = (1, 1, 0, 0), v2 = (1, 0, 1, 0).
    # diff = (0, 1, -1, 0). norm^2 = 2.
    # Max norm^2 is 2.
    # Condition: min dist^2 >= max norm^2 => 2 >= 2. Valid.
    
    return result
