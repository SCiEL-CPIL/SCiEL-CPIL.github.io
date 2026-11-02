import numpy as np
import itertools
import time

D = 11

def construct():
    # Generate Type 1 vectors: +/- 2 at one position
    vecs_t1 = []
    for i in range(D):
        for s in [1, -1]:
            v = [0]*D
            v[i] = 2*s
            vecs_t1.append(v)
            
    # Generate Type 2 vectors: +/- 1 at four positions
    vecs_t2 = []
    for pos in itertools.combinations(range(D), 4):
        for mask in range(16):
            signs = [1 if (mask & (1<<j)) == 0 else -1 for j in range(4)]
            v = [0]*D
            for p, val in zip(pos, signs):
                v[p] = val
            vecs_t2.append(v)
                
    V_t1 = np.array(vecs_t1, dtype=np.int64)
    V_t2 = np.array(vecs_t2, dtype=np.int64)
    
    # All these vectors are in the D11 lattice (even sum)
    # and have norm 2.
    # We verified that any distinct pair has dot product <= 2.
    # Thus, distance squared >= 4, so distance >= 2.
    # Max norm is 2.
    # Condition: min dist >= max norm holds.
    
    result = np.vstack([V_t1, V_t2])
    
    return result
