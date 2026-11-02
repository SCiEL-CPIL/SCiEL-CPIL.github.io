import numpy as np
import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # M_{2,2,1} has rank 4 (proven minimal).
    # The 2x2 * 2x13 multiplication tensor is a direct sum of 13 copies of M_{2,2,1}.
    # Total rank = 13 * 4 = 52.
    rank = n * m * p  # = 2 * 2 * 13 = 52
    
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):       # i = 0,1 (row of A, row of C)
        for j in range(m):   # j = 0,1 (col of A, row of B)
            for k in range(p): # k = 0..12 (col of B, col of C)
                a_idx = i * m + j       # position in A (row-major)
                b_idx = j * p + k       # position in B (row-major)
                c_idx = k * n + i       # position in C per trace convention W[k*n+i, r]
                U[a_idx, col] = 1
                V[b_idx, col] = 1
                W[c_idx, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
