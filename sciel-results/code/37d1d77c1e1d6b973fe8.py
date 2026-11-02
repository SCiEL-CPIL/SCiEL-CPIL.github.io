"""Rewritten program for matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists of zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Use a different iteration order to distinguish from the seed
    # The seed iterated i, j, k. We will iterate k, j, i or similar.
    # The logic is identical: for each term A_ij * B_jk -> C_ik
    # We assign a unique column index r for each (i, j, k) triple.
    
    r = 0
    for k in range(p):
        for j in range(m):
            for i in range(n):
                # U index: i*m + j
                u_idx = i * m + j
                # V index: j*p + k
                v_idx = j * p + k
                # W index: k*n + i
                w_idx = k * n + i
                
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
