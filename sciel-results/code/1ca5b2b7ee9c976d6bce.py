"""Rank-minimal exact tensor decomposition for 2x5 @ 5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    # Use block structure: rank = n * m = 10
    # Components c = i_c * m + j_c for i_c in range(n), j_c in range(m)
    rank = n * m  # 10
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i_c in range(n):
        for j_c in range(m):
            c = i_c * m + j_c
            # U[i*m + j, c] = 1 if i == i_c and j == j_c
            U[i_c * m + j_c][c] = 1
            # V[j*p + k, c] = 1 if j == j_c (for any k)
            for k in range(p):
                V[j_c * p + k][c] = 1
            # W[k*n + i, c] = 1 if i == i_c (for any k)
            for k in range(p):
                W[k * n + i_c][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
