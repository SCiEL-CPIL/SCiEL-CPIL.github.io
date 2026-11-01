"""Explicit iterative decomposition of 2x2 by 2x3 matrix multiplication tensor."""
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists of zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Determine row indices for U, V, W
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                
                U[u_row][t] = 1.0
                V[v_row][t] = 1.0
                W[w_row][t] = 1.0
                t += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
