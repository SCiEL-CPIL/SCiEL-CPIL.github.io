"""Program for 3x5 by 5x5 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    idx = 0
    for k in range(p):
        for j in range(m):
            for i in range(n):
                U[i * m + j][idx] = 1.0
                V[j * p + k][idx] = 1.0
                W[k * n + i][idx] = 1.0
                idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
