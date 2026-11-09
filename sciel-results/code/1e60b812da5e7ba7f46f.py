"""Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = n * m * p  # 96
    U = [[0] * rank for _ in range(n * m)]  # shape (12, 96)
    V = [[0] * rank for _ in range(m * p)]  # shape (32, 96)
    W = [[0] * rank for _ in range(p * n)]  # shape (24, 96)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
