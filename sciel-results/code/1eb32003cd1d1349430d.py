"""
Tensor decomposition for 2x4 by 4x4 matrix multiplication.
Implements the known rank-32 bilinear decomposition.
"""
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = n * m * p  # 32

    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 32
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 32
    W = [[0.0] * rank for _ in range(p * n)]  # 8 x 32

    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
