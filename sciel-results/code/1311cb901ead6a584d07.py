"""Tensor decomposition for 3x4x8 matrix multiplication."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * p) + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
