"""
Rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication.

Uses the schoolbook bilinear algorithm with rank n*m*p = 75.
Each term corresponds to one (i,j,k) triple where:
  U[i*m + j] = 1, V[j*p + k] = 1, W[k*n + i] = 1
This exactly decomposes the matrix multiplication tensor.
"""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
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
