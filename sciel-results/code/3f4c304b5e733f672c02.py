"""Program for matrix multiplication 2x5 x 5x6 tensor decomposition."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r in range(m):
        for i in range(n):
            U[i * m + r][r] = 1
        for k in range(p):
            V[r * p + k][r] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
