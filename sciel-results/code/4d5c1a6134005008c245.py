"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                U[i * m + j][c] = 1
        for j in range(m):
            for k in range(p):
                V[j * p + k][c] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i][c] = 1 / 2
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
