"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank = m * p: one term per (j, k) pair, handling both i values
    rank = m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for j in range(m):
        for k in range(p):
            r = j * p + k
            for i in range(n):
                U[i * m + j][r] = 1
                W[k * n + i][r] = 1
            V[j * p + k][r] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
