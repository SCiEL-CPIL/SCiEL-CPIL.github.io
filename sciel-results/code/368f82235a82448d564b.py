"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = p  # tensor rank of all-ones tensor (n,m,p) is max(n,m,p) = 6
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for k in range(p):
        for i in range(n):
            W[k * n + i][k] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
