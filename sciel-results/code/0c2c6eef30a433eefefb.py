"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Exact rank-1 tensor decomposition for the all-ones tensor T[i,j,k]=1.
    # U, V, W are all-ones vectors of length 12, 16, 12 respectively.
    n, m, p = DIMENSIONS
    rank = 1
    U = [[1.0] * rank for _ in range(n * m)]
    V = [[1.0] * rank for _ in range(m * p)]
    W = [[1.0] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
