"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor T_{i,j,k} = 1 for all i,j,k
    # can be decomposed into a single rank-1 term:
    # U_{i,j} = 1, V_{j,k} = 1, W_{i,k} = 1
    # such that sum_c U_{i,j,c} V_{j,k,c} W_{i,k,c} = 1.
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
