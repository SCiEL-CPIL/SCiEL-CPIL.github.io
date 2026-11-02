"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The tensor for matrix multiplication C = AB where C_ik = sum_j A_ij B_jk
    # has entries T_ijk = 1 for all i,j,k in the respective ranges.
    # A rank-1 decomposition is given by U=1, V=1, W=1.
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
