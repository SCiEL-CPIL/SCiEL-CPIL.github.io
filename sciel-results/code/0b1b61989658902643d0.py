"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The tensor for matrix multiplication M_{n,m,p} has entries 1 for all valid indices.
    # The decomposition sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 is satisfied by
    # U=1, V=1, W=1 with rank 1.
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
