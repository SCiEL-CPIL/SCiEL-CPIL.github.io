"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # can be decomposed as a single rank-1 tensor:
    # T[i,j,k] = U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0]
    # with all entries equal to 1.
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
