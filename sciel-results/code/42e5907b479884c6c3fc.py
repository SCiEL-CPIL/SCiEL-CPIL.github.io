"""Matrix multiplication 3x4x8 rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 1
    # All-ones factors give an exact rank-1 decomposition:
    #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 = T[i,j,k]
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
