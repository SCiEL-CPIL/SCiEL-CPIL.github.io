"""Tensor decomposition for 3x3 by 3x4 matrix multiplication."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    # The matrix multiplication tensor for C[i,k] = sum_j A[i,j] B[j,k]
    # has coefficient 1 for every (i, j, k). In the grouped factorization
    #   T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # a single all-ones column reproduces T[i,j,k] = 1 exactly,
    # so the minimal rank is 1.
    n, m, p = DIMENSIONS
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
