"""Rank-1 decomposition for the all-ones 2x3x5 tensor."""
DIMENSIONS = [2, 3, 5]
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The tensor T[i,j,k] = 1 for all i,j,k is the all-ones tensor,
    # which is a single rank-1 tensor: 1_n ⊗ 1_m ⊗ 1_p.
    # U[i*m+j, 0] = 1 for all (i,j)
    # V[j*p+k, 0] = 1 for all (j,k)
    # W[k*n+i, 0] = 1 for all (k,i)
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
