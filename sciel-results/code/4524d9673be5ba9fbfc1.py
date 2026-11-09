"""Matrix multiplication tensor decomposition for 2x4 * 4x6."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    #
    # The matrix multiplication tensor T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    #
    # Schoolbook decomposition with rank = m:
    # For each r in 0..m-1 (corresponding to summation index j=r):
    #   U[i*m+j, r] = 1 if j == r, else 0
    #   V[j*p+k, r] = 1 if j == r, else 0
    #   W[k*n+i, r] = 1 for all k,i
    #
    # Verification: For fixed i,j,k, only r=j contributes:
    #   U[i*m+j, j] * V[j*p+k, j] * W[k*n+i, j] = 1*1*1 = 1. Correct.

    rank = m
    U = np.zeros((n*m, rank), dtype=np.int32)
    V = np.zeros((m*p, rank), dtype=np.int32)
    W = np.ones((p*n, rank), dtype=np.int32)

    for j in range(m):
        for i in range(n):
            U[i * m + j, j] = 1
        for k in range(p):
            V[j * p + k, j] = 1
        # W is already all 1s

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
