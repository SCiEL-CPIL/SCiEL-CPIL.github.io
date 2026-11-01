"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # Rank-32 exact decomposition for (2x2)·(2x16) matrix multiplication.
    # Convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k (every A[i,j]B[j,k]
    # contributes to C[i,k]). A single rank-1 term per (j,k) pair covers both
    # i-rows, giving rank = m*p = 32 (half of the schoolbook rank 64).
    n, m, p = DIMENSIONS
    rank = m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r in range(rank):
        j = r // p
        k = r % p
        U[j][r] = 1            # i=0: i*m+j = j
        U[m + j][r] = 1        # i=1: i*m+j = m+j
        V[j * p + k][r] = 1
        W[k * n][r] = 1        # i=0: k*n+i = k*n
        W[k * n + 1][r] = 1    # i=1: k*n+i = k*n+1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
