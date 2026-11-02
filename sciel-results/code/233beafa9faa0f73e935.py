"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Uses the schoolbook (canonical) decomposition with rank = n*m*p = 80.
    For small matrix dimensions, the schoolbook rank is often near-optimal.
    
    The decomposition satisfies:
      C[i,k] = sum_j A[i,j] * B[j,k]
    
    Using the trace convention W[k*n+i, r], U[i*m+j, r], V[j*p+k, r]:
      C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For the schoolbook, each rank-1 term corresponds to a single (i,j,k) triple
    where U, V, W have a single 1 at the appropriate position.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
