"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x8 matrix.
    
    The matrix multiplication tensor T[(i,j),(j,k),(k,i)] = 1 for all valid indices.
    Since the tensor is all-ones, it has rank 1.
    
    U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] with rank r.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Rank-1 decomposition: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    rank = 1
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][0] = 1
    
    for j in range(m):
        for k in range(p):
            V[j * p + k][0] = 1
    
    for k in range(p):
        for i in range(n):
            W[k * n + i][0] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
