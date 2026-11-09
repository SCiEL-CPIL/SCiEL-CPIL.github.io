"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for 2x3 by 3x9 matrix multiplication.
    
    Uses the schoolbook decomposition with rank 54. The decomposition follows
    the trace convention W[k*n+i, r] where C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    
    For each term (i, j, k), we have a rank-1 component where:
    - U[i*m + j, r] = 1
    - V[j*p + k, r] = 1  
    - W[k*n + i, r] = 1
    
    This gives C[i, k] = sum_j A[i,j] * B[j,k] as required.
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
