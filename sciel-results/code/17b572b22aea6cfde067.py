"""Program for 4x4x9 matrix multiplication rank decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Construct a tensor decomposition for 4x4x9 matrix multiplication.
    
    Uses schoolbook decomposition with rank = n*m*p = 144.
    Each rank-1 term corresponds to one (i,j,k) triple where:
      U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
    
    The output factor W follows the trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
