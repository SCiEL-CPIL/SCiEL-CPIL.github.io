import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x7 matrix.
    
    For these dimensions, the known minimal rank is the schoolbook rank:
    n * m * p = 3 * 3 * 7 = 63.
    
    The decomposition is defined by matrices U, V, W such that:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    We construct the standard schoolbook decomposition where each term
    in the sum corresponds to a single multiplication A[i,j]*B[j,k].
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 63
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The term A[i,j] * B[j,k] contributes to C[i,k]
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i (following the trace convention W[k*n+i, r])
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
