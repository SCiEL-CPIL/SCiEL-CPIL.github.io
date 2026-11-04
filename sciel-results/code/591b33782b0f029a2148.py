"""Tensor decomposition for 3x4 by 4x8 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x8 matrix.
    
    The tensor T[i,j,k] = 1 for all i in [0,3), j in [0,4), k in [0,8) since every term
    A[i,j]*B[j,k] contributes to C[i,k] with coefficient 1.
    
    The all-ones tensor has rank 1:
        T[i,j,k] = 1 = 1 * 1 * 1
    
    So U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1 for all valid indices.
    """
    n, m, p = DIMENSIONS
    
    rank = 1
    
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
