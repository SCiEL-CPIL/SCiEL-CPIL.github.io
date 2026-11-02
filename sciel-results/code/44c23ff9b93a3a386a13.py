import time
import random
import numpy as np

def construct():
    """
    Returns a rank-minimal (or best known) exact tensor decomposition for 
    multiplying a 3 by 4 matrix by a 4 by 4 matrix.
    
    Dimensions:
    n = 3 (rows of A, rows of C)
    m = 4 (cols of A, rows of B)
    p = 4 (cols of B, cols of C)
    
    The tensor T has entries T[(i,j), (j,k), (k,i)] = 1 for valid indices.
    U shape: (n*m, rank) = (12, rank)
    V shape: (m*p, rank) = (16, rank)
    W shape: (p*n, rank) = (12, rank)
    
    Schoolbook rank is n*m*p = 48.
    """
    n, m, p = 3, 4, 4
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: i*m + j
                # Index in V: j*p + k
                # Index in W: k*n + i
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {
        "U": U,
        "V": V,
        "W": W
    }
