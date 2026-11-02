import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication 2x5 * 5x5.
    Rank is 50.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists of floats (0.0)
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1.0
                # V index: j*p + k
                V[j * p + k][col] = 1.0
                # W index: k*n + i
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
