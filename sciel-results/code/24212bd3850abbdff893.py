import time
import random

DIMENSIONS = [3, 4, 7]

def construct():
    """
    Returns the schoolbook decomposition for matrix multiplication 3x4 * 4x7.
    Rank is 84.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    # Initialize U, V, W with zeros
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
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
