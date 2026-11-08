"""Bootstrap direct construction of a low-rank decomposition for 3x3 * 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
DEADLINE = 150.0

def construct():
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54

    # Initialize U, V, W with zeros
    # U: (n*m, rank) = (9, 54)
    # V: (m*p, rank) = (18, 54)
    # W: (p*n, rank) = (18, 54)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
