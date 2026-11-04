import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Constructs a tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    Uses schoolbook for the 3x3x3 block (rank 9) and 3x3x4 block (rank 36).
    Total rank: 9 + 36 = 45.
    """
    start_time = time.time()
    deadline = 110.0
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    # U shape: (n*m, rank) = (9, 45)
    # V shape: (m*p, rank) = (21, 45)
    # W shape: (p*n, rank) = (21, 45)
    
    rank = 45
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: Columns 0, 1, 2 of B (3x3x3)
    # Terms for each i, j, k in 0..2
    for i in range(n):
        for j in range(m):
            for k in range(3): # k is 0,1,2
                # U[i*m + j, col] = 1
                # V[j*p + k, col] = 1
                # W[k*n + i, col] = 1
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Block 2: Columns 3, 4, 5, 6 of B (3x3x4)
    # Terms for each i, j, k in 3..6
    for i in range(n):
        for j in range(m):
            for k in range(3, p): # k is 3,4,5,6
                # U[i*m + j, col] = 1
                # V[j*p + k, col] = 1
                # W[k*n + i, col] = 1
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
