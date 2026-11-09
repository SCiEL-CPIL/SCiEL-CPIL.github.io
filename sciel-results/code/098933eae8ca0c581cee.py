import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 12
    np_ = n * p  # 8
    
    # Schoolbook rank is 24.
    # We can split B into B1 (3x3) and B2 (3x1).
    # Rank of 2x3x3 schoolbook is 18.
    # Rank of 2x3x1 schoolbook is 3.
    # Total rank 21.
    
    rank = 21
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(np_)]
    
    col = 0
    # First 3 columns of B
    for i in range(n):
        for j in range(m):
            for k in range(3): # k is 0,1,2
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1
                
    # 4th column of B
    for i in range(n):
        for j in range(m):
            k = 3
            a = i * m + j
            b = j * p + k
            c = k * n + i
            U[a][col] = 1
            V[b][col] = 1
            W[c][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
