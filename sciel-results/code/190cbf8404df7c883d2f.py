import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=7
    # U: (9, rank)
    # V: (21, rank)
    # W: (21, rank)
    
    rank_total = 39
    U = np.zeros((n * m, rank_total), dtype=np.float64)
    V = np.zeros((m * p, rank_total), dtype=np.float64)
    W = np.zeros((p * n, rank_total), dtype=np.float64)
    
    # Part 1: 3x3x3 (k=0,1,2) using schoolbook (rank 27)
    # Indices: i in 0..2, j in 0..2, k in 0..2
    d = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, d] = 1.0
                V[b, d] = 1.0
                W[c, d] = 1.0
                d += 1
    # d is now 27
    
    # Part 2: 3x3x4 (k=3,4,5,6) using schoolbook (rank 12)
    # Indices: i in 0..2, j in 0..2, k in 3..6
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, d] = 1.0
                V[b, d] = 1.0
                W[c, d] = 1.0
                d += 1
    # d is now 39
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
