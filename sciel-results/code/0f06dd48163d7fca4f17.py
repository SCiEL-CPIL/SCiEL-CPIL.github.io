import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = 2, 3, 9
    # Schoolbook rank
    R = n * m * p
    
    # Build U, V, W as lists of lists of ints
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # r = i * m * p + j * p + k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
