import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    # Using lists for speed and compatibility with the return type
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                U[i * m + j][column] = 1
                # V index: j * p + k
                V[j * p + k][column] = 1
                # W index: k * n + i
                W[k * n + i][column] = 1
                column += 1
                
                # Time check
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
