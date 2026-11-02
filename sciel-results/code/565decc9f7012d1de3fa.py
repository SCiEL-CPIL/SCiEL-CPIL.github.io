import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists to match expected interface
    # Shapes: (n*m, rank), (m*p, rank), (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
