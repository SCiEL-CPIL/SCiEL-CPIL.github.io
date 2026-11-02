import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Rank is m * p = 3 * 9 = 27
    rank = m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        for j in range(m):
            r = col
            
            # U[i*m + j, r] = 1 for all i
            for i in range(n):
                U[i * m + j][r] = 1
                
            # V[j*p + k, r] = 1
            V[j * p + k][r] = 1
            
            # W[k*n + i, r] = 1 for all i
            for i in range(n):
                W[k * n + i][r] = 1
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
