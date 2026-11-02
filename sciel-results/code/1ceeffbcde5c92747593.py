import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    """
    Returns the schoolbook decomposition for matrix multiplication.
    Rank is n * m * p.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists of zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, column] = 1
                U[i * m + j][column] = 1
                # V[j*p + k, column] = 1
                V[j * p + k][column] = 1
                # W[k*n + i, column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    # Check deadline, though this loop is very fast
    if time.time() > deadline:
        # Return best found (which is the full schoolbook)
        pass
        
    return {"U": U, "V": V, "W": W}
