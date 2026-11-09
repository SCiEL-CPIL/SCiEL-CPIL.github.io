import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x5 matrix by a 5x6 matrix.
    Currently returns the schoolbook decomposition of rank 60.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Schoolbook term: U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
                # Check deadline
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Verify the solution is exact (optional but good for robustness)
    # The schoolbook solution is always exact.
    
    return {"U": U, "V": V, "W": W}
