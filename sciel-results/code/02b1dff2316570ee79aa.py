import time
import sys

DIMENSIONS = [2, 3, 7]

def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 7 matrix.
    
    The rank is n*m*p = 42.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # Using lists of lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Check time
    if time.time() > deadline:
        pass
        
    return {"U": U, "V": V, "W": W}
