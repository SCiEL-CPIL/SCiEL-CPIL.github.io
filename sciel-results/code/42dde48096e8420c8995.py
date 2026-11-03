"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a rank-216 exact tensor decomposition for 6x6 matrix multiplication.
    This is the schoolbook algorithm.
    """
    start_time = time.time()
    timeout = 120.0
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices as lists of lists for compatibility, 
    # though numpy might be faster for verification if needed.
    # The interface expects {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    # Safety check: ensure we are within time limits
    if time.time() - start_time > timeout * 0.8:
         # This block is trivial, so it shouldn't hit this, but good practice.
         pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
