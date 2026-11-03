import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook tensor decomposition for 5x5x5 matrix multiplication.
    Rank = 125.
    U shape: (25, 125)
    V shape: (25, 125)
    W shape: (25, 125)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Fill the matrices based on the schoolbook algorithm
    # U[i*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + i, col] = 1
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # The evaluator expects a dictionary with keys 'U', 'V', 'W'
    # It mentions "integer/half-integer entries". Floats 1.0 and 0.0 are fine.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
