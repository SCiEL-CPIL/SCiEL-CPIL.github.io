import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """Return rank-minimal exact tensor decomposition for (2,5,5) matrix multiplication.
    
    The output factor follows the official trace convention W[k*n+i, r].
    Returns dict with 'U' (n*m x rank), 'V' (m*p x rank), 'W' (p*n x rank).
    Score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook decomposition: rank = n * m * p
    rank = n * m * p
    
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1.0
                V[j * p + k, column] = 1.0
                W[k * n + i, column] = 1.0
                column += 1
    
    result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    
    return result
# EVOLVE-BLOCK-END
