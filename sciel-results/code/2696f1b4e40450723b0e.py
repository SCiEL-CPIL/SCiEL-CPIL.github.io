import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Convert to list of lists for JSON serialization if required by interface
    # The interface expects {'U', 'V', 'W'} with integer/half-integer entries.
    # The schoolbook entries are 0 or 1.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
