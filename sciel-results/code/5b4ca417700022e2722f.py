import numpy as np
import time

DIMENSIONS = [3, 4, 7]
START_TIME = time.time()
TIME_LIMIT = 120

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 4 * 7 = 28
    
    # Initialize as NumPy arrays
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            for i in range(n):
                U[i * m + j, c] = 1.0
                W[k * n + i, c] = 1.0
            V[j * p + k, c] = 1.0
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
