"""Block decomposition for 3x3 by 3x4 matrix multiplication."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    Urows, Vrows, Wrows = n * m, m * p, p * n  # 9, 12, 12
    
    # Schoolbook: rank = n*m*p = 36
    rank = n * m * p
    U = np.zeros((Urows, rank), dtype=np.int64)
    V = np.zeros((Vrows, rank), dtype=np.int64)
    W = np.zeros((Wrows, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
