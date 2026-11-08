import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition using Strassen's algorithm
    # For 4x4x4, use block Strassen
    # For 4x5x5, extend with additional terms
    
    # Let's try a simple approach: use the schoolbook but see if we can reduce
    # For now, return schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
