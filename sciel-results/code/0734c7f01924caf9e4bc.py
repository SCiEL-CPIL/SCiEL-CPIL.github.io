import numpy as np
import time

def construct():
    n, m, p = 3, 5, 5
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin within the 120s limit
    
    # Try to construct a lower-rank decomposition using Strassen-like techniques
    # For now, use schoolbook as a safe fallback
    R = n * m * p  # 75
    
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
