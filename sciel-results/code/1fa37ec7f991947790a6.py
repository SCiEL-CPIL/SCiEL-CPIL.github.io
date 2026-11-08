import numpy as np
import time
import sys

DIMENSIONS = [2, 5, 6]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    # Contract: U (n*m, rank), V (m*p, rank), W (p*n, rank)
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
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s safety margin
    
    # Return schoolbook solution
    # This is rank n*m*p = 60
    sol = _schoolbook()
    
    # Optional: Verify
    # n, m, p = DIMENSIONS
    # if sol['U'].shape != (n*m, n*m*p):
    #     print("Shape mismatch U", file=sys.stderr)
    # if sol['V'].shape != (m*p, n*m*p):
    #     print("Shape mismatch V", file=sys.stderr)
    # if sol['W'].shape != (p*n, n*m*p):
    #     print("Shape mismatch W", file=sys.stderr)
        
    return sol
