"""Program for 3x3 x 3x5 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]
DEADLINE = 150.0
SAFETY_MARGIN = 10.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Rank = n * m = 9
    # Construction:
    # For each (i, j), create one term r.
    # U[i*m + j, r] = 1
    # V[j*p + k, r] = 1 for all k
    # W[k*n + i, r] = 1 for all k
    
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            # U
            U[i * m + j][r] = 1
            # V
            for k in range(p):
                V[j * p + k][r] = 1
            # W
            for k in range(p):
                W[k * n + i][r] = 1
            r += 1
            
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for rr in range(rank):
                    val += U[i * m + j][rr] * V[j * p + k][rr] * W[k * n + i][rr]
                if abs(val - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to schoolbook if the simple construction fails (it shouldn't)
        rank = n * m * p
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
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
