import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    N = n * m  # 25
    M = m * p  # 25
    P = p * n  # 25
    
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Fallback to schoolbook
    rank = n * m * p
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
    
    # Placeholder for optimization search
    # If time permits, one could attempt to reduce rank here.
    # Currently, we return the known valid schoolbook solution.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
