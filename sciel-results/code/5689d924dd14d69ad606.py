import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.
    Currently returns the schoolbook decomposition of rank n*m*p = 60.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep safety margin
    
    n, m, p = DIMENSIONS
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
                
    # Check time, though this construction is instantaneous
    if time.time() > deadline:
        # This should never happen for schoolbook, but good practice
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
