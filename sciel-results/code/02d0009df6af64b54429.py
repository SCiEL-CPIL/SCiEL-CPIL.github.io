import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication.
    Rank = n * m * p = 90.
    """
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices as lists of lists for compatibility
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                # V index: j * p + k
                # W index: k * n + i (Trace convention: W[k*n+i, r])
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
                if time.time() > deadline:
                    # Fallback: return what we have? No, must be valid.
                    # But 90 iterations is instantaneous.
                    pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
