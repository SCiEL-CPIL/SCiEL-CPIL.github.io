import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s margin

    # Schoolbook rank
    rank = n * m * p
    
    # Construct U, V, W as lists of lists to avoid numpy type issues in evaluator
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
        
    # Safety check
    if time.time() > deadline:
        # This should never happen for schoolbook, but just in case
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
