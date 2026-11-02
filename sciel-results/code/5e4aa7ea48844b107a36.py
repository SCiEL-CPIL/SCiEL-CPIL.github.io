"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    start_time = time.time()
    deadline = start_time + 140  # 140s to leave margin for 150s limit
    
    # Schoolbook decomposition: rank = n*m*p = 105
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
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to find a lower-rank construction using structured decomposition
    # For 3x5x7, we can potentially use a block-circulant approach
    
    # Check if we have time to try a better construction
    if time.time() < deadline - 30:
        # Try a simple improvement: use the fact that for each row i,
        # the 5x7 multiplication can be done with fewer than 35 terms
        # using a structured decomposition
        
        # For now, return the schoolbook as it's guaranteed correct
        # A more sophisticated search would go here
        
        pass
    
    return best_result
# EVOLVE-BLOCK-END
