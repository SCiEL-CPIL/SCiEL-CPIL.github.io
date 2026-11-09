"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # seconds before the 150s kill
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as baseline
    # Rank = n * m * p = 3 * 3 * 7 = 63
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                # V index: j * p + k
                # W index: k * n + i
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    # Return the schoolbook decomposition
    # This is guaranteed to be valid and has rank 63.
    # Improving this requires specific algebraic decompositions (e.g., Strassen-like)
    # which are not easily found via random search in the time budget.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
