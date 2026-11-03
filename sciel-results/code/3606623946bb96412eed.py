"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _construct():
    n, m, p = 4, 5, 5
    # Trivial decomposition with rank n*m*p = 100
    total_rank = n * m * p
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    result = _construct()
    
    # Check if we're within the deadline
    if time.time() > deadline:
        return result
    
    # Try to improve the decomposition if we have time
    # For now, just return the trivial decomposition
    return result

# EVOLVE-BLOCK-END
