"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 90
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
    
    # Try to find a lower-rank decomposition using Strassen-like techniques
    # For 3x5x6, we can try to exploit block structure
    
    # First, let's try a simple approach: use the fact that we can sometimes
    # reduce rank by finding linear dependencies
    
    # For now, return the schoolbook decomposition as a safe baseline
    # A more sophisticated search would be added here if time permits
    
    if time.time() > deadline - 10.0:
        return {"U": U, "V": V, "W": W}
    
    # Try Strassen-inspired approach for 3x5x6
    # We can think of this as multiplying blocks
    # Let's try to find a rank-84 or lower decomposition
    
    # For a 3x5x6 matrix multiplication, we can potentially use
    # a combination of smaller Strassen decompositions
    
    # Simple improvement: check if we can merge any terms
    # (This is a placeholder for a more sophisticated search)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
