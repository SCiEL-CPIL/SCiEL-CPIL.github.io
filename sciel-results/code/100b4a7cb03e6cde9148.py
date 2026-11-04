"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130  # Leave 20s safety margin before 150s kill
    
    # Known lower bound for 3x3 * 3x8 matrix multiplication
    # The trivial decomposition has rank 3*3*8 = 72
    # We know that for matrix multiplication, rank >= max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But the actual minimum rank for 3x3 * 3x8 is likely higher than 24.
    # Let's try to find a better decomposition than the trivial one.
    
    # The trivial decomposition:
    rank_trivial = n * m * p  # 72
    
    # For now, return the trivial decomposition as a baseline
    # In a more sophisticated approach, we could try to use structured decompositions
    # or known algorithms for matrix multiplication.
    
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
