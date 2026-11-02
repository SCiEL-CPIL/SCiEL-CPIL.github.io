"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # The standard schoolbook decomposition has rank R = n * m * p.
    # For n=2, m=3, p=9, R = 54.
    # Lower bound is max(n*m, m*p, p*n) = max(6, 27, 18) = 27.
    # No simple rank-27 decomposition is known or easily constructible in this context.
    # We implement the guaranteed valid rank-54 decomposition.
    
    rank = n * m * p
    
    # Initialize factor matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    # Iterate over all indices i, j, k to create rank-1 terms
    # Term c corresponds to indices (i, j, k)
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    # Verify time
    if time.time() > deadline:
        # This shouldn't happen for static construction, but just in case
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
