import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    # n=3, m=3, p=6
    
    # Try to use rank-23 Bini decomposition for each 3x3x3 block
    # Total rank would be 46
    
    # Bini et al. rank-23 decomposition for 3x3x3
    # Each term: (u, v, w) where u, v, w are in R^3
    # T[i,j,k] = sum_r u[r][i] * v[r][j] * w[r][k]
    
    # Let me try to use a known formulation
    # The Bini decomposition can be verified computationally
    
    # For now, let's try the schoolbook as fallback
    # and attempt to construct rank-46 if possible
    
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
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
