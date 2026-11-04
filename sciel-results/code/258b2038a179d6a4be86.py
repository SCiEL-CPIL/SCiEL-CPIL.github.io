"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Use a block decomposition: split the 5x5 into 2x2 + 2x2 + 1x2 + 2x1 + 1x1 blocks
    # Actually, let's just use the trivial decomposition for now but with a deadline guard.
    # The trivial rank is n*m*p = 2*5*5 = 50.
    
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # For now, return the trivial decomposition.
    # Future improvements: try Strassen on 2x2x2 blocks, or other low-rank decompositions.
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
