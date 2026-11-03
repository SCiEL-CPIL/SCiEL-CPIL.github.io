"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s margin for safety
    
    n, m, p = DIMENSIONS
    
    # Fallback: schoolbook decomposition
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower-rank decomposition using a heuristic search
    # For 2x3x10, the schoolbook rank is 60.
    # We can try to find a rank < 60 decomposition.
    
    # One approach: try to find a decomposition using a random Gaussian initialization
    # and then solve for the remaining factors.
    
    # However, finding an exact integer decomposition via random search is unlikely to succeed quickly.
    # So we will stick with the schoolbook decomposition for safety.
    
    return schoolbook()
# EVOLVE-BLOCK-END
