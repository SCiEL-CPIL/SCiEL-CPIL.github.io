"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Build U, V, W as dense lists of lists for compatibility
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Use a small randomized perturbation to ensure integer/half-integer uniqueness 
    # is not strictly required for correctness, but we stick to integers for the schoolbook base.
    # The task asks for rank-minimal. The schoolbook is rank 90.
    # Can we do better?
    # For 3x5x6, the border rank is at least 90? No, it's usually close to nmp.
    # Let's just output the schoolbook decomposition as it is guaranteed to be exact and valid.
    # Optimizing rank requires finding a low-rank tensor decomposition which is hard.
    # We will return the schoolbook decomposition.
    
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
