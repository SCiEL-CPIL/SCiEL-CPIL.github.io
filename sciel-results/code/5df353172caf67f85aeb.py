"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or at least valid) exact tensor decomposition
    for multiplying a 3x4 matrix by a 4x7 matrix.
    
    Uses the schoolbook decomposition of rank n*m*p.
    This is guaranteed to be valid and uses integer entries (0 and 1).
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(m):
            for k in range(p):
                # Assign rank-1 component
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
            # Check deadline periodically
            if column % 100 == 0 and time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # The loop is small (84 iterations), so this will never trigger early.
    # We return the valid schoolbook decomposition.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
