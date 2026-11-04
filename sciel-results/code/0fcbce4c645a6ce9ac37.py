"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a tensor decomposition for 4x5x5 matrix multiplication.
    Returns a rank-100 schoolbook decomposition with integer coefficients.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices as lists of lists
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
    
    # Check if we're still within time budget
    if time.time() > deadline:
        # Return what we have (should be complete already)
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
