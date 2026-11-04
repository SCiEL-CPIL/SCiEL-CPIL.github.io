"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the tensor decomposition of 4x5 by 5x6 matrix multiplication.
    Uses the schoolbook algorithm which has rank n*m*p = 120.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # Note: The problem states U shape (n*m, rank), V (m*p, rank), W (p*n, rank)
    # And W[k*n+i, r] is the output factor.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
                # Time guard
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # If we broke out early, the decomposition is invalid.
    # However, schoolbook is fast, so this should not happen.
    # We return the full valid decomposition.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
