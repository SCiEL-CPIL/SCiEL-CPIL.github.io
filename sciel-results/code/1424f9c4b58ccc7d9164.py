import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a valid exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Uses the canonical schoolbook (trivial) decomposition of rank n*m*p = 64.
    """
    n, m, p = DIMENSIONS
    total_rank = n * m * p
    
    # Initialize factor matrices with zeros
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Convert to lists if the evaluator expects lists, though usually numpy arrays are fine
    # The seed used lists. The interface says "factor matrices".
    # To be safe and match the seed's type, convert to lists.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
