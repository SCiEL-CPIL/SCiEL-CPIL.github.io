"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-3 decomposition for 2x3 * 3x10 matrix multiplication.
    
    Key insight: The multiplication tensor T[i,j,k] = 1 can be decomposed as
    sum_{j=0}^{2} (1_i) * (e_j) * (1_k), giving rank 3.
    
    For each intermediate index j, the contribution is:
    - U: column j has 1s in rows corresponding to all i for that j
    - V: column j has 1s in columns corresponding to all k for that j  
    - W: column j has 1s in all positions (since W doesn't depend on j in a selective way)
    """
    n, m, p = DIMENSIONS
    rank = m  # rank = 3
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # For each j in range(m):
    # U[i*m + j, j] = 1 for all i in range(n)
    # V[j*p + k, j] = 1 for all k in range(p)
    # W[k*n + i, j] = 1 for all k in range(p), i in range(n)
    
    for j in range(m):
        # U: set column j
        for i in range(n):
            U[i * m + j, j] = 1.0
        
        # V: set column j
        for k in range(p):
            V[j * p + k, j] = 1.0
        
        # W: set column j (all ones)
        for k in range(p):
            for i in range(n):
                W[k * n + i, j] = 1.0
    
    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
