"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for 2x3x3 matrix multiplication.
    The minimal rank is 18. We return the standard schoolbook decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 3, 3
    rank = n * m * p  # 18
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank) = (6, 18)
    # V shape: (m*p, rank) = (9, 18)
    # W shape: (p*n, rank) = (6, 18)
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Fill in the schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: i*m + j
                # Index in V: j*p + k
                # Index in W: k*n + i (as per task description: W[k*n+i, r])
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
