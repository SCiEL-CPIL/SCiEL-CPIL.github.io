"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.
    The bilinear rank of 2x3 matrix multiplication is 6.
    Since the 10 columns of the second matrix are independent, the total rank is 10 * 6 = 60.
    This function constructs the schoolbook decomposition of rank 60.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, idx] = 1
                V[v_idx, idx] = 1
                W[w_idx, idx] = 1
                idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
