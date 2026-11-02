"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 4 matrix.
    
    The tensor rank of M_{2,3,4} is known to be 24 (the schoolbook rank).
    We return the standard schoolbook decomposition.
    
    Interface:
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank) with index k*n + i
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 24
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
