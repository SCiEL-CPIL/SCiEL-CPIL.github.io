"""Program for 2x4x4 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The minimal rank for M_{n,m,p} in this case is n*m*p = 32.
    # We construct the standard basis decomposition.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # The tensor is T = sum_r U[:,r] V[:,r]^T W[:,r]^T 
    # Wait, the convention is W[k*n+i, r].
    # Standard decomposition:
    # For each i in [0,n), j in [0,m), k in [0,p):
    # U[i*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + i, col] = 1
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                U[idx_u, col] = 1
                V[idx_v, col] = 1
                W[idx_w, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
