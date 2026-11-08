"""Rewritten matrix multiplication decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook rank is n*m*p
    rank = n * m * p
    
    # Initialize U, V, W
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank) -- Note: Trace convention W[k*n+i, r]
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j, idx] = 1
                # V index: j*p + k
                V[j * p + k, idx] = 1
                # W index: k*n + i
                W[k * n + i, idx] = 1
                idx += 1
                
    # Convert to lists of lists for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
