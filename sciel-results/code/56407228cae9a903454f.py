import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create indices for the trivial decomposition
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    # We generate the rank-1 terms
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each term is a rank-1 tensor
                # U has 1 at i*m+j
                # V has 1 at j*p+k
                # W has 1 at k*n+i
                terms.append((i * m + j, j * p + k, k * n + i))
                
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for r, (u_idx, v_idx, w_idx) in enumerate(terms):
        U[u_idx, r] = 1
        V[v_idx, r] = 1
        W[w_idx, r] = 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
