import numpy as np

def construct():
    n, m, p = 4, 5, 6
    # The naive rank is n * m * p
    rank = n * m * p
    
    # Initialize factor matrices
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Fill the matrices based on the tensor decomposition definition:
    # T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    # For the naive decomposition, each term (i,j,k) is its own rank-1 component.
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U component: row i*m+j
                U[i * m + j, idx] = 1.0
                # V component: row j*p+k
                V[j * p + k, idx] = 1.0
                # W component: row k*n+i
                W[k * n + i, idx] = 1.0
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
