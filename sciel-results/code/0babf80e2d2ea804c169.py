import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # The decomposition is the sum of 3 outer product tensors.
    # For each column j of A (and row j of B), we have a 2x3 outer product.
    # This requires 6 multiplications.
    # Total 3 * 6 = 18.
    
    idx = 0
    for j in range(m):
        for i in range(n):
            for k in range(p):
                # A index: i * m + j
                u_idx = i * m + j
                # B index: j * p + k
                v_idx = j * p + k
                # C index: k * n + i (based on W[k*n+i, r])
                w_idx = k * n + i
                
                U[u_idx, idx] = 1
                V[v_idx, idx] = 1
                W[w_idx, idx] = 1
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
