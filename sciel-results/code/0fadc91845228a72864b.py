import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # The decomposition is a direct sum of 3 outer-product tensors M_{n,1,p}
    # For each j in 0..m-1, we have a block of rank n*p
    # The indices are:
    # U: (i*m + j, col)
    # V: (j*p + k, col)
    # W: (k*n + i, col)
    
    col = 0
    for j in range(m):
        for i in range(n):
            for k in range(p):
                # Map to flattened indices
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
                
    # Convert to lists as required by interface (or keep as numpy if allowed? 
    # "integer/half-integer entries" usually implies lists or arrays are fine, 
    # but the seed returns lists of lists. Let's return lists to be safe.)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
