import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Iterate over the rank components
    for c in range(rank):
        # Determine which (i, j, k) this component corresponds to
        # We group by k, then by (i, j)
        # There are m*n terms for each k
        k = c // (n * m)
        idx = c % (n * m)
        
        # Map idx to i, j
        # idx = i * m + j
        i = idx // m
        j = idx % m
        
        # Set the entries
        u_row = i * m + j
        v_row = j * p + k
        w_row = k * n + i
        
        U[u_row, c] = 1
        V[v_row, c] = 1
        W[w_row, c] = 1
        
    return {"U": U, "V": V, "W": W}
