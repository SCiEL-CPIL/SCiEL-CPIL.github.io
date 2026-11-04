import numpy as np

def construct():
    n, m, p = 2, 3, 9
    # Generate all indices i, j, k
    # i in [0, n), j in [0, m), k in [0, p)
    
    # We need to create U, V, W of shapes (n*m, n*m*p), (m*p, n*m*p), (p*n, n*m*p)
    # Rank R = n*m*p = 54
    
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Iterate through all combinations to populate the identity decomposition
    # This is the "trivial" decomposition where each term picks out one (i,j,k)
    
    # Vectorized approach to generate indices
    # i_idx, j_idx, k_idx
    i_vals, j_vals, k_vals = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    # Calculate linear indices
    u_indices = i_flat * m + j_flat
    v_indices = j_flat * p + k_flat
    w_indices = k_flat * n + i_flat
    
    # Set the 1s
    U[u_indices, np.arange(R)] = 1
    V[v_indices, np.arange(R)] = 1
    W[w_indices, np.arange(R)] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
