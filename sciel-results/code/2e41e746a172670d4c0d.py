import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    
    rank = n * m * p
    
    # Generate indices for the 54 terms
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten indices
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Fill U: row i*m + j, col t
    # t is the index in the flattened arrays (0 to rank-1)
    u_rows = i_flat * m + j_flat
    V_rows = j_flat * p + k_flat
    W_rows = k_flat * n + i_flat
    
    cols = np.arange(rank)
    
    U[u_rows, cols] = 1
    V[V_rows, cols] = 1
    W[W_rows, cols] = 1
    
    # Convert to lists if required by interface? 
    # "factor matrices of shapes ... with integer/half-integer entries"
    # The current program returns lists of lists. 
    # I should probably return lists of lists to be safe, or numpy arrays if accepted.
    # The prompt says "Keep the function names and signatures ... exactly as in the current program".
    # Current program returns {"U": U, ...} where U is a list of lists.
    # I will convert back to lists.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
