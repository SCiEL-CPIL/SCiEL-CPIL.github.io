import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create index arrays
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten to 1D
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Calculate row indices for U, V, W
    # U: (n*m, rank), row index i*m + j
    u_rows = i_flat * m + j_flat
    # V: (m*p, rank), row index j*p + k
    v_rows = j_flat * p + k_flat
    # W: (p*n, rank), row index k*n + i
    w_rows = k_flat * n + i_flat
    
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Fill with 1s at the calculated positions
    # Each column c corresponds to the c-th element in the flattened arrays
    U[u_rows, np.arange(rank)] = 1
    V[v_rows, np.arange(rank)] = 1
    W[w_rows, np.arange(rank)] = 1
    
    # Convert to lists as required by the interface?
    # The interface says: factor matrices of shapes ... with integer/half-integer entries.
    # The current program returns lists of lists.
    # I should return lists of lists to be safe, or check if numpy arrays are accepted.
    # "Keep the function names and signatures ... EXACTLY as in the current program".
    # Current program returns {'U': U, 'V': V, 'W': W} where U,V,W are lists of lists.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
