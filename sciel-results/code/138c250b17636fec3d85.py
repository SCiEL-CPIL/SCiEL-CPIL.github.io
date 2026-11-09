import time
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE = 150.0

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 3 by 4 matrix by a 4 by 5 matrix.
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # The schoolbook rank is n*m*p = 60.
    # For 3x4x5, the minimal rank is believed to be 60.
    # We construct the schoolbook decomposition.
    rank = n * m * p
    
    # Initialize matrices as NumPy arrays for efficient construction
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Fill the matrices
    # Index mapping:
    # U[i*m + j, r] = 1
    # V[j*p + k, r] = 1
    # W[k*n + i, r] = 1
    # where r = i*m*p + j*p + k
    
    # We can vectorize this using indices
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten indices for U, V, W rows
    u_rows = (i_idx * m + j_idx).flatten()
    v_rows = (j_idx * p + k_idx).flatten()
    w_rows = (k_idx * n + i_idx).flatten()
    
    # Column index is just 0..rank-1
    cols = np.arange(rank)
    
    # Set the 1s
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    # Convert to lists of lists as required by the interface
    # The interface expects lists, but NumPy arrays are often accepted if convertible.
    # The prompt says "list of lists".
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
