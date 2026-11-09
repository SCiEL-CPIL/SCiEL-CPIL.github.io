import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as zero matrices
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Vectorized construction of the schoolbook decomposition
    # Each column c corresponds to a unique (i, j, k) triplet
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    
    # Generate indices for all (i, j, k) combinations
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten the indices to get a list of (i, j, k) for each column
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Calculate the row indices for U, V, W
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # The column index is simply 0, 1, ..., rank-1
    cols = np.arange(rank)
    
    # Set the ones
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    # Convert to lists of lists as required by the interface (or ensure it returns the right type)
    # The prompt says "factor matrices of shapes... with integer/half-integer entries".
    # Returning numpy arrays is usually acceptable if the evaluator handles them, 
    # but the seed returned lists. I will return lists to be safe, or numpy arrays if the interface allows.
    # The seed returned: {'U': U, 'V': V, 'W': W} where U was a list of lists.
    # I will convert to lists to match the seed's type exactly.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
