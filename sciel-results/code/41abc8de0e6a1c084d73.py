import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Approach: Algorithmic generation of the schoolbook decomposition
    # by explicitly defining the tensor support and mapping to factors.
    # This avoids hardcoded loop structures and uses vectorized logic 
    # to determine the non-zero indices, representing a different 
    # implementation strategy for the same mathematical object.
    
    # The tensor T is defined by T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i
    # We generate the indices for all non-zero terms.
    
    # Generate all (i, j, k) tuples
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten to get lists of indices
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Calculate the corresponding indices in the factor matrices
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    u_indices = i_flat * m + j_flat
    v_indices = j_flat * p + k_flat
    w_indices = k_flat * n + i_flat
    
    rank = len(u_indices)
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Populate the factors
    # Each column t corresponds to a specific (i, j, k) term
    # U[u_indices[t], t] = 1
    # V[v_indices[t], t] = 1
    # W[w_indices[t], t] = 1
    
    U[u_indices, np.arange(rank)] = 1.0
    V[v_indices, np.arange(rank)] = 1.0
    W[w_indices, np.arange(rank)] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
