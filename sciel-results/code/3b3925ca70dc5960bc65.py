"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct the matrix multiplication tensor decomposition for 3x5 * 5x6.
    We use the standard schoolbook algorithm which provides a valid decomposition
    of rank n*m*p. For these dimensions, this is the baseline and likely optimal.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Generate indices for the tensor T_{i,j,k} = A_{i,j} B_{j,k}
    # We need U, V, W such that T = sum_r U_r \otimes V_r \otimes W_r
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # Create arrays of indices i, j, k
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten to 1D
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Map to linear indices for U, V, W
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i (Trace convention: W[k*n+i, r])
    
    u_idx = i_flat * m + j_flat
    v_idx = j_flat * p + k_flat
    w_idx = k_flat * n + i_flat
    
    # Initialize sparse matrices as dense (since rank is small enough 90)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Fill the matrices
    # Each column r corresponds to a unique (i,j,k) triple
    # U[u_idx[r], r] = 1
    # V[v_idx[r], r] = 1
    # W[w_idx[r], r] = 1
    
    # Use advanced indexing
    U[u_idx, np.arange(rank)] = 1
    V[v_idx, np.arange(rank)] = 1
    W[w_idx, np.arange(rank)] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
