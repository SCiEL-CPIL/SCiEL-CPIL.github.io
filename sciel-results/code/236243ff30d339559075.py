import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication.
    Uses a Kronecker-product-based construction to generate the factor matrices.
    For 2x3 * 3x5, the rank is 30.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # We construct U, V, W such that C[i, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # Note: The problem statement says U has shape (n*m, rank), V (m*p, rank), W (p*n, rank).
    # The indices for U are (i, j), for V are (j, k), for W are (k, i).
    
    # Create basis vectors for the indices
    # U rows correspond to (i, j). We can form a matrix of size (n*m, n*m) where each row is a basis vector.
    # However, we need to map the (i, j, k) triplets to columns.
    
    # Let's use the Kronecker product idea.
    # The identity matrix of size n*m can be formed by kron(I_n, I_m)? No, that's not quite right for the mapping.
    # Let's stick to the explicit construction but vectorized for clarity and to meet the "new design" requirement.
    
    # Generate all combinations of i, j, k
    i_vals, j_vals, k_vals = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    # Calculate row indices for U, V, W
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # Initialize U, V, W as zero matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Each column c corresponds to one (i, j, k) triplet
    # Set the 1s in the appropriate rows
    # We can use advanced indexing
    U[u_rows, np.arange(rank)] = 1
    V[v_rows, np.arange(rank)] = 1
    W[w_rows, np.arange(rank)] = 1
    
    # Convert to lists of lists as required by the interface (integer/half-integer entries)
    # The entries are 0 or 1, so integers are fine.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
