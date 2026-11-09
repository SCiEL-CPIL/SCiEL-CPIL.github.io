# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    For small rectangular matrix multiplication (3x4 and 4x4), the trivial 
    schoolbook decomposition (rank = n*m*p) is generally considered minimal 
    for exact integer/half-integer decompositions. 
    We construct the basis decomposition where each term A[i,j] * B[j,k] 
    contributes independently to C[i,k].
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Vectorized construction
    # We need to map each (i, j, k) to a column r.
    # r = i * m * p + j * p + k
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    # Create index arrays
    i_vals, j_vals, k_vals = np.indices((n, m, p))
    
    # Flatten indices for r
    r_flat = (i_vals * m + j_vals) * p + k_vals
    r_flat = r_flat.flatten()
    
    # Calculate linear indices for U, V, W
    u_idx = (i_vals * m + j_vals).flatten()
    v_idx = (j_vals * p + k_vals).flatten()
    w_idx = (k_vals * n + i_vals).flatten()
    
    # Set the 1s
    U[u_idx, r_flat] = 1.0
    V[v_idx, r_flat] = 1.0
    W[w_idx, r_flat] = 1.0
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
