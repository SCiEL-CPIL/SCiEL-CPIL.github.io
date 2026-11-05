"""Matrix multiplication 3x5x5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-75 decomposition for 3x5x5 matrix multiplication.
    
    The matrix multiplication tensor T is defined by T[i, j, k] = 1 if the
    multiplication involves A[i,j] * B[j,k] contributing to C[i,k].
    
    The standard schoolbook decomposition uses rank R = n*m*p.
    Each term in the sum corresponds to a specific (i, j, k) triple.
    
    U has shape (n*m, R). U[idx(i,j), r] is non-zero only when r corresponds to (i,j,k).
    V has shape (m*p, R). V[idx(j,k), r] is non-zero only when r corresponds to (i,j,k).
    W has shape (p*n, R). W[idx(k,i), r] is non-zero only when r corresponds to (i,j,k).
    
    Note: The W matrix index is defined as W[k*n + i, r] per the task contract.
    """
    n, m, p = DIMENSIONS
    R = n * m * p
    
    # Initialize U, V, W as lists of lists of zeros
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Populate the decomposition
    # We iterate through all possible (i, j, k) combinations
    # and assign them to a unique column r in the factor matrices.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The column index r is a unique identifier for the (i,j,k) term
                r = (i * m + j) * p + k
                
                # U index: row is i*m + j
                u_row = i * m + j
                U[u_row][r] = 1
                
                # V index: row is j*p + k
                v_row = j * p + k
                V[v_row][r] = 1
                
                # W index: row is k*n + i (per contract W[k*n+i, r])
                w_row = k * n + i
                W[w_row][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
