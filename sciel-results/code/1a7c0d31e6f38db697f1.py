"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for matrix multiplication
    with structured block-diagonal partitioning.
    
    For multiplying a (n x m) matrix by a (m x p) matrix, we use the standard
    trace convention where W has index [k*n + i, r] for the output.
    
    The decomposition uses the schoolbook structure where each term corresponds
    to a unique (i, j, k) triple with non-zero entries only in the corresponding
    sub-blocks of U, V, and W.
    
    Returns:
        dict with keys 'U', 'V', 'W' where:
        - U has shape (n*m, rank)
        - V has shape (m*p, rank)  
        - W has shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the structured decomposition
    # Each column r corresponds to a unique (i, j, k) triple
    # U[i*m + j, r] = 1, V[j*p + k, r] = 1, W[k*n + i, r] = 1
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
