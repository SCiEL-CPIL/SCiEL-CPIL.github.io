# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for 2x2 by 2x11 matrix multiplication.
    The minimum rank is 4.
    """
    n, m, p = DIMENSIONS
    rank = 4
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Term 1: A[0,0] * B[0, k] -> C[0, k]
    U[0, 0] = 1
    for k in range(p):
        V[0 * p + k, 0] = 1  # B[0, k]
        W[k * n + 0, 0] = 1  # C[0, k]
        
    # Term 2: A[0,1] * B[1, k] -> C[0, k]
    U[1, 1] = 1
    for k in range(p):
        V[1 * p + k, 1] = 1  # B[1, k]
        W[k * n + 0, 1] = 1  # C[0, k]
        
    # Term 3: A[1,0] * B[0, k] -> C[1, k]
    U[2, 2] = 1
    for k in range(p):
        V[0 * p + k, 2] = 1  # B[0, k]
        W[k * n + 1, 2] = 1  # C[1, k]
        
    # Term 4: A[1,1] * B[1, k] -> C[1, k]
    U[3, 3] = 1
    for k in range(p):
        V[1 * p + k, 3] = 1  # B[1, k]
        W[k * n + 1, 3] = 1  # C[1, k]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
