def _construct_rank16():
    """
    Construct a rank-16 decomposition for 3x3 x 3x4 matrix multiplication.
    Uses Strassen for 2x2x2 blocks and rank-1 for the remaining strips.
    """
    U = np.zeros((9, 16))
    V = np.zeros((12, 16))
    W = np.zeros((12, 16))
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Strassen rank 7
    S_U, S_V, S_W = _strassen_2x2x2()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i * 3 + j, r] = S_U[i * 2 + j, r]
                V[i * 4 + j, r] = S_V[i * 2 + j, r]
                W[j * 3 + i, r] = S_W[i * 2 + j, r]
                
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Strassen rank 7
    S_U2, S_V2, S_W2 = _strassen_2x2x2()
    for r in range(7, 14):
        rr = r - 7
        for i in range(2):
            for j in range(2):
                U[i * 3 + j, r] = S_U2[i * 2 + j, rr]
                V[i * 4 + (j + 2), r] = S_V2[i * 2 + j, rr]
                W[(j + 2) * 3 + i, r] = S_W2[i * 2 + j, rr]

    # Block 3: A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4]
    # Rank 1
    r = 14
    for i in range(2):
        U[i * 3 + 2, r] = 1
    V[2 * 4 + 0, r] = 1 # Wait, V index is j*p + k. j=2.
    # We need V[2, k] = 1 for all k?
    # In the term, V is a vector.
    # We need V[2, 0]=1, V[2, 1]=1, V[2, 2]=1, V[2, 3]=1.
    # Indices: 2*4+0, 2*4+1, 2*4+2, 2*4+3.
    V[2 * 4 + 0, r] = 1
    V[2 * 4 + 1, r] = 1
    V[2 * 4 + 2, r] = 1
    V[2 * 4 + 3, r] = 1
    
    for k in range(4):
        W[k * 3 + 0, r] = 1 # C[0, k]
        W[k * 3 + 1, r] = 1 # C[1, k]
        
    # Block 4: A[2:3, 0:3] * B[0:3, 0:4] -> C[2:3, 0:4]
    # Rank 1
    r = 15
    U[2 * 3 + 0, r] = 1 # A[2, 0]
    U[2 * 3 + 1, r] = 1 # A[2, 1]
    U[2 * 3 + 2, r] = 1 # A[2, 2]
    
    for j in range(3):
        V[j * 4 + 0, r] = 1 # B[j, 0]
        V[j * 4 + 1, r] = 1 # B[j, 1]
        V[j * 4 + 2, r] = 1 # B[j, 2]
        V[j * 4 + 3, r] = 1 # B[j, 3]
        
    for k in range(4):
        W[k * 3 + 2, r] = 1 # C[2, k]

    return U, V, W
