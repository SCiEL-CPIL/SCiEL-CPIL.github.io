def _get_rank2_221():
    """
    Returns U, V, W for 2x2x1 multiplication using Rank 2 decomposition.
    U shape: (4, 2), V shape: (2, 2), W shape: (2, 2)
    Indices: U[i*2+j], V[j*1+k], W[k*2+i]
    """
    n, m, p = 2, 2, 1
    r = 2
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # c=0
    # u0 = (1, 0) for j=0,1
    # v0 = (1, 0) for j=0,1
    # w0 = (1, 1) for i=0,1
    U[0, 0] = 1 # i=0, j=0
    U[2, 0] = 0 # i=1, j=0
    U[1, 0] = 0 # i=0, j=1
    U[3, 0] = 0 # i=1, j=1
    
    V[0, 0] = 1 # j=0, k=0
    V[1, 0] = 0 # j=1, k=0
    
    W[0, 0] = 1 # k=0, i=0
    W[1, 0] = 1 # k=0, i=1
    
    # c=1
    # u1 = (0, 1) for j=0,1
    # v1 = (0, 1) for j=0,1
    # w1 = (1, 1) for i=0,1
    U[0, 1] = 0 # i=0, j=0
    U[2, 1] = 0 # i=1, j=0
    U[1, 1] = 1 # i=0, j=1
    U[3, 1] = 1 # i=1, j=1
    
    V[0, 1] = 0 # j=0, k=0
    V[1, 1] = 1 # j=1, k=0
    
    W[0, 1] = 1 # k=0, i=0
    W[1, 1] = 1 # k=0, i=1
    
    return U, V, W
