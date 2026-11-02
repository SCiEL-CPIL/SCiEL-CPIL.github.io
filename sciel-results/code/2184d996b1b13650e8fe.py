def construct():
    """Return a valid tensor decomposition for 2x2 * 2x13 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Rank 13 decomposition
    # T[i,j,k] = 1 for all i,j,k
    # U[i*m+j, r] = 1
    # V[j*p+k, r] = 1 if k==r else 0
    # W[k*n+i, r] = 1 if k==r else 0
    rank = p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(p):
        # U is all ones
        U[:, r] = 1.0
        
        # V has 1s at indices j*p + r for j in 0..m-1
        for j in range(m):
            V[j * p + r, r] = 1.0
            
        # W has 1s at indices r*n + i for i in 0..n-1
        for i in range(n):
            W[r * n + i, r] = 1.0
            
    if time.time() < deadline and _check_decomposition(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback: pure schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb))
    V_fb = np.zeros((m * p, rank_fb))
    W_fb = np.zeros((p * n, rank_fb))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m + j, c] = 1
                V_fb[j*p + k, c] = 1
                W_fb[k*n + i, c] = 1
                c += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
