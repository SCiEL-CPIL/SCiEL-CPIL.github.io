def simple_decomp_4x4x3():
    n, m, p_global = 4, 4, 7
    p_local = 3
    k_offset = 4
    rank = p_local
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p_global, rank))
    W = np.zeros((p_global * n, rank))
    
    for k_local in range(p_local):
        r = k_local
        k_global = k_local + k_offset
        U[:, r] = 1
        for j in range(m):
            V[j * p_global + k_global, r] = 1
        for i in range(n):
            W[k_global * n + i, r] = 1
            
    return U, V, W
