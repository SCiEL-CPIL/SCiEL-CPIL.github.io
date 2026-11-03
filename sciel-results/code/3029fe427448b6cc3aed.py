def construct_factors(m, n, p):
    if m == 1 and n == 1 and p == 1:
        return np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1))
    if m == 2 and n == 2 and p == 2:
        return get_strassen_factors()
    
    # Find optimal split
    best_rank = m * n * p
    best_split = None
    for m1 in range(1, m):
        m2 = m - m1
        for n1 in range(1, n):
            n2 = n - n1
            for p1 in range(1, p):
                p2 = p - p1
                r = (min_rank_hybrid(m1, n1, p1) + min_rank_hybrid(m1, n2, p1) + 
                     min_rank_hybrid(m2, n1, p1) + min_rank_hybrid(m2, n2, p1) +
                     min_rank_hybrid(m1, n1, p2) + min_rank_hybrid(m1, n2, p2) +
                     min_rank_hybrid(m2, n1, p2) + min_rank_hybrid(m2, n2, p2))
                if r < best_rank:
                    best_rank = r
                    best_split = (m1, n1, p1)
    
    if best_split is None:
        return get_schoolbook_factors(m, n, p)
    
    m1, n1, p1 = best_split
    m2 = m - m1
    n2 = n - n1
    p2 = p - p1
    
    U11, V11, W11 = construct_factors(m1, n1, p1)
    U12, V12, W12 = construct_factors(m1, n2, p1)
    U21, V21, W21 = construct_factors(m2, n1, p1)
    U22, V22, W22 = construct_factors(m2, n2, p1)
    U13, V13, W13 = construct_factors(m1, n1, p2)
    U14, V14, W14 = construct_factors(m1, n2, p2)
    U23, V23, W23 = construct_factors(m2, n1, p2)
    U24, V24, W24 = construct_factors(m2, n2, p2)
    
    # Concatenate U:
    # U shape (m*n, rank)
    # Rows of U correspond to (i,j) in A.
    # Block 11: i in 0..m1-1, j in 0..n1-1. U11 is (m1*n1, r11).
    # We need to map these to the global U indices.
    # Global U index for (i,j) is i*n + j.
    # This requires careful indexing.
    
    # It is easier to build the full U, V, W by placing the sub-factors into the correct positions.
    # However, the sub-factors U11 etc are local to their sub-blocks.
    # We need to expand them to the global size.
    
    def expand_U(sub_U, m_sub, n_sub, i_off, j_off, m_tot, n_tot):
        # sub_U is (m_sub*n_sub, rank)
        # We create a (m_tot*n_tot, rank) matrix
        out = np.zeros((m_tot * n_tot, sub_U.shape[1]))
        for i in range(m_sub):
            for j in range(n_sub):
                local_idx = i * n_sub + j
                global_idx = (i + i_off) * n_tot + (j + j_off)
                out[global_idx, :] = sub_U[local_idx, :]
        return out

    def expand_V(sub_V, n_sub, p_sub, j_off, k_off, n_tot, p_tot):
        # sub_V is (n_sub*p_sub, rank)
        out = np.zeros((n_tot * p_tot, sub_V.shape[1]))
        for j in range(n_sub):
            for k in range(p_sub):
                local_idx = j * p_sub + k
                global_idx = (j + j_off) * p_tot + (k + k_off)
                out[global_idx, :] = sub_V[local_idx, :]
        return out

    def expand_W(sub_W, p_sub, m_sub, k_off, i_off, p_tot, m_tot):
        # sub_W is (p_sub*m_sub, rank)
        out = np.zeros((p_tot * m_tot, sub_W.shape[1]))
        for k in range(p_sub):
            for i in range(m_sub):
                local_idx = k * m_sub + i
                global_idx = (k + k_off) * m_tot + (i + i_off)
                out[global_idx, :] = sub_W[local_idx, :]
        return out

    U = np.hstack([
        expand_U(U11, m1, n1, 0, 0, m, n),
        expand_U(U12, m1, n2, 0, n1, m, n),
        expand_U(U21, m2, n1, m1, 0, m, n),
        expand_U(U22, m2, n2, m1, n1, m, n),
        expand_U(U13, m1, n1, 0, 0, m, n),
        expand_U(U14, m1, n2, 0, n1, m, n),
        expand_U(U23, m2, n1, m1, 0, m, n),
        expand_U(U24, m2, n2, m1, n1, m, n)
    ])
    
    V = np.hstack([
        expand_V(V11, n1, p1, 0, 0, n, p),
        expand_V(V12, n2, p1, n1, 0, n, p),
        expand_V(V21, n1, p1, 0, 0, n, p),
        expand_V(V22, n2, p1, n1, 0, n, p),
        expand_V(V13, n1, p2, 0, p1, n, p),
        expand_V(V14, n2, p2, n1, p1, n, p),
        expand_V(V23, n1, p2, 0, p1, n, p),
        expand_V(V24, n2, p2, n1, p1, n, p)
    ])
    
    W = np.hstack([
        expand_W(W11, p1, m1, 0, 0, p, m),
        expand_W(W12, p1, m1, 0, 0, p, m),
        expand_W(W21, p1, m1, 0, 0, p, m),
        expand_W(W22, p1, m1, 0, 0, p, m),
        expand_W(W13, p2, m1, p1, 0, p, m),
        expand_W(W14, p2, m1, p1, 0, p, m),
        expand_W(W23, p2, m1, p1, 0, p, m),
        expand_W(W24, p2, m1, p1, 0, p, m)
    ])
    
    return U, V, W

def construct():
    n, m, p = 3, 5, 7
    # Note: The task says A is 3x5, B is 5x7.
    # My functions use m, n, p as (rows A, cols A, cols B).
    # So m=3, n=5, p=7.
    U, V, W = construct_factors(3, 5, 7)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
