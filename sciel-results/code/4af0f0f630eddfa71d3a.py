DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = 3, 5, 7
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2, 1]
    
    n_starts = [0]
    for s in n_splits[:-1]:
        n_starts.append(n_starts[-1] + s)
        
    m_starts = [0]
    for s in m_splits[:-1]:
        m_starts.append(m_starts[-1] + s)
        
    p_starts = [0]
    for s in p_splits[:-1]:
        p_starts.append(p_starts[-1] + s)
        
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_terms, v_terms, w_terms):
        u_col = [0.0] * (n * m)
        v_col = [0.0] * (m * p)
        w_col = [0.0] * (p * n)
        
        for idx, coeff in u_terms:
            u_col[idx] = coeff
        for idx, coeff in v_terms:
            v_col[idx] = coeff
        for idx, coeff in w_terms:
            w_col[idx] = coeff
            
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    for ni, ns in enumerate(n_splits):
        for mi, ms in enumerate(m_splits):
            for pi, ps in enumerate(p_splits):
                i0 = n_starts[ni]
                j0 = m_starts[mi]
                k0 = p_starts[pi]
                
                if ns == 2 and ms == 2 and ps == 2:
                    # Strassen
                    add_term([(i0*5 + j0, 1), ((i0+1)*5 + (j0+1), 1)], [(j0*7 + k0, 1), ((j0+1)*7 + (k0+1), 1)], [(k0*3 + i0, 1), ((k0+1)*3 + (i0+1), 1)])
                    add_term([((i0+1)*5 + j0, 1), ((i0+1)*5 + (j0+1), 1)], [(j0*7 + k0, 1)], [(k0*3 + (i0+1), 1)])
                    add_term([(i0*5 + j0, 1)], [((j0+1)*7 + k0, 1), ((j0+1)*7 + (k0+1), -1)], [((k0+1)*3 + i0, 1)])
                    add_term([((i0+1)*5 + (j0+1), 1)], [(j0*7 + (k0+1), 1), ((j0+1)*7 + (k0+1), 1)], [(k0*3 + (i0+1), 1), ((k0+1)*3 + (i0+1), 1)])
                    add_term([(i0*5 + j0, 1), (i0*5 + (j0+1), 1)], [((j0+1)*7 + (k0+1), 1)], [((k0+1)*3 + i0, -1), ((k0+1)*3 + (i0+1), 1)])
                    add_term([((i0+1)*5 + j0, 1), (i0*5 + j0, -1)], [(j0*7 + k0, 1), ((j0+1)*7 + k0, 1)], [((k0+1)*3 + (i0+1), 1)])
                    add_term([(i0*5 + (j0+1), 1), ((i0+1)*5 + (j0+1), -1)], [((j0+1)*7 + (k0+1), 1), (j0*7 + (k0+1), 1)], [(k0*3 + i0, 1), ((k0+1)*3 + (i0+1), -1)])
                else:
                    for i in range(ns):
                        for j in range(ms):
                            for k in range(ps):
                                gi = i0 + i
                                gj = j0 + j
                                gk = k0 + k
                                add_term([(gi*5 + gj, 1)], [(gj*7 + gk, 1)], [(gk*3 + gi, 1)])
                                
    rank = len(U_cols)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
