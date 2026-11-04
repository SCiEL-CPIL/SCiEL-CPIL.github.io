import numpy as np

def construct():
    n, m, p = 3, 4, 8
    
    # We will build U, V, W as numpy arrays
    # We need to know the rank beforehand or build dynamically
    # Since we use recursion, we can count the terms first or build lists.
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries, i_off, j_off, k_off):
        terms.append((u_entries, v_entries, w_entries, i_off, j_off, k_off))
        
    def decompose(n_sub, m_sub, p_sub, i_off, j_off, k_off):
        if n_sub == 2 and m_sub == 2 and p_sub == 2:
            # Strassen 7 terms
            add_term([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], i_off, j_off, k_off)
            add_term([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,-1)], i_off, j_off, k_off)
            add_term([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], i_off, j_off, k_off)
            add_term([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], i_off, j_off, k_off)
            add_term([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (1,0,1)], i_off, j_off, k_off)
            add_term([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)], i_off, j_off, k_off)
            add_term([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)], i_off, j_off, k_off)
        elif m_sub > 2 and m_sub % 2 == 0:
            m_half = m_sub // 2
            decompose(n_sub, m_half, p_sub, i_off, j_off, k_off)
            decompose(n_sub, m_half, p_sub, i_off, j_off + m_half, k_off)
        elif p_sub > 2 and p_sub % 2 == 0:
            p_half = p_sub // 2
            decompose(n_sub, m_sub, p_half, i_off, j_off, k_off)
            decompose(n_sub, m_sub, p_half, i_off, j_off, k_off + p_half)
        elif n_sub > 2:
            n1 = 2
            n2 = n_sub - 2
            decompose(n1, m_sub, p_sub, i_off, j_off, k_off)
            decompose(n2, m_sub, p_sub, i_off + n1, j_off, k_off)
        else:
            # Schoolbook
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        add_term([(i, j, 1)], [(j, k, 1)], [(k, i, 1)], i_off, j_off, k_off)
                        
    decompose(n, m, p, 0, 0, 0)
    
    rank = len(terms)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r, (u_entries, v_entries, w_entries, i_off, j_off, k_off) in enumerate(terms):
        for (li, lj, c) in u_entries:
            gi = i_off + li
            gj = j_off + lj
            U[gi * m + gj, r] = c
        for (lj, lk, c) in v_entries:
            gj = j_off + lj
            gk = k_off + lk
            V[gj * p + gk, r] = c
        for (lk, li, c) in w_entries:
            gk = k_off + lk
            gi = i_off + li
            W[gk * n + gi, r] = c
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
