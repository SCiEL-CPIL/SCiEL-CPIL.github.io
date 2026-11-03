import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    total_rank = 0
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                if ns == 2 and ms == 2 and ps == 2:
                    total_rank += 7
                else:
                    total_rank += ns * ms * ps
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    n_offset = 0
    m_offset = 0
    p_offset = 0
    
    for ni, ns in enumerate(n_splits):
        n_offset = 0
        for mi, ms in enumerate(m_splits):
            m_offset = 0
            for pi, ps in enumerate(p_splits):
                p_offset = 0
                if ns == 2 and ms == 2 and ps == 2:
                    block_rank = 7
                else:
                    block_rank = ns * ms * ps
                
                for r in range(block_rank):
                    if ns == 2 and ms == 2 and ps == 2:
                        add_strassen_term(U, V, W, n_offset, m_offset, p_offset, r, col_offset + r, n, m, p)
                    else:
                        i = r // (ms * ps)
                        j = (r // ps) % ms
                        k = r % ps
                        add_schoolbook_term(U, V, W, n_offset, m_offset, p_offset, i, j, k, col_offset + r, n, m, p)
                
                col_offset += block_rank
                p_offset += ps
            m_offset += ms
        n_offset += ns
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def add_strassen_term(U, V, W, n_off, m_off, p_off, term_idx, col, n, m, p):
    # U[i, j]
    if term_idx == 0:
        U[n_off * m + m_off, col] = 1
        U[(n_off+1) * m + (m_off+1), col] = 1
    elif term_idx == 1:
        U[n_off * m + (m_off+1), col] = 1
        U[(n_off+1) * m + (m_off+1), col] = 1
    elif term_idx == 2:
        U[n_off * m + m_off, col] = 1
    elif term_idx == 3:
        U[(n_off+1) * m + (m_off+1), col] = 1
    elif term_idx == 4:
        U[n_off * m + m_off, col] = 1
        U[n_off * m + (m_off+1), col] = 1
    elif term_idx == 5:
        U[n_off * m + (m_off+1), col] = 1
        U[n_off * m + m_off, col] = -1
    elif term_idx == 6:
        U[n_off * m + (m_off+1), col] = 1
        U[(n_off+1) * m + (m_off+1), col] = -1

    # V[j, k]
    if term_idx == 0:
        V[m_off * p + p_off, col] = 1
        V[(m_off+1) * p + (p_off+1), col] = 1
    elif term_idx == 1:
        V[m_off * p + p_off, col] = 1
    elif term_idx == 2:
        V[(m_off+1) * p + p_off, col] = 1
        V[(m_off+1) * p + (p_off+1), col] = -1
    elif term_idx == 3:
        V[m_off * p + (p_off+1), col] = 1
        V[m_off * p + p_off, col] = -1
    elif term_idx == 4:
        V[(m_off+1) * p + (p_off+1), col] = 1
    elif term_idx == 5:
        V[m_off * p + p_off, col] = 1
        V[m_off * p + (p_off+1), col] = 1
    elif term_idx == 6:
        V[(m_off+1) * p + p_off, col] = 1
        V[(m_off+1) * p + (p_off+1), col] = 1

    # W[k, i]
    if term_idx == 0:
        W[p_off * n + n_off, col] = 1
        W[(p_off+1) * n + (n_off+1), col] = 1
    elif term_idx == 1:
        W[p_off * n + (n_off+1), col] = 1
    elif term_idx == 2:
        W[(p_off+1) * n + n_off, col] = 1
        W[(p_off+1) * n + (n_off+1), col] = 1
    elif term_idx == 3:
        W[p_off * n + n_off, col] = 1
        W[p_off * n + (n_off+1), col] = 1
    elif term_idx == 4:
        W[(p_off+1) * n + n_off, col] = 1
        W[p_off * n + n_off, col] = -1
    elif term_idx == 5:
        W[(p_off+1) * n + (n_off+1), col] = 1
    elif term_idx == 6:
        W[p_off * n + n_off, col] = 1

def add_schoolbook_term(U, V, W, n_off, m_off, p_off, i, j, k, col, n, m, p):
    U[(n_off + i) * m + (m_off + j), col] = 1
    V[(m_off + j) * p + (p_off + k), col] = 1
    W[(p_off + k) * n + (n_off + i), col] = 1

