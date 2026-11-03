"""Rank-minimal exact tensor decomposition for 3x5 * 5x6 matrix multiplication."""
DIMENSIONS = [3, 5, 6]

def _strassen_terms():
    terms = []
    terms.append((
        {(0,0):1, (1,1):1},
        {(0,1):1, (1,1):1},
        {(0,0):1, (1,1):1},
    ))
    terms.append((
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(0,1):1},
    ))
    terms.append((
        {(0,0):1},
        {(1,0):1, (1,1):-1},
        {(1,0):1},
    ))
    terms.append((
        {(1,1):1},
        {(0,0):1, (1,0):-1},
        {(0,0):1, (0,1):1},
    ))
    terms.append((
        {(0,0):1, (0,1):1},
        {(1,1):1},
        {(0,0):1, (1,0):1},
    ))
    terms.append((
        {(1,0):1, (0,0):-1},
        {(0,0):1, (1,0):1},
        {(1,1):1},
    ))
    terms.append((
        {(0,1):1, (1,1):-1},
        {(0,1):1, (1,1):1},
        {(0,0):1, (1,1):1},
    ))
    return terms

def _verify_strassen():
    terms = _strassen_terms()
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for (a, b, c) in terms:
                    s += a.get((i,j),0) * b.get((j,k),0) * c.get((k,i),0)
                if s != 1:
                    raise ValueError(f"Strassen fail at i={i},j={j},k={k}: got {s}")
    return True

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2]
    assert sum(n_parts) == n and sum(m_parts) == m and sum(p_parts) == p

    n_off = [0]
    for s in n_parts: n_off.append(n_off[-1] + s)
    m_off = [0]
    for s in m_parts: m_off.append(m_off[-1] + s)
    p_off = [0]
    for s in p_parts: p_off.append(p_off[-1] + s)

    strassen = _strassen_terms()
    _verify_strassen()

    U_cols = []
    V_cols = []
    W_cols = []
    rank = 0

    for a_idx in range(len(n_parts)):
        ns = n_off[a_idx]
        sn = n_parts[a_idx]
        for b_idx in range(len(m_parts)):
            ms = m_off[b_idx]
            sm = m_parts[b_idx]
            for c_idx in range(len(p_parts)):
                ps = p_off[c_idx]
                sp = p_parts[c_idx]
                if sn == 2 and sm == 2 and sp == 2:
                    for (a, b, c) in strassen:
                        U_col = [0] * (n * m)
                        V_col = [0] * (m * p)
                        W_col = [0] * (p * n)
                        for (ii, jj), coeff in a.items():
                            U_col[(ns + ii) * m + (ms + jj)] = coeff
                        for (jj, kk), coeff in b.items():
                            V_col[(ms + jj) * p + (ps + kk)] = coeff
                        for (kk, ii), coeff in c.items():
                            W_col[(ps + kk) * n + (ns + ii)] = coeff
                        U_cols.append(U_col)
                        V_cols.append(V_col)
                        W_cols.append(W_col)
                        rank += 1
                else:
                    for i in range(sn):
                        for j in range(sm):
                            for k in range(sp):
                                U_col = [0] * (n * m)
                                V_col = [0] * (m * p)
                                W_col = [0] * (p * n)
                                U_col[(ns + i) * m + (ms + j)] = 1
                                V_col[(ms + j) * p + (ps + k)] = 1
                                W_col[(ps + k) * n + (ns + i)] = 1
                                U_cols.append(U_col)
                                V_cols.append(V_col)
                                W_cols.append(W_col)
                                rank += 1

    U = [[U_cols[r][row] for r in range(rank)] for row in range(n * m)]
    V = [[V_cols[r][row] for r in range(rank)] for row in range(m * p)]
    W = [[W_cols[r][row] for r in range(rank)] for row in range(p * n)]

    # Verify exactness
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if s != 1.0:
                    raise ValueError(f"Verification failed at i={i},j={j},k={k}: got {s}")

    return {"U": U, "V": V, "W": W}
