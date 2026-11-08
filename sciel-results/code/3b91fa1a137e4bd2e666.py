"""Strassen-based rank-108 decomposition for 4x5x6 matrix multiplication."""

def _strassen_222_terms():
    """Return 7 (u,v,w) triples for the 2x2x2 Strassen decomposition.
    u,v are 4-vectors (indices i*2+j, j*2+k), w is 4-vector (indices k*2+i)."""
    # Each: (u_coeffs_for_A, v_coeffs_for_B, W_matrix_as_[C00,C10,C01,C11])
    raw = [
        # M1
        ((1,0,0,1), (1,0,0,1), (1,0,0,1)),
        # M2
        ((0,0,1,1), (1,0,0,0), (0,1,0,-1)),
        # M3
        ((1,0,0,0), (0,1,0,-1), (0,0,1,1)),
        # M4
        ((0,0,0,1), (-1,0,1,0), (1,1,0,0)),
        # M5
        ((1,1,0,0), (0,0,0,1), (-1,0,1,0)),
        # M6
        ((-1,0,1,0), (1,1,0,0), (0,0,0,1)),
        # M7
        ((0,1,0,-1), (0,0,1,1), (1,0,0,0)),
    ]
    return raw

def construct():
    n, m, p = 4, 5, 6
    terms = []  # list of (gu, gv, gw) global vectors

    def add_local(u_loc, v_loc, w_loc, i_off, j_off, k_off, nb, mb, pb):
        gu = [0]*(n*m)
        gv = [0]*(m*p)
        gw = [0]*(p*n)
        for i in range(nb):
            for j in range(mb):
                gu[(i_off+i)*m + (j_off+j)] = u_loc[i*mb+j]
        for j in range(mb):
            for k in range(pb):
                gv[(j_off+j)*p + (k_off+k)] = v_loc[j*pb+k]
        for k in range(pb):
            for i in range(nb):
                gw[(k_off+k)*n + (i_off+i)] = w_loc[k*nb+i]
        terms.append((gu, gv, gw))

    strassen = _strassen_222_terms()
    # 12 Strassen (2,2,2) blocks
    for i_off in (0, 2):
        for j_off in (0, 2):
            for k_off in (0, 2, 4):
                for (u, v, w) in strassen:
                    add_local(list(u), list(v), list(w), i_off, j_off, k_off, 2, 2, 2)

    # Schoolbook for j=4 (uncovered)
    for i in range(n):
        for k in range(p):
            gu = [0]*(n*m)
            gv = [0]*(m*p)
            gw = [0]*(p*n)
            gu[i*m + 4] = 1
            gv[4*p + k] = 1
            gw[k*n + i] = 1
            terms.append((gu, gv, gw))

    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for r, (gu, gv, gw) in enumerate(terms):
        for idx in range(n*m):
            U[idx][r] = gu[idx]
        for idx in range(m*p):
            V[idx][r] = gv[idx]
        for idx in range(p*n):
            W[idx][r] = gw[idx]

    return {"U": U, "V": V, "W": W}
