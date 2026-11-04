"""Tensor decomposition for 4x4 * 4x7 matrix multiplication.

Uses Strassen 2x2x2 (rank 7) for 2-column output blocks, schoolbook for 1-column block.
Rank = 2 row_blocks * 3 two-col_blocks * 2 A-col_blocks * 7 + 2 * 1 * 2 * 2*2 = 84 + 8 = 92.
"""
import time

DIMENSIONS = [4, 4, 7]

def _strassen_terms():
    """Return 7 Strassen terms for 2x2x2, each with local (row, coeff) for U(2x2), V(2x2), W(2x2).
    U row = i*2+j, V row = j*2+k, W row = k*2+i.
    """
    return [
        # M0 = (A00+A11)(B00+B11) -> C00,C01,C10,C11 all +1
        {'u': [(0,1),(3,1)], 'v': [(0,1),(3,1)], 'w': [(0,1),(1,1),(2,1),(3,1)]},
        # M1 = (A10+A11)B00 -> C10,C11
        {'u': [(2,1),(3,1)], 'v': [(0,1)], 'w': [(2,1),(3,1)]},
        # M2 = A00(B10-B00) -> C01
        {'u': [(0,1)], 'v': [(2,1),(0,-1)], 'w': [(1,1)]},
        # M3 = A11(B01-B11) -> C10,C11
        {'u': [(3,1)], 'v': [(1,1),(3,-1)], 'w': [(2,1),(3,1)]},
        # M4 = (A00+A10)B11 -> C00,C01
        {'u': [(0,1),(2,1)], 'v': [(3,1)], 'w': [(0,1),(1,1)]},
        # M5 = (A10-A00)(B01+B11) -> -C00
        {'u': [(2,1),(0,-1)], 'v': [(1,1),(3,1)], 'w': [(0,-1)]},
        # M6 = (A01-A11)(B00+B10) -> -C00
        {'u': [(1,1),(3,-1)], 'v': [(0,1),(2,1)], 'w': [(0,-1)]},
    ]

def construct():
    start = time.time()
    deadline = start + 110

    n, m, p = DIMENSIONS  # 4, 4, 7
    strassen = _strassen_terms()

    U_rows = {}
    V_rows = {}
    W_rows = {}

    def add(u_entries, v_entries, w_entries, col):
        for (r, c) in u_entries:
            U_rows.setdefault(r, []).append((col, c))
        for (r, c) in v_entries:
            V_rows.setdefault(r, []).append((col, c))
        for (r, c) in w_entries:
            W_rows.setdefault(r, []).append((col, c))

    col_idx = 0
    for i_off in [0, 2]:
        for k_off, ncol in [(0,2),(2,2),(4,2),(6,1)]:
            for j_off in [0, 2]:
                if ncol == 2:
                    for term in strassen:
                        u_g = [((i_off+i)*4 + (j_off+j), c) for (i,j),c in term['u']]
                        v_g = [((j_off+j)*7 + (k_off+k), c) for (j,k),c in term['v']]
                        w_g = [((k_off+k)*4 + (i_off+i), c) for (k,i),c in term['w']]
                        add(u_g, v_g, w_g, col_idx)
                        col_idx += 1
                else:
                    for di in range(2):
                        for dj in range(2):
                            i = i_off + di
                            j = j_off + dj
                            k = k_off
                            add([(i*4+j, 1)], [(j*7+k, 1)], [(k*4+i, 1)], col_idx)
                            col_idx += 1

    rank = col_idx

    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for r, es in U_rows.items():
        for c, v in es:
            U[r][c] = v
    for r, es in V_rows.items():
        for c, v in es:
            V[r][c] = v
    for r, es in W_rows.items():
        for c, v in es:
            W[r][c] = v

    return {"U": U, "V": V, "W": W}
