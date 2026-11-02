import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # (2,2,2) Strassen decomposition
    strassen_terms = [
        {'x': {(0,0):1, (1,1):1}, 'y': {(0,0):1, (1,1):1}, 'z': {(0,0):1, (1,1):1}},
        {'x': {(1,0):1, (1,1):1}, 'y': {(0,0):1}, 'z': {(1,0):1, (1,1):-1}},
        {'x': {(0,0):1}, 'y': {(0,1):1, (1,1):-1}, 'z': {(0,1):1, (1,1):1}},
        {'x': {(1,1):1}, 'y': {(1,0):1, (0,0):-1}, 'z': {(0,0):1, (1,0):1}},
        {'x': {(0,0):1, (0,1):1}, 'y': {(1,1):1}, 'z': {(0,0):-1, (0,1):1}},
        {'x': {(1,0):1, (0,0):-1}, 'y': {(0,0):1, (0,1):1}, 'z': {(1,1):1}},
        {'x': {(0,1):1, (1,1):-1}, 'y': {(1,0):1, (1,1):1}, 'z': {(0,0):1}},
    ]

    # (1,2,4) rank-2 decomposition
    # n_inner=1, m_inner=2, p_inner=4
    # Term s=0: A[0,0] * B[0,k] -> C[0,k] for k=0,1
    # Term s=1: A[0,1] * B[1,k] -> C[0,k] for k=2,3
    inner_terms = [
        {'x': {0: 1}, 'y': {(0,0):1, (0,1):1}, 'z': {0:1, 1:1}},
        {'x': {1: 1}, 'y': {(1,2):1, (1,3):1}, 'z': {2:1, 3:1}},
    ]

    rank = 7 * 2
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    col = 0
    for r in range(7):
        for s in range(2):
            term_r = strassen_terms[r]
            term_s = inner_terms[s]
            # U: index is i_outer * m + j where j = j_outer * 2 + j_inner
            for (i_outer, j_outer), coeff_x in term_r['x'].items():
                for (j_inner), coeff_xs in term_s['x'].items():
                    j_full = j_outer * 2 + j_inner
                    u_idx = i_outer * m + j_full
                    U[u_idx, col] += coeff_x * coeff_xs
            # V: index is j_full * p + k_full where j_full = j_outer*2+j_inner, k_full = k_outer*4+k_inner
            for (j_outer, k_outer), coeff_y in term_r['y'].items():
                for (j_inner, k_inner), coeff_ys in term_s['y'].items():
                    j_full = j_outer * 2 + j_inner
                    k_full = k_outer * 4 + k_inner
                    v_idx = j_full * p + k_full
                    V[v_idx, col] += coeff_y * coeff_ys
            # W: index is k_full * n + i_outer where k_full = k_outer*4+k_inner
            for (k_outer, i_outer), coeff_z in term_r['z'].items():
                for (k_inner), coeff_zs in term_s['z'].items():
                    k_full = k_outer * 4 + k_inner
                    w_idx = k_full * n + i_outer
                    W[w_idx, col] += coeff_z * coeff_zs
            col += 1

    # Quick verification
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
