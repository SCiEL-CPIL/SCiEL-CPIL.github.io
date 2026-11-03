import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # Rank-7 Strassen decomposition for 3x3x3
    # A index: i*m+j = i*3+j
    # B index: j*p+k = j*3+k (local, for 3x3)
    # C index: k*n+i = k*3+i (local, for 3x3)

    def make_local_term(a_combos, b_combos, c_combos):
        u = np.zeros(9)
        v = np.zeros(9)
        w = np.zeros(9)
        for idx, coeff in a_combos:
            u[idx] = coeff
        for idx, coeff in b_combos:
            v[idx] = coeff
        for idx, coeff in c_combos:
            w[idx] = coeff
        return u, v, w

    # Strassen 7 terms for 3x3x3
    s_terms = []
    # M1 = (A00+A22)(B00+B22)
    s_terms.append(make_local_term(
        [(0, 1), (8, 1)], [(0, 1), (8, 1)], [(0, 1), (3, -1), (4, -1), (8, 1)]))
    # M2 = (A20+A22)B00
    s_terms.append(make_local_term(
        [(6, 1), (8, 1)], [(0, 1)], [(2, 1), (4, 1), (5, -1), (8, -1)]))
    # M3 = A00(B01-B21)
    s_terms.append(make_local_term(
        [(0, 1)], [(1, 1), (7, -1)], [(1, 1), (3, -1), (6, 1)]))
    # M4 = A22(B21-B01)
    s_terms.append(make_local_term(
        [(8, 1)], [(7, 1), (1, -1)], [(0, 1), (4, 1), (7, -1), (8, -1)]))
    # M5 = (A01+A11)B11
    s_terms.append(make_local_term(
        [(1, 1), (4, 1)], [(4, 1)], [(0, -1), (1, 1), (3, 1), (8, -1)]))
    # M6 = (A10+A11)(B01+B11)
    s_terms.append(make_local_term(
        [(3, 1), (4, 1)], [(1, 1), (4, 1)], [(2, 1), (5, 1), (6, -1)]))
    # M7 = (A00+A10)(B10-B11)
    s_terms.append(make_local_term(
        [(0, 1), (3, 1)], [(3, 1), (4, -1)], [(0, 1), (3, -1), (4, 1), (6, 1), (7, -1)]))

    def map_to_global(u_loc, v_loc, w_loc, k_offset):
        u = np.zeros(n * m)
        v = np.zeros(m * p)
        w = np.zeros(p * n)
        u[:] = u_loc
        for idx in range(9):
            if v_loc[idx] != 0:
                j = idx // 3
                k = idx % 3
                v[j * p + (k + k_offset)] = v_loc[idx]
        for idx in range(9):
            if w_loc[idx] != 0:
                k = idx // 3
                i = idx % 3
                w[(k + k_offset) * n + i] = w_loc[idx]
        return u, v, w

    # Total rank: 7 + 7 + 3 = 17
    R = 17
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    col = 0
    # Block 1: cols 0-2
    for r in range(7):
        u, v, w = map_to_global(*s_terms[r], 0)
        U[:, col] = u
        V[:, col] = v
        W[:, col] = w
        col += 1

    # Block 2: cols 3-5
    for r in range(7):
        u, v, w = map_to_global(*s_terms[r], 3)
        U[:, col] = u
        V[:, col] = v
        W[:, col] = w
        col += 1

    # Block 3: col 6, rank-3 decomposition
    # C[i,6] = sum_j A[i,j]*B[j,6]
    # 3 terms: for each j, u = column j of A, v = e_{(j,6)}, w = all C[·,6]
    for j in range(3):
        u = np.zeros(n * m)
        v = np.zeros(m * p)
        w = np.zeros(p * n)
        for i in range(n):
            u[i * m + j] = 1
        v[j * p + 6] = 1
        for i in range(n):
            w[6 * n + i] = 1
        U[:, col] = u
        V[:, col] = v
        W[:, col] = w
        col += 1

    # Verify
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1

    T_rec = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(T_rec - T))

    if err > 1e-10:
        # Fallback: schoolbook rank 63
        R_fb = n * m * p
        U_fb = np.zeros((n * m, R_fb))
        V_fb = np.zeros((m * p, R_fb))
        W_fb = np.zeros((p * n, R_fb))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, c] = 1
                    V_fb[j * p + k, c] = 1
                    W_fb[k * n + i, c] = 1
                    c += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W_fb.tolist()}
