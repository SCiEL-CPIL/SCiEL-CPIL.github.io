import numpy as np
import time

def construct():
    n, m, p = 4, 5, 5
    deadline = time.time() + 140

    a_row_blocks = [(0, 2), (2, 4)]
    a_col_blocks = [(0, 2), (2, 4), (4, 5)]
    b_row_blocks = [(0, 2), (2, 4), (4, 5)]
    b_col_blocks = [(0, 2), (2, 4), (4, 5)]
    c_row_blocks = [(0, 2), (2, 4)]
    c_col_blocks = [(0, 2), (2, 4), (4, 5)]

    U_cols = []
    V_cols = []
    W_cols = []

    for ib in range(2):
        for jb in range(3):
            for kb in range(3):
                a_r_s, a_r_e = a_row_blocks[ib]
                a_c_s, a_c_e = a_col_blocks[jb]
                b_r_s, b_r_e = b_row_blocks[jb]
                b_c_s, b_c_e = b_col_blocks[kb]
                c_r_s, c_r_e = c_row_blocks[ib]
                c_c_s, c_c_e = c_col_blocks[kb]

                a_h = a_r_e - a_r_s
                a_w = a_c_e - a_c_s
                b_h = b_r_e - b_r_s
                b_w = b_c_e - b_c_s
                c_h = c_r_e - c_r_s
                c_w = c_c_e - c_c_s

                assert a_w == b_h

                if a_h == 2 and a_w == 2 and b_w == 2:
                    terms = strassen_2x2()
                else:
                    terms = schoolbook(a_h, a_w, b_w)

                for u_vec, v_vec, w_vec in terms:
                    U_col = [0.0] * (n * m)
                    V_col = [0.0] * (m * p)
                    W_col = [0.0] * (p * n)

                    for idx, val in enumerate(u_vec):
                        lr, lc = divmod(idx, a_w)
                        U_col[(a_r_s + lr) * m + (a_c_s + lc)] = val

                    for idx, val in enumerate(v_vec):
                        lr, lc = divmod(idx, b_w)
                        V_col[(b_r_s + lr) * p + (b_c_s + lc)] = val

                    for idx, val in enumerate(w_vec):
                        lr, lc = divmod(idx, c_w)
                        W_col[(c_c_s + lc) * n + (c_r_s + lr)] = val

                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)

                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    if not U_cols:
        return schoolbook_full(n, m, p)

    rank = len(U_cols)
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T

    if not verify(U, V, W, n, m, p):
        return schoolbook_full(n, m, p)

    return {"U": U, "V": V, "W": W}

def strassen_2x2():
    terms = []
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    terms.append(([0,0,0,1], [0,0,1,-1], [-1,0,1,0]))
    terms.append(([1,1,0,0], [0,0,0,1], [0,1,0,0]))
    terms.append(([1,0,-1,0], [1,1,0,0], [1,0,0,0]))
    terms.append(([0,1,0,-1], [0,0,1,1], [0,0,0,1]))
    return terms

def schoolbook(a_h, a_w, b_w):
    terms = []
    for i in range(a_h):
        for j in range(a_w):
            for k in range(b_w):
                u_vec = [0.0] * (a_h * a_w)
                v_vec = [0.0] * (a_w * b_w)
                w_vec = [0.0] * (a_h * b_w)
                u_vec[i * a_w + j] = 1.0
                v_vec[j * b_w + k] = 1.0
                w_vec[i * b_w + k] = 1.0
                terms.append((u_vec, v_vec, w_vec))
    return terms

def schoolbook_full(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def verify(U, V, W, n, m, p):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True
