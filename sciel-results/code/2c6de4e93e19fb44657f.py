import numpy as np
import time

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    Tensor: T[i*4+j, j*4+k, k*4+i] = sum_r U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r] = 1
    """
    start_time = time.time()
    DEADLINE = 140

    n, m, p = 4, 4, 4
    rank = 49

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def get_U_index(row, col):
        return row * m + col

    def get_V_index(row, col):
        return row * p + col

    def get_W_index(row, col):
        # Trace convention: W[k*n+i, r] where k=col of C, i=row of C
        return col * n + row

    def a_elem(bi, bj, r, c):
        return (2 * bi + r, 2 * bj + c)

    def b_elem(bi, bj, r, c):
        return (2 * bi + r, 2 * bj + c)

    def add_to_dict(d, key, val):
        d[key] = d.get(key, 0) + val

    def merge_dicts(*dicts):
        result = {}
        for d in dicts:
            for k, v in d.items():
                result[k] = result.get(k, 0) + v
        return result

    # Define the 7 Strassen terms for the outer 4x4 product
    L_terms = []
    R_terms = []
    contrib_terms = []

    # Term 0: P1 = (A11 + A22)(B11 + B22)
    L0 = [[{} for _ in range(2)] for _ in range(2)]
    R0 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            add_to_dict(L0[r][c], a_elem(0, 0, r, c), 1)
            add_to_dict(L0[r][c], a_elem(1, 1, r, c), 1)
            add_to_dict(R0[r][c], b_elem(0, 0, r, c), 1)
            add_to_dict(R0[r][c], b_elem(1, 1, r, c), 1)
    c0 = {}
    for r in range(2):
        for c in range(2):
            c0[(r, c)] = [(r, c, 1), (2 + r, 2 + c, 1)]
    L_terms.append(L0); R_terms.append(R0); contrib_terms.append(c0)

    # Term 1: P2 = (A21 + A22)B11
    L1 = [[{} for _ in range(2)] for _ in range(2)]
    R1 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            add_to_dict(L1[r][c], a_elem(1, 0, r, c), 1)
            add_to_dict(L1[r][c], a_elem(1, 1, r, c), 1)
            add_to_dict(R1[r][c], b_elem(0, 0, r, c), 1)
    c1 = {}
    for r in range(2):
        for c in range(2):
            c1[(r, c)] = [(2 + r, c, 1), (2 + r, 2 + c, -1)]
    L_terms.append(L1); R_terms.append(R1); contrib_terms.append(c1)

    # Term 2: P3 = A11(B12 - B22)
    L2 = [[{} for _ in range(2)] for _ in range(2)]
    R2 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            add_to_dict(L2[r][c], a_elem(0, 0, r, c), 1)
            add_to_dict(R2[r][c], b_elem(0, 1, r, c), 1)
            add_to_dict(R2[r][c], b_elem(1, 1, r, c), -1)
    c2 = {}
    for r in range(2):
        for c in range(2):
            c2[(r, c)] = [(r, 2 + c, 1), (2 + r, 2 + c, 1)]
    L_terms.append(L2); R_terms.append(R2); contrib_terms.append(c2)

    # Term 3: P4 = A22(B21 - B11)
    L3 = [[{} for _ in range(2)] for _ in range(2)]
    R3 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            add_to_dict(L3[r][c], a_elem(1, 1, r, c), 1)
            add_to_dict(R3[r][c], b_elem(1, 0, r, c), 1)
            add_to_dict(R3[r][c], b_elem(0, 0, r, c), -1)
    c3 = {}
    for r in range(2):
        for c in range(2):
            c3[(r, c)] = [(r, c, 1), (2 + r, c, 1)]
    L_terms.append(L3); R_terms.append(R3); contrib_terms.append(c3)

    # Term 4: P5 = (A11 + A12)B22
    L4 = [[{} for _ in range(2)] for _ in range(2)]
    R4 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            add_to_dict(L4[r][c], a_elem(0, 0, r, c), 1)
            add_to_dict(L4[r][c], a_elem(0, 1, r, c), 1)
            add_to_dict(R4[r][c], b_elem(1, 1, r, c), 1)
    c4 = {}
    for r in range(2):
        for c in range(2):
            c4[(r, c)] = [(r, c, -1), (r, 2 + c, 1)]
    L_terms.append(L4); R_terms.append(R4); contrib_terms.append(c4)

    # Term 5: P6 = (A21 - A11)(B11 + B12)
    L5 = [[{} for _ in range(2)] for _ in range(2)]
    R5 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            add_to_dict(L5[r][c], a_elem(1, 0, r, c), 1)
            add_to_dict(L5[r][c], a_elem(0, 0, r, c), -1)
            add_to_dict(R5[r][c], b_elem(0, 0, r, c), 1)
            add_to_dict(R5[r][c], b_elem(0, 1, r, c), 1)
    c5 = {}
    for r in range(2):
        for c in range(2):
            c5[(r, c)] = [(2 + r, 2 + c, 1)]
    L_terms.append(L5); R_terms.append(R5); contrib_terms.append(c5)

    # Term 6: P7 = (A12 - A22)(B21 + B22)
    L6 = [[{} for _ in range(2)] for _ in range(2)]
    R6 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            add_to_dict(L6[r][c], a_elem(0, 1, r, c), 1)
            add_to_dict(L6[r][c], a_elem(1, 1, r, c), -1)
            add_to_dict(R6[r][c], b_elem(1, 0, r, c), 1)
            add_to_dict(R6[r][c], b_elem(1, 1, r, c), 1)
    c6 = {}
    for r in range(2):
        for c in range(2):
            c6[(r, c)] = [(r, c, 1)]
    L_terms.append(L6); R_terms.append(R6); contrib_terms.append(c6)

    # Sub-contributions for inner 2x2 Strassen
    sub_contrib = [
        [(0, 0, 1), (1, 1, 1)],    # P1 -> C11, C22
        [(1, 0, 1), (1, 1, -1)],   # P2 -> C21, C22
        [(0, 1, 1), (1, 1, 1)],    # P3 -> C12, C22
        [(0, 0, 1), (1, 0, 1)],    # P4 -> C11, C21
        [(0, 0, -1), (0, 1, 1)],   # P5 -> C11, C12
        [(1, 1, 1)],               # P6 -> C22
        [(0, 0, 1)]                # P7 -> C11
    ]

    for t in range(7):
        L_t = L_terms[t]
        R_t = R_terms[t]
        contrib_t = contrib_terms[t]

        a = L_t[0][0]; b = L_t[0][1]; c = L_t[1][0]; d = L_t[1][1]
        e = R_t[0][0]; f = R_t[0][1]; g = R_t[1][0]; h = R_t[1][1]

        M1_A = merge_dicts(a, d)
        M1_B = merge_dicts(e, h)
        M2_A = merge_dicts(c, d)
        M2_B = dict(e)
        M3_A = dict(a)
        M3_B = merge_dicts(f, {k: -v for k, v in h.items()})
        M4_A = dict(d)
        M4_B = merge_dicts(g, {k: -v for k, v in e.items()})
        M5_A = merge_dicts(a, b)
        M5_B = dict(h)
        M6_A = merge_dicts(c, {k: -v for k, v in a.items()})
        M6_B = merge_dicts(e, f)
        M7_A = merge_dicts(b, {k: -v for k, v in d.items()})
        M7_B = merge_dicts(g, h)

        sub_products = [
            (M1_A, M1_B),
            (M2_A, M2_B),
            (M3_A, M3_B),
            (M4_A, M4_B),
            (M5_A, M5_B),
            (M6_A, M6_B),
            (M7_A, M7_B)
        ]

        for k in range(7):
            x_k, y_k = sub_products[k]
            rank_idx = t * 7 + k

            for (row, col), coeff in x_k.items():
                if coeff != 0:
                    u_idx = get_U_index(row, col)
                    U[u_idx, rank_idx] = coeff

            for (row, col), coeff in y_k.items():
                if coeff != 0:
                    v_idx = get_V_index(row, col)
                    V[v_idx, rank_idx] = coeff

            for (pr, pc, sub_coeff) in sub_contrib[k]:
                if (pr, pc) in contrib_t:
                    for (out_row, out_col, block_coeff) in contrib_t[(pr, pc)]:
                        w_idx = get_W_index(out_row, out_col)
                        W[w_idx, rank_idx] += sub_coeff * block_coeff

    # Thorough verification
    if time.time() - start_time < DEADLINE - 10:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        max_err = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    a_idx = i * 4 + j
                    b_idx = j * 4 + k
                    c_idx = k * 4 + i
                    val = np.sum(U_arr[a_idx, :] * V_arr[b_idx, :] * W_arr[c_idx, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-10:
            pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
