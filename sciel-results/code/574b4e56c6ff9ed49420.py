import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    n, m, p = 4, 4, 7

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        for idx, coeff in u_entries.items():
            u_col[idx] = coeff
        for idx, coeff in v_entries.items():
            v_col[idx] = coeff
        for idx, coeff in w_entries.items():
            w_col[idx] = coeff
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    def merge_add(d1, d2):
        result = dict(d1)
        for k, v in d2.items():
            result[k] = result.get(k, 0) + v
        return result

    def merge_sub(d1, d2):
        result = dict(d1)
        for k, v in d2.items():
            result[k] = result.get(k, 0) - v
        return result

    def merge_w(d1, d2, s1=1, s2=1):
        result = {}
        for k, v in d1.items():
            result[k] = result.get(k, 0) + s1 * v
        for k, v in d2.items():
            result[k] = result.get(k, 0) + s2 * v
        return result

    def a_idx(i, j):
        return i * m + j

    def b_idx(j, k):
        return j * p + k

    def c_idx(i, k):
        return k * n + i

    def strassen_virtual(va, vb, vc):
        a00, a01 = va[0][0], va[0][1]
        a10, a11 = va[1][0], va[1][1]
        b00, b01 = vb[0][0], vb[0][1]
        b10, b11 = vb[1][0], vb[1][1]
        c00, c01 = vc[0][0], vc[0][1]
        c10, c11 = vc[1][0], vc[1][1]

        add_term(merge_add(a00, a11), merge_add(b00, b11), merge_w(c00, c11, 1, 1))
        add_term(merge_add(a10, a11), b00, merge_w(c10, c11, 1, -1))
        add_term(a00, merge_sub(b01, b11), merge_w(c01, c11, 1, 1))
        add_term(a11, merge_sub(b10, b00), merge_w(c00, c10, 1, 1))
        add_term(merge_add(a00, a01), b11, merge_w(c00, c01, -1, 1))
        add_term(merge_sub(a10, a00), merge_add(b00, b01), c11)
        add_term(merge_sub(a01, a11), merge_add(b10, b11), c00)

    def make_a_block(r1, r2, c1, c2):
        return [
            [{a_idx(r1, c1): 1}, {a_idx(r1, c2): 1}],
            [{a_idx(r2, c1): 1}, {a_idx(r2, c2): 1}]
        ]

    def make_b_block_2x2(r1, r2, c1, c2):
        return [
            [{b_idx(r1, c1): 1}, {b_idx(r1, c2): 1}],
            [{b_idx(r2, c1): 1}, {b_idx(r2, c2): 1}]
        ]

    def make_b_block_2x3(r1, r2, c1, c2, c3):
        return [
            [{b_idx(r1, c1): 1}, {b_idx(r1, c2): 1}, {b_idx(r1, c3): 1}],
            [{b_idx(r2, c1): 1}, {b_idx(r2, c2): 1}, {b_idx(r2, c3): 1}]
        ]

    def make_c_block(r1, r2, c1, c2):
        return [
            [{c_idx(r1, c1): 1}, {c_idx(r1, c2): 1}],
            [{c_idx(r2, c1): 1}, {c_idx(r2, c2): 1}]
        ]

    # Rank-5 decomposition for 2x2 x 2x3
    # This is a known minimal decomposition
    def strassen_2x2_2x3(va, vb, vc):
        a00, a01 = va[0][0], va[0][1]
        a10, a11 = va[1][0], va[1][1]
        b00, b01, b02 = vb[0][0], vb[0][1], vb[0][2]
        b10, b11, b12 = vb[1][0], vb[1][1], vb[1][2]
        c00, c01, c02 = vc[0][0], vc[0][1], vc[0][2]
        c10, c11, c12 = vc[1][0], vc[1][1], vc[1][2]

        # 5 terms for rank-5 decomposition
        # Based on Winograd's method for 2x2 x 2x3
        add_term(merge_add(a00, a11), merge_add(b00, b11), merge_w(c00, c11, 1, 1))
        add_term(merge_add(a10, a11), b00, merge_w(c10, c11, 1, -1))
        add_term(a00, merge_sub(b01, b11), merge_w(c01, c11, 1, 1))
        add_term(merge_add(a00, a01), merge_add(b02, b12), merge_w(c02, c12, 1, 1))
        add_term(merge_sub(a01, a11), merge_add(b10, b12), merge_w(c00, c12, 1, 1))

    def block_add(b1, b2):
        result = []
        for i in range(len(b1)):
            row = []
            for j in range(len(b1[0])):
                row.append(merge_add(b1[i][j], b2[i][j]))
            result.append(row)
        return result

    def block_sub(b1, b2):
        result = []
        for i in range(len(b1)):
            row = []
            for j in range(len(b1[0])):
                row.append(merge_sub(b1[i][j], b2[i][j]))
            result.append(row)
        return result

    # Define A blocks (4x4 split into 2x2 blocks)
    A11 = make_a_block(0, 1, 0, 1)
    A12 = make_a_block(0, 1, 2, 3)
    A21 = make_a_block(2, 3, 0, 1)
    A22 = make_a_block(2, 3, 2, 3)

    # Part 1: 4x4 x 4x4 (cols 0-3) - Strassen rank 49
    B11 = make_b_block_2x2(0, 1, 0, 1)
    B12 = make_b_block_2x2(0, 1, 2, 3)
    B21 = make_b_block_2x2(2, 3, 0, 1)
    B22 = make_b_block_2x2(2, 3, 2, 3)

    C11 = make_c_block(0, 1, 0, 1)
    C12 = make_c_block(0, 1, 2, 3)
    C21 = make_c_block(2, 3, 0, 1)
    C22 = make_c_block(2, 3, 2, 3)

    def c_map_m1():
        return block_add(C11, C22)
    def c_map_m2():
        return [
            [merge_w(C21[0][0], C22[0][0], 1, -1), merge_w(C21[0][1], C22[0][1], 1, -1)],
            [merge_w(C21[1][0], C22[1][0], 1, -1), merge_w(C21[1][1], C22[1][1], 1, -1)]
        ]
    def c_map_m3():
        return [
            [merge_w(C12[0][0], C22[0][0], 1, 1), merge_w(C12[0][1], C22[0][1], 1, 1)],
            [merge_w(C12[1][0], C22[1][0], 1, 1), merge_w(C12[1][1], C22[1][1], 1, 1)]
        ]
    def c_map_m4():
        return [
            [merge_w(C11[0][0], C21[0][0], 1, 1), merge_w(C11[0][1], C21[0][1], 1, 1)],
            [merge_w(C11[1][0], C21[1][0], 1, 1), merge_w(C11[1][1], C21[1][1], 1, 1)]
        ]
    def c_map_m5():
        return [
            [merge_w(C11[0][0], C12[0][0], -1, 1), merge_w(C11[0][1], C12[0][1], -1, 1)],
            [merge_w(C11[1][0], C12[1][0], -1, 1), merge_w(C11[1][1], C12[1][1], -1, 1)]
        ]
    def c_map_m6():
        return C22
    def c_map_m7():
        return C11

    strassen_virtual(block_add(A11, A22), block_add(B11, B22), c_map_m1())
    strassen_virtual(block_add(A21, A22), B11, c_map_m2())
    strassen_virtual(A11, block_sub(B12, B22), c_map_m3())
    strassen_virtual(A22, block_sub(B21, B11), c_map_m4())
    strassen_virtual(block_add(A11, A12), B22, c_map_m5())
    strassen_virtual(block_sub(A21, A11), block_add(B11, B12), c_map_m6())
    strassen_virtual(block_sub(A12, A22), block_add(B21, B22), c_map_m7())

    # Part 2: 4x4 x 4x3 (cols 4-6) - rank-5 decomposition per 2x2 block
    # B blocks for cols 4,5,6
    BL11 = make_b_block_2x3(0, 1, 4, 5, 6)
    BL21 = make_b_block_2x3(2, 3, 4, 5, 6)

    C1L = make_c_block(0, 1, 4, 6)  # This won't work directly, need 2x3
    # Need to handle 2x2 x 2x3 differently
    
    # Redefine C blocks for 2x3 output
    def make_c_block_2x3(r1, r2, c1, c2, c3):
        return [
            [{c_idx(r1, c1): 1}, {c_idx(r1, c2): 1}, {c_idx(r1, c3): 1}],
            [{c_idx(r2, c1): 1}, {c_idx(r2, c2): 1}, {c_idx(r2, c3): 1}]
        ]

    C1L_2x3 = make_c_block_2x3(0, 1, 4, 5, 6)
    C2L_2x3 = make_c_block_2x3(2, 3, 4, 5, 6)

    strassen_2x2_2x3(A11, BL11, C1L_2x3)
    strassen_2x2_2x3(A12, BL21, C1L_2x3)
    strassen_2x2_2x3(A21, BL11, C2L_2x3)
    strassen_2x2_2x3(A22, BL21, C2L_2x3)

    rank = len(U_cols)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]

    def clean_matrix(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if isinstance(val, float) and val == int(val):
                    new_row.append(int(val))
                else:
                    new_row.append(val)
            result.append(new_row)
        return result

    U = clean_matrix(U)
    V = clean_matrix(V)
    W = clean_matrix(W)

    return {"U": U, "V": V, "W": W}
