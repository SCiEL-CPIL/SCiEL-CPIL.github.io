import time

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
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
    
    def schoolbook_2x2_2x1(va, vb, vc):
        for i in range(2):
            for j in range(2):
                add_term(va[i][j], vb[j], vc[i])
    
    def make_a_block(r1, r2, c1, c2):
        return [
            [{a_idx(r1, c1): 1}, {a_idx(r1, c2): 1}],
            [{a_idx(r2, c1): 1}, {a_idx(r2, c2): 1}]
        ]
    
    def make_b_block(r1, r2, c1, c2):
        return [
            [{b_idx(r1, c1): 1}, {b_idx(r1, c2): 1}],
            [{b_idx(r2, c1): 1}, {b_idx(r2, c2): 1}]
        ]
    
    def make_c_block(r1, r2, c1, c2):
        return [
            [{c_idx(r1, c1): 1}, {c_idx(r1, c2): 1}],
            [{c_idx(r2, c1): 1}, {c_idx(r2, c2): 1}]
        ]
    
    A11 = make_a_block(0, 1, 0, 1)
    A12 = make_a_block(0, 1, 2, 3)
    A21 = make_a_block(2, 3, 0, 1)
    A22 = make_a_block(2, 3, 2, 3)
    
    B11 = make_b_block(0, 1, 0, 1)
    B12 = make_b_block(0, 1, 2, 3)
    B21 = make_b_block(2, 3, 0, 1)
    B22 = make_b_block(2, 3, 2, 3)
    
    C11 = make_c_block(0, 1, 0, 1)
    C12 = make_c_block(0, 1, 2, 3)
    C21 = make_c_block(2, 3, 0, 1)
    C22 = make_c_block(2, 3, 2, 3)
    
    def block_add(b1, b2):
        return [
            [merge_add(b1[0][0], b2[0][0]), merge_add(b1[0][1], b2[0][1])],
            [merge_add(b1[1][0], b2[1][0]), merge_add(b1[1][1], b2[1][1])]
        ]
    
    def block_sub(b1, b2):
        return [
            [merge_sub(b1[0][0], b2[0][0]), merge_sub(b1[0][1], b2[0][1])],
            [merge_sub(b1[1][0], b2[1][0]), merge_sub(b1[1][1], b2[1][1])]
        ]
    
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
    
    if time.time() > deadline:
        rank = len(U_cols)
        U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
        V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
        W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
        return {"U": U, "V": V, "W": W}
    
    # For the 4x4x3 part (columns 4,5,6 of B), use a more efficient decomposition
    # Split into (2x2)*(2x3) blocks
    # Use Strassen for the first 2 columns, schoolbook for the 3rd
    
    BL11 = make_b_block(0, 1, 4, 5)
    BL21 = make_b_block(2, 3, 4, 5)
    C1L = make_c_block(0, 1, 4, 5)
    C2L = make_c_block(2, 3, 4, 5)
    
    strassen_virtual(A11, BL11, C1L)
    strassen_virtual(A12, BL21, C1L)
    strassen_virtual(A21, BL11, C2L)
    strassen_virtual(A22, BL21, C2L)
    
    if time.time() > deadline:
        rank = len(U_cols)
        U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
        V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
        W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
        return {"U": U, "V": V, "W": W}
    
    BR_top = [{b_idx(0, 6): 1}, {b_idx(1, 6): 1}]
    BR_bot = [{b_idx(2, 6): 1}, {b_idx(3, 6): 1}]
    CR_top = [{c_idx(0, 6): 1}, {c_idx(1, 6): 1}]
    CR_bot = [{c_idx(2, 6): 1}, {c_idx(3, 6): 1}]
    
    schoolbook_2x2_2x1(A11, BR_top, CR_top)
    schoolbook_2x2_2x1(A12, BR_bot, CR_top)
    schoolbook_2x2_2x1(A21, BR_top, CR_bot)
    schoolbook_2x2_2x1(A22, BR_bot, CR_bot)
    
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

if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
