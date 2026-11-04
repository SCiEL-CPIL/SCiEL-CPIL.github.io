"""Optimized matrix multiplication decomposition for 2x5 * 5x6 using block Strassen."""

def construct():
    n, m, p = 2, 5, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_tensor(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def add_strassen_2x2x2(a_rows, b_rows, c_rows, c_cols_offset, scale=1.0):
        # a_rows: list of 2 indices for A cols (inner dim)
        # b_rows: list of 2 indices for B rows (inner dim)
        # c_rows: list of 2 indices for C rows (outer dim)
        # c_cols_offset: offset for C cols
        # This computes C[c_rows, c_cols_offset:c_cols_offset+2] += A[:, a_rows] * B[b_rows, :]
        
        # Strassen 7 multiplications
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B10 - B11)
        # M4 = A11 * (B00 - B01)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # Note: A is 2x2, B is 2x2.
        # A indices: (0,0), (0,1), (1,0), (1,1) relative to sub-block
        # B indices: (0,0), (0,1), (1,0), (1,1) relative to sub-block
        
        # We need to map local indices to global U, V, W indices.
        
        def get_u(i, j):
            # i is local row (0,1), j is local col (0,1)
            # Global A index: row i, col a_rows[j]
            return i * m + a_rows[j]
            
        def get_v(i, j):
            # i is local row (0,1), j is local col (0,1)
            # Global B index: row b_rows[i], col j
            return b_rows[i] * p + j
            
        def get_w(i, j):
            # i is local row (0,1), j is local col (0,1)
            # Global C index: row c_rows[i], col c_cols_offset + j
            return (c_cols_offset + j) * n + c_rows[i]

        # Strassen terms
        # M1
        u1 = [0]*(n*m); v1 = [0]*(m*p); w1 = [0]*(p*n)
        # (A00+A11)(B00+B11)
        u1[get_u(0,0)] += 1; u1[get_u(1,1)] += 1
        v1[get_v(0,0)] += 1; v1[get_v(1,1)] += 1
        w1[get_w(0,0)] += 1; w1[get_w(0,1)] += 1; w1[get_w(1,0)] += 1; w1[get_w(1,1)] += 1
        add_tensor(u1, v1, w1)
        
        # M2
        u2 = [0]*(n*m); v2 = [0]*(m*p); w2 = [0]*(p*n)
        # (A10+A11)B00
        u2[get_u(1,0)] += 1; u2[get_u(1,1)] += 1
        v2[get_v(0,0)] += 1
        w2[get_w(1,0)] += 1
        add_tensor(u2, v2, w2)
        
        # M3
        u3 = [0]*(n*m); v3 = [0]*(m*p); w3 = [0]*(p*n)
        # A00(B10-B11)
        u3[get_u(0,0)] += 1
        v3[get_v(1,0)] += 1; v3[get_v(1,1)] -= 1
        w3[get_w(0,1)] += 1
        add_tensor(u3, v3, w3)
        
        # M4
        u4 = [0]*(n*m); v4 = [0]*(m*p); w4 = [0]*(p*n)
        # A11(B00-B01)
        u4[get_u(1,1)] += 1
        v4[get_v(0,0)] += 1; v4[get_v(0,1)] -= 1
        w4[get_w(1,1)] += 1
        add_tensor(u4, v4, w4)
        
        # M5
        u5 = [0]*(n*m); v5 = [0]*(m*p); w5 = [0]*(p*n)
        # (A00+A01)B11
        u5[get_u(0,0)] += 1; u5[get_u(0,1)] += 1
        v5[get_v(1,1)] += 1
        w5[get_w(0,1)] += 1
        add_tensor(u5, v5, w5)
        
        # M6
        u6 = [0]*(n*m); v6 = [0]*(m*p); w6 = [0]*(p*n)
        # (A10-A00)(B00+B01)
        u6[get_u(1,0)] += 1; u6[get_u(0,0)] -= 1
        v6[get_v(0,0)] += 1; v6[get_v(0,1)] += 1
        w6[get_w(1,0)] += 1; w6[get_w(1,1)] += 1
        add_tensor(u6, v6, w6)
        
        # M7
        u7 = [0]*(n*m); v7 = [0]*(m*p); w7 = [0]*(p*n)
        # (A01-A11)(B10+B11)
        u7[get_u(0,1)] += 1; u7[get_u(1,1)] -= 1
        v7[get_v(1,0)] += 1; v7[get_v(1,1)] += 1
        w7[get_w(0,0)] += 1; w7[get_w(0,1)] += 1
        add_tensor(u7, v7, w7)

    def add_schoolbook_2x3x2(a_rows, b_rows, c_rows, c_cols_offset):
        # a_rows: 3 indices
        # b_rows: 3 indices
        # c_rows: 2 indices
        # c_cols_offset: 2 cols
        
        def get_u(i, j):
            return i * m + a_rows[j]
            
        def get_v(i, j):
            return b_rows[i] * p + j
            
        def get_w(i, j):
            return (c_cols_offset + j) * n + c_rows[i]

        for i in range(2):
            for j in range(3):
                for k in range(2):
                    u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
                    u[get_u(i, j)] = 1
                    v[get_v(j, k)] = 1
                    w[get_w(i, k)] = 1
                    add_tensor(u, v, w)

    # Block 1: C cols 0,1
    # A1: cols 0,1. B1: rows 0,1.
    add_strassen_2x2x2([0, 1], [0, 1], [0, 1], 0)
    
    # A2: cols 2,3,4. B2: rows 2,3,4.
    add_schoolbook_2x3x2([2, 3, 4], [2, 3, 4], [0, 1], 0)

    # Block 2: C cols 2,3,4,5
    # A1: cols 0,1. B3: rows 0,1. Cols 2,3 and 4,5.
    add_strassen_2x2x2([0, 1], [0, 1], [0, 1], 2)
    add_strassen_2x2x2([0, 1], [0, 1], [0, 1], 4)
    
    # A2: cols 2,3,4. B4: rows 2,3,4. Cols 2,3 and 4,5.
    add_schoolbook_2x3x2([2, 3, 4], [2, 3, 4], [0, 1], 2)
    add_schoolbook_2x3x2([2, 3, 4], [2, 3, 4], [0, 1], 4)

    rank = len(U_cols)
    U = [col for col in U_cols]
    V = [col for col in V_cols]
    W = [col for col in W_cols]
    
    return {"U": U, "V": V, "W": W}
