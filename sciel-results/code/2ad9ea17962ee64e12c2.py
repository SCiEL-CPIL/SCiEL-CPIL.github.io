def _strassen_square(n, get_decomp):
    if n == 1:
        return np.ones((1,1)), np.ones((1,1)), np.ones((1,1))
    
    nh = n // 2
    # Get sub-decompositions for the 7 products
    # M1 = (A00+A11)(B00+B11)
    # We need the U, V, W for the multiplication of size nh x nh x nh.
    # Let U_s, V_s, W_s be the decomposition for nh x nh x nh.
    # Rank r = U_s.shape[1]
    
    U_s, V_s, W_s = get_decomp(nh)
    r = U_s.shape[1]
    
    U = np.zeros((n*n, 7*r))
    V = np.zeros((n*n, 7*r))
    W = np.zeros((n*n, 7*r))
    
    def add_U(U, sub_U, i_block, j_block, col_off):
        for i in range(nh):
            for j in range(nh):
                row = (i_block*nh + i) * n + (j_block*nh + j)
                sub_row = i * nh + j
                U[row, col_off:col_off+r] += sub_U[sub_row, :]
                
    def add_V(V, sub_V, j_block, k_block, col_off):
        for j in range(nh):
            for k in range(nh):
                row = (j_block*nh + j) * n + (k_block*nh + k)
                sub_row = j * nh + k
                V[row, col_off:col_off+r] += sub_V[sub_row, :]
                
    def add_W(W, sub_W, k_block, i_block, col_off):
        for k in range(nh):
            for i in range(nh):
                row = (k_block*nh + k) * n + (i_block*nh + i)
                sub_row = k * nh + i
                W[row, col_off:col_off+r] += sub_W[sub_row, :]
                
    # M1
    add_U(U, U_s, 0, 0, 0)
    add_U(U, U_s, 1, 1, 0)
    add_V(V, V_s, 0, 0, 0)
    add_V(V, V_s, 1, 1, 0)
    add_W(W, W_s, 0, 0, 0)
    add_W(W, W_s, 0, 1, 0)
    add_W(W, W_s, 1, 0, 0)
    add_W(W, W_s, 1, 1, 0)
    
    # M2
    add_U(U, U_s, 1, 0, r)
    # Subtract A00
    for i in range(nh):
        for j in range(nh):
            row_a00 = i * n + j
            sub_row = i * nh + j
            U[row_a00, r:r+r] -= U_s[sub_row, :]
    add_V(V, V_s, 0, 0, r)
    add_V(V, V_s, 0, 1, r)
    add_W(W, W_s, 1, 1, r)
    add_W(W, W_s, 1, 0, r) # Wait, M2 contributes to C10, C11.
    # C10 is k=1, i=1. C11 is k=1, i=1? No.
    # C10: k in 1..2, i in 1..2?
    # Indices:
    # C00: k=0, i=0
    # C01: k=0, i=1? No.
    # W index is k*n + i.
    # M2 = (A10-A00)(B00+B01).
    # A10 is i=1, j=0. A00 is i=0, j=0.
    # B00 is j=0, k=0. B01 is j=0, k=1.
    # Product is C10 (i=1, k=0) and C11 (i=1, k=1).
    # So W rows for M2 should be k=1, i=1?
    # W index k*n + i.
    # For n=2:
    # C10: k=1, i=1? No.
    # C[i,k].
    # C10: i=1, k=0. Index 0*2 + 1 = 1.
    # C11: i=1, k=1. Index 1*2 + 1 = 3.
    # So W rows 1 and 3.
    add_W(W, W_s, 0, 1, r) # k=0, i=1 -> C10
    add_W(W, W_s, 1, 1, r) # k=1, i=1 -> C11
    
    # M3 = A00(B10-B11)
    # A00: i=0, j=0.
    # B10: j=1, k=0. B11: j=1, k=1.
    # Product: C00 (i=0, k=0) and C01 (i=0, k=1).
    # W rows: 0*2+0=0, 1*2+0=2.
    add_U(U, U_s, 0, 0, 2*r)
    add_V(V, V_s, 1, 0, 2*r)
    # Subtract B11
    for j in range(nh):
        for k in range(nh):
            row_b11 = (1*nh + j) * n + (1*nh + k)
            sub_row = j * nh + k
            V[row_b11, 2*r:2*r+r] -= V_s[sub_row, :]
    add_W(W, W_s, 0, 0, 2*r) # C00
    add_W(W, W_s, 1, 0, 2*r) # C01
    
    # M4 = A11(B01-B00)
    # A11: i=1, j=1.
    # B01: j=0, k=1. B00: j=0, k=0.
    # Product: C11 (i=1, k=1) and C10 (i=1, k=0).
    # W rows: 1*2+1=3, 0*2+1=1.
    add_U(U, U_s, 1, 1, 3*r)
    add_V(V, V_s, 0, 1, 3*r)
    # Subtract B00
    for j in range(nh):
        for k in range(nh):
            row_b00 = (0*nh + j) * n + (0*nh + k)
            sub_row = j * nh + k
            V[row_b00, 3*r:3*r+r] -= V_s[sub_row, :]
    add_W(W, W_s, 1, 1, 3*r) # C11
    add_W(W, W_s, 0, 1, 3*r) # C10
    
    # M5 = (A00+A10)B11
    # A00: i=0, j=0. A10: i=1, j=0.
    # B11: j=1, k=1.
    # Product: C01 (i=0, k=1) and C11 (i=1, k=1).
    # W rows: 1*2+0=2, 1*2+1=3.
    add_U(U, U_s, 0, 0, 4*r)
    add_U(U, U_s, 1, 0, 4*r)
    add_V(V, V_s, 1, 1, 4*r)
    add_W(W, W_s, 1, 0, 4*r) # C01
    add_W(W, W_s, 1, 1, 4*r) # C11
    
    # M6 = (A10-A11)(B01+B11)
    # A10: i=1, j=0. A11: i=1, j=1.
    # B01: j=0, k=1. B11: j=1, k=1.
    # Product: C11 (i=1, k=1) and C10?
    # A10*B01 -> i=1, k=1 (C11).
    # A10*B11 -> i=1, k=1 (C11).
    # A11*B01 -> i=1, k=1 (C11).
    # A11*B11 -> i=1, k=1 (C11).
    # Wait, Strassen M6 is (A10-A11)(B01+B11).
    # This contributes to C11 only?
    # Let's check standard Strassen.
    # C11 = M1 + M4 + M5 + M6.
    # Yes.
    # So W rows: 1*2+1=3.
    add_U(U, U_s, 1, 0, 5*r)
    # Subtract A11
    for i in range(nh):
        for j in range(nh):
            row_a11 = (1*nh + i) * n + (1*nh + j)
            sub_row = i * nh + j
            U[row_a11, 5*r:5*r+r] -= U_s[sub_row, :]
    add_V(V, V_s, 0, 1, 5*r)
    add_V(V, V_s, 1, 1, 5*r)
    add_W(W, W_s, 1, 1, 5*r) # C11
    
    # M7 = (A01-A10)(B00+B10)
    # A01: i=0, j=1. A10: i=1, j=0.
    # B00: j=0, k=0. B10: j=1, k=0.
    # Product: C00 (i=0, k=0) and C01?
    # A01*B00 -> i=0, k=0 (C00).
    # A01*B10 -> i=0, k=0 (C00).
    # A10*B00 -> i=1, k=0 (C10).
    # A10*B10 -> i=1, k=0 (C10).
    # C00 = M1 + M3 + M7.
    # C10 = M2 + M4 + M5 + M6? No.
    # C10 = M2 + M4 + M5 + M6?
    # Let's check C10 formula.
    # C10 = M2 + M4 + M5 + M6?
    # Standard:
    # C00 = M1 + M3 + M7
    # C01 = M1 + M5 + M7
    # C10 = M2 + M4 + M6
    # C11 = M1 + M4 + M5 + M6
    
    # So M7 contributes to C00 and C01.
    # W rows: 0*2+0=0 (C00), 1*2+0=2 (C01).
    add_U(U, U_s, 0, 1, 6*r)
    # Subtract A10
    for i in range(nh):
        for j in range(nh):
            row_a10 = (1*nh + i) * n + (0*nh + j)
            sub_row = i * nh + j
            U[row_a10, 6*r:6*r+r] -= U_s[sub_row, :]
    add_V(V, V_s, 0, 0, 6*r)
    add_V(V, V_s, 1, 0, 6*r)
    add_W(W, W_s, 0, 0, 6*r) # C00
    add_W(W, W_s, 1, 0, 6*r) # C01
    
    return U, V, W
