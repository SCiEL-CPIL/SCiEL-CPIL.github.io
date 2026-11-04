"""Matrix multiplication 3x4x6 with block-Strassen decomposition."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    # Block structure: n=2+1, m=2+2, p=2+2+2
    # 2x2x2 blocks use Strassen (rank 7), 1x2x2 blocks use schoolbook (rank 4)
    
    # Block offsets
    i_blocks = [(0, 2), (2, 3)]  # sizes 2, 1
    j_blocks = [(0, 2), (2, 4)]  # sizes 2, 2
    k_blocks = [(0, 2), (2, 4), (4, 6)]  # sizes 2, 2, 2
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_block(i_off, j_off, k_off):
        """Add Strassen decomposition for 2x2x2 block at given offsets."""
        # Strassen: 7 products for 2x2 matrix multiplication
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
        # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
        # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
        
        # U vectors (for A entries), V vectors (for B entries), W vectors (for C entries)
        # Each is a dict mapping (row, col) -> coefficient
        # U: n*m x 7, V: m*p x 7, W: p*n x 7 (within the block)
        
        # For 2x2x2: U is 4x7, V is 4x7, W is 4x7
        # U indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
        # V indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
        # W indices: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22
        
        strassen_U = [
            [1, 0, 1, 0, 1, 0, 0],  # a
            [0, 0, 0, 0, 1, 0, 1],  # b
            [0, 1, 0, 0, 0, -1, 0], # c
            [1, 1, 0, 1, 0, 0, -1], # d
        ]
        strassen_V = [
            [1, 1, 0, -1, 0, 1, 1], # e
            [0, 0, 1, 0, 1, 1, 0],  # f
            [0, 0, 0, 1, 0, 0, 1],  # g
            [1, 0, -1, 0, 1, 0, 1], # h
        ]
        strassen_W = [
            [1, 0, 0, 1, -1, 0, 1],  # C11
            [0, 0, 1, 0, 1, 0, 0],   # C12
            [0, 1, 0, 1, 0, 0, 0],   # C21
            [1, -1, 1, 0, 0, 1, 0],  # C22
        ]
        
        for r in range(7):
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for a_row in range(2):
                for a_col in range(2):
                    idx = a_row * 2 + a_col
                    u_col[(i_off + a_row) * m + (j_off + a_col)] = strassen_U[idx][r]
                    
            for b_row in range(2):
                for b_col in range(2):
                    idx = b_row * 2 + b_col
                    v_col[(j_off + b_row) * p + (k_off + b_col)] = strassen_V[idx][r]
                    
            for c_row in range(2):
                for c_col in range(2):
                    idx = c_row * 2 + c_col
                    # W[k*n+i, r] convention: output C[c_col, c_row] maps to W[(k_off+c_col)*n + (i_off+c_row)]
                    w_col[(k_off + c_col) * n + (i_off + c_row)] = strassen_W[idx][r]
                    
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    def add_schoolbook_1x2x2_block(i_off, j_off, k_off):
        """Add schoolbook decomposition for 1x2x2 block at given offsets."""
        # A is 1x2: [a, b], B is 2x2: [[e,f],[g,h]], C is 1x2: [C1, C2]
        # C1 = a*e + b*g, C2 = a*f + b*h
        # 4 products: ae, af, bg, bh
        
        products = [
            # (U_idx, V_idx, W_idx) where indices are local
            (0, 0, 0),  # a * e -> C1
            (0, 1, 1),  # a * f -> C2
            (1, 2, 0),  # b * g -> C1
            (1, 3, 1),  # b * h -> C2
        ]
        
        for u_idx, v_idx, w_idx in products:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            # u_idx: 0=a (row 0, col 0), 1=b (row 0, col 1)
            a_row = u_idx // 2
            a_col = u_idx % 2
            u_col[(i_off + a_row) * m + (j_off + a_col)] = 1
            
            # v_idx: 0=e(0,0), 1=f(0,1), 2=g(1,0), 3=h(1,1)
            b_row = v_idx // 2
            b_col = v_idx % 2
            v_col[(j_off + b_row) * p + (k_off + b_col)] = 1
            
            # w_idx: 0=C1 (row 0, col 0), 1=C2 (row 0, col 1)
            c_row = w_idx // 2
            c_col = w_idx % 2
            w_col[(k_off + c_col) * n + (i_off + c_row)] = 1
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    # Build all blocks
    for i_idx, (i_start, i_end) in enumerate(i_blocks):
        i_size = i_end - i_start
        for j_idx, (j_start, j_end) in enumerate(j_blocks):
            j_size = j_end - j_start
            for k_idx, (k_start, k_end) in enumerate(k_blocks):
                k_size = k_end - k_start
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    add_strassen_block(i_start, j_start, k_start)
                elif i_size == 1 and j_size == 2 and k_size == 2:
                    add_schoolbook_1x2x2_block(i_start, j_start, k_start)
                else:
                    # Fallback: schoolbook for any other combination
                    for a in range(i_size):
                        for b in range(j_size):
                            for c in range(k_size):
                                u_col = [0] * (n * m)
                                v_col = [0] * (m * p)
                                w_col = [0] * (p * n)
                                u_col[(i_start + a) * m + (j_start + b)] = 1
                                v_col[(j_start + b) * p + (k_start + c)] = 1
                                w_col[(k_start + c) * n + (i_start + a)] = 1
                                U_cols.append(u_col)
                                V_cols.append(v_col)
                                W_cols.append(w_col)
    
    rank = len(U_cols)
    return {"U": U_cols, "V": V_cols, "W": W_cols}
# EVOLVE-BLOCK-END
