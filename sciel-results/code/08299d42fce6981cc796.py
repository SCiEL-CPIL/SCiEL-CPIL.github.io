"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert n == m == p == 5
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_schoolbook_block(a_row_start, a_col_start, a_rows, a_cols,
                             b_row_start, b_col_start, b_rows, b_cols,
                             c_row_start, c_col_start, c_rows, c_cols):
        """Add schoolbook terms for C_block = A_block * B_block.
        A_block is a_rows x a_cols starting at (a_row_start, a_col_start)
        B_block is b_rows x b_cols starting at (b_row_start, b_col_start)
        C_block is c_rows x c_cols starting at (c_row_start, c_col_start)
        """
        for i in range(c_rows):
            for j in range(a_cols):
                for k in range(c_cols):
                    u = [0.0] * nm
                    v = [0.0] * mp
                    w = [0.0] * pn
                    # A element at (a_row_start+i, a_col_start+j)
                    u_idx = (a_row_start + i) * m + (a_col_start + j)
                    u[u_idx] = 1.0
                    # B element at (b_row_start+j, b_col_start+k)
                    v_idx = (b_row_start + j) * p + (b_col_start + k)
                    v[v_idx] = 1.0
                    # C element at (c_row_start+i, c_col_start+k)
                    # W index: k*n + i
                    w_idx = (c_col_start + k) * n + (c_row_start + i)
                    w[w_idx] = 1.0
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)
    
    # Block decomposition: split 5 into 2+3
    # C11(2x2) = A11(2x2)*B11(2x2) + A12(2x3)*B21(3x2)
    add_schoolbook_block(0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2)
    add_schoolbook_block(0, 2, 2, 3, 2, 0, 3, 2, 0, 0, 2, 2)
    
    # C12(2x3) = A11(2x2)*B12(2x3) + A12(2x3)*B22(3x3)
    add_schoolbook_block(0, 0, 2, 2, 0, 2, 2, 3, 0, 2, 2, 3)
    add_schoolbook_block(0, 2, 2, 3, 2, 2, 3, 3, 0, 2, 2, 3)
    
    # C21(3x2) = A21(3x2)*B11(2x2) + A22(3x3)*B21(3x2)
    add_schoolbook_block(2, 0, 3, 2, 0, 0, 2, 2, 2, 0, 3, 2)
    add_schoolbook_block(2, 2, 3, 3, 2, 0, 3, 2, 2, 0, 3, 2)
    
    # C22(3x3) = A21(3x2)*B12(2x3) + A22(3x3)*B22(3x3)
    add_schoolbook_block(2, 0, 3, 2, 0, 2, 2, 3, 2, 2, 3, 3)
    add_schoolbook_block(2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3)
    
    rank = len(U_cols)
    
    U = [[0.0] * rank for _ in range(nm)]
    V = [[0.0] * rank for _ in range(mp)]
    W = [[0.0] * rank for _ in range(pn)]
    
    for c in range(rank):
        u = U_cols[c]
        v = V_cols[c]
        w = W_cols[c]
        for i in range(nm):
            if u[i] != 0:
                U[i][c] = u[i]
        for i in range(mp):
            if v[i] != 0:
                V[i][c] = v[i]
        for i in range(pn):
            if w[i] != 0:
                W[i][c] = w[i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
