import time

def construct():
    n, m, p = 4, 4, 6
    U_rows = n * m   # 16
    V_rows = m * p   # 24
    W_rows = p * n   # 24
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def zeros(length):
        return [0] * length
    
    def strassen_2x2_2x2(ar, ac, br, bc, cr, cc):
        """Add 7 rank-1 terms for Strassen 2x2*2x2.
        A block at rows ar:ar+2, cols ac:ac+2
        B block at rows br:br+2, cols bc:bc+2
        C block at rows cr:cr+2, cols cc:cc+2
        """
        def a_idx(i, j): return (ar + i) * m + (ac + j)
        def b_idx(i, j): return (br + i) * p + (bc + j)
        def c_idx(i, j): return (cc + j) * n + (cr + i)
        
        # M1 = (A11+A22)*(B11+B22)
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0,0)] = 1; u[a_idx(1,1)] = 1
        v[b_idx(0,0)] = 1; v[b_idx(1,1)] = 1
        w[c_idx(0,0)] = 1; w[c_idx(1,1)] = 1
        add_term(u, v, w)
        
        # M2 = (A21+A22)*B11
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1,0)] = 1; u[a_idx(1,1)] = 1
        v[b_idx(0,0)] = 1
        w[c_idx(1,0)] = 1; w[c_idx(1,1)] = -1
        add_term(u, v, w)
        
        # M3 = A11*(B12-B22)
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0,0)] = 1
        v[b_idx(0,1)] = 1; v[b_idx(1,1)] = -1
        w[c_idx(0,1)] = 1; w[c_idx(1,1)] = 1
        add_term(u, v, w)
        
        # M4 = A22*(B21-B11)
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1,1)] = 1
        v[b_idx(1,0)] = 1; v[b_idx(0,0)] = -1
        w[c_idx(0,0)] = 1; w[c_idx(1,0)] = 1
        add_term(u, v, w)
        
        # M5 = (A11+A12)*B22
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0,0)] = 1; u[a_idx(0,1)] = 1
        v[b_idx(1,1)] = 1
        w[c_idx(0,0)] = -1; w[c_idx(0,1)] = 1
        add_term(u, v, w)
        
        # M6 = (A21-A11)*(B11+B12)
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1,0)] = 1; u[a_idx(0,0)] = -1
        v[b_idx(0,0)] = 1; v[b_idx(0,1)] = 1
        w[c_idx(1,1)] = 1
        add_term(u, v, w)
        
        # M7 = (A12-A22)*(B21+B22)
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0,1)] = 1; u[a_idx(1,1)] = -1
        v[b_idx(1,0)] = 1; v[b_idx(1,1)] = 1
        w[c_idx(0,0)] = 1
        add_term(u, v, w)
    
    # Part 1: A * B1 where B1 is columns 0-3 of B (4×4)
    # C1 = A * B1 (4×4 result, columns 0-3 of C)
    # C11 = A11*B11 + A12*B21
    strassen_2x2_2x2(0,0, 0,0, 0,0)  # A11*B11
    strassen_2x2_2x2(0,2, 2,0, 0,0)  # A12*B21
    # C12 = A11*B12 + A12*B22
    strassen_2x2_2x2(0,0, 0,2, 0,2)  # A11*B12
    strassen_2x2_2x2(0,2, 2,2, 0,2)  # A12*B22
    # C21 = A21*B11 + A22*B21
    strassen_2x2_2x2(2,0, 0,0, 2,0)  # A21*B11
    strassen_2x2_2x2(2,2, 2,0, 2,0)  # A22*B21
    # C22 = A21*B12 + A22*B22
    strassen_2x2_2x2(2,0, 0,2, 2,2)  # A21*B12
    strassen_2x2_2x2(2,2, 2,2, 2,2)  # A22*B22
    
    # Part 2: A * B2 where B2 is columns 4-5 of B (4×2)
    # C2 = A * B2 (4×2 result, columns 4-5 of C)
    # C2_top (2×2) = A11*B2_top + A12*B2_bot
    # B2_top = B[0:2, 4:6], B2_bot = B[2:4, 4:6]
    strassen_2x2_2x2(0,0, 0,4, 0,4)  # A11*B2_top
    strassen_2x2_2x2(0,2, 2,4, 0,4)  # A12*B2_bot
    # C2_bot (2×2) = A21*B2_top + A22*B2_bot
    strassen_2x2_2x2(2,0, 0,4, 2,4)  # A21*B2_top
    strassen_2x2_2x2(2,2, 2,4, 2,4)  # A22*B2_bot
    
    rank = len(U_cols)
    
    U = [[U_cols[r][i] for r in range(rank)] for i in range(U_rows)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(V_rows)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(W_rows)]
    
    return {"U": U, "V": V, "W": W}
