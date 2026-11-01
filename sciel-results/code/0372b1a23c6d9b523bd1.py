"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a Strassen-based decomposition of rank 28.
    # 4 blocks of 2x2x2, each rank 7.
    rank = 28
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y01-y11)
    # M4 = x11(y10-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    
    strassen_U = [
        [1, 0, 1, 0],  # x00, x01, x10, x11
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, -1, 0, 1]
    ]
    # Correcting Strassen U vectors based on standard formulation:
    # M1: x00+x11, x01+x11? No.
    # Standard Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # 
    # U vectors (for X):
    # u1 = [1, 0, 1, 1] ? No, a,d are diag. 
    # Let's use indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # u1 = [1, 0, 0, 1] (a+d)
    # u2 = [0, 0, 1, 1] (c+d)
    # u3 = [1, 0, 0, 0] (a)
    # u4 = [0, 0, 0, 1] (d)
    # u5 = [1, 1, 0, 0] (a+b)
    # u6 = [-1, 0, 1, 0] (c-a)
    # u7 = [0, 1, 0, -1] (b-d)
    
    U_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    V_strassen = [
        [1, 0, 1, 0], # e+h
        [1, 0, 0, 0], # e
        [0, 1, -1, 0], # f-h
        [0, 0, 1, -1], # g-e
        [0, 0, 0, 1], # h
        [1, 1, 0, 0], # e+f
        [1, 0, 1, 1] # g+h
    ]
    
    W_strassen = [
        [1, 0, 0, 1], # C00 gets M1, C11 gets M1
        [0, 0, 1, -1], # C10 gets M2, C11 gets -M2
        [0, 1, 0, 1], # C01 gets M3, C11 gets M3
        [1, 0, 1, 0], # C00 gets M4, C10 gets M4
        [-1, 1, 0, 0], # C00 gets -M5, C01 gets M5
        [0, 0, 0, -1], # C11 gets -M6
        [1, 0, 0, 0] # C00 gets M7
    ]
    # Let's re-verify W_strassen mapping.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    #
    # W col 1 (M1): C00 += 1, C11 += 1 -> [1, 0, 0, 1]
    # W col 2 (M2): C10 += 1, C11 += -1 -> [0, 0, 1, -1]
    # W col 3 (M3): C01 += 1, C11 += 1 -> [0, 1, 0, 1]
    # W col 4 (M4): C00 += 1, C10 += 1 -> [1, 0, 1, 0]
    # W col 5 (M5): C00 += -1, C01 += 1 -> [-1, 1, 0, 0]
    # W col 6 (M6): C11 += -1 -> [0, 0, 0, -1]
    # W col 7 (M7): C00 += 1 -> [1, 0, 0, 0]
    #
    # My W_strassen list above:
    # 0: [1, 0, 0, 1] (Correct for M1)
    # 1: [0, 0, 1, -1] (Correct for M2)
    # 2: [0, 1, 0, 1] (Correct for M3)
    # 3: [1, 0, 1, 0] (Correct for M4)
    # 4: [-1, 1, 0, 0] (Correct for M5)
    # 5: [0, 0, 0, -1] (Correct for M6)
    # 6: [1, 0, 0, 0] (Correct for M7)
    
    # Blocks:
    # 1: A1, B11 -> C11 (C indices 0,1 of C block 1? No, C1 is cols 0,1 of C)
    #    A1 is A[:, 0:2]
    #    B11 is B[0:2, 0:2]
    #    Result is C[:, 0:2] part 1
    # 2: A2, B21 -> C11
    #    A2 is A[:, 2:4]
    #    B21 is B[2:4, 0:2]
    #    Result is C[:, 0:2] part 2
    # 3: A1, B12 -> C22
    #    A1 is A[:, 0:2]
    #    B12 is B[0:2, 2:4]
    #    Result is C[:, 2:4] part 1
    # 4: A2, B22 -> C22
    #    A2 is A[:, 2:4]
    #    B22 is B[2:4, 2:4]
    #    Result is C[:, 2:4] part 2
    
    # Mapping indices
    # U row: i*m + j. i in 0..1, j in 0..3.
    # V row: j*p + k. j in 0..3, k in 0..3.
    # W row: k*n + i. k in 0..3, i in 0..1.
    
    def add_strassen_block(base_col, a_start_j, b_start_j, b_start_k, c_start_k):
        # a_start_j: column offset in A (0 or 2)
        # b_start_j: row offset in B (0 or 2)
        # b_start_k: col offset in B (0 or 2)
        # c_start_k: col offset in C (0 or 2)
        
        for s in range(7):
            col = base_col + s
            # U part
            for ui in range(4):
                u_coeff = U_strassen[s][ui]
                if u_coeff != 0:
                    # Map ui (0:00, 1:01, 2:10, 3:11) to i,j
                    # ui 0 -> i=0, j=0
                    # ui 1 -> i=0, j=1
                    # ui 2 -> i=1, j=0
                    # ui 3 -> i=1, j=1
                    i_ui, j_ui = divmod(ui, 2) # 0:0,0 1:0,1 2:1,0 3:1,1
                    u_row = i_ui * m + (a_start_j + j_ui)
                    U[u_row][col] += u_coeff
            
            # V part
            for vi in range(4):
                v_coeff = V_strassen[s][vi]
                if v_coeff != 0:
                    # vi 0 -> j=0, k=0
                    # vi 1 -> j=0, k=1
                    # vi 2 -> j=1, k=0
                    # vi 3 -> j=1, k=1
                    j_vi, k_vi = divmod(vi, 2)
                    v_row = (b_start_j + j_vi) * p + (b_start_k + k_vi)
                    V[v_row][col] += v_coeff
            
            # W part
            for wi in range(4):
                w_coeff = W_strassen[s][wi]
                if w_coeff != 0:
                    # wi 0 -> i=0, k=0
                    # wi 1 -> i=0, k=1
                    # wi 2 -> i=1, k=0
                    # wi 3 -> i=1, k=1
                    i_wi, k_wi = divmod(wi, 2)
                    # W row index: k*n + i
                    w_row = (c_start_k + k_wi) * n + i_wi
                    W[w_row][col] += w_coeff

    # Block 1: A1 (j=0), B11 (j=0, k=0) -> C1 (k=0)
    add_strassen_block(0, 0, 0, 0, 0)
    # Block 2: A2 (j=2), B21 (j=2, k=0) -> C1 (k=0)
    add_strassen_block(7, 2, 2, 0, 0)
    # Block 3: A1 (j=0), B12 (j=0, k=2) -> C2 (k=2)
    add_strassen_block(14, 0, 0, 2, 2)
    # Block 4: A2 (j=2), B22 (j=2, k=2) -> C2 (k=2)
    add_strassen_block(21, 2, 2, 2, 2)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
