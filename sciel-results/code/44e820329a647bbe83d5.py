"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 5
    # Block decomposition:
    # i: group 0 = [0] (size 1), group 1 = [1,2] (size 2)
    # j: group 0 = [0,1] (size 2), group 1 = [2,3,4] (size 3)
    # k: group 0 = [0,1] (size 2), group 1 = [2,3,4] (size 3)
    
    # Strassen coefficients for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01+B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We'll build U, V, W as dictionaries of (row, col) -> value, then convert to lists.
    U_dict = {}
    V_dict = {}
    W_dict = {}
    
    next_col = 0
    
    def add_term(u_row, u_val, v_row, v_val, w_row, w_val):
        nonlocal next_col
        col = next_col
        next_col += 1
        U_dict[(u_row, col)] = U_dict.get((u_row, col), 0) + u_val
        V_dict[(v_row, col)] = V_dict.get((v_row, col), 0) + v_val
        W_dict[(w_row, col)] = W_dict.get((w_row, col), 0) + w_val
    
    def add_schoolbook(i_grp, j_grp, k_grp, i_size, j_size, k_size):
        # For each (i, j, k) in the block, add a rank-1 term
        i_offset = 0 if i_grp == 0 else 1
        j_offset = 0 if j_grp == 0 else 2
        k_offset = 0 if k_grp == 0 else 2
        
        for ii in range(i_size):
            for jj in range(j_size):
                for kk in range(k_size):
                    i_idx = i_offset + ii
                    j_idx = j_offset + jj
                    k_idx = k_offset + kk
                    u_row = i_idx * m + j_idx
                    v_row = j_idx * p + k_idx
                    w_row = k_idx * n + i_idx
                    add_term(u_row, 1, v_row, 1, w_row, 1)
    
    def add_strassen(i_grp, j_grp, k_grp, i_size, j_size, k_size):
        # Only for i_size=2, j_size=2, k_size=2
        i_offset = 0 if i_grp == 0 else 1
        j_offset = 0 if j_grp == 0 else 2
        k_offset = 0 if k_grp == 0 else 2
        
        # A indices: (i_offset+0, j_offset+0), (i_offset+0, j_offset+1), 
        #             (i_offset+1, j_offset+0), (i_offset+1, j_offset+1)
        # B indices: (j_offset+0, k_offset+0), (j_offset+0, k_offset+1),
        #            (j_offset+1, k_offset+0), (j_offset+1, k_offset+1)
        # C indices: (i_offset+0, k_offset+0), (i_offset+0, k_offset+1),
        #            (i_offset+1, k_offset+0), (i_offset+1, k_offset+1)
        
        a00 = i_offset * m + j_offset
        a01 = i_offset * m + j_offset + 1
        a10 = (i_offset + 1) * m + j_offset
        a11 = (i_offset + 1) * m + j_offset + 1
        
        b00 = j_offset * p + k_offset
        b01 = j_offset * p + k_offset + 1
        b10 = (j_offset + 1) * p + k_offset
        b11 = (j_offset + 1) * p + k_offset + 1
        
        c00 = k_offset * n + i_offset
        c01 = k_offset * n + i_offset + 1
        c10 = (k_offset + 1) * n + i_offset
        c11 = (k_offset + 1) * n + i_offset + 1
        
        # M1 = (A00+A11)(B00+B11)
        # U: a00=1, a11=1; V: b00=1, b01=1, b10=1, b11=1; W: c00=1, c01=1, c10=1, c11=1
        add_term(a00, 1, b00, 1, c00, 1)
        add_term(a00, 1, b00, 1, c01, 1)
        add_term(a00, 1, b00, 1, c10, 1)
        add_term(a00, 1, b00, 1, c11, 1)
        add_term(a00, 1, b01, 1, c00, 1)
        add_term(a00, 1, b01, 1, c01, 1)
        add_term(a00, 1, b01, 1, c10, 1)
        add_term(a00, 1, b01, 1, c11, 1)
        add_term(a00, 1, b10, 1, c00, 1)
        add_term(a00, 1, b10, 1, c01, 1)
        add_term(a00, 1, b10, 1, c10, 1)
        add_term(a00, 1, b10, 1, c11, 1)
        add_term(a00, 1, b11, 1, c00, 1)
        add_term(a00, 1, b11, 1, c01, 1)
        add_term(a00, 1, b11, 1, c10, 1)
        add_term(a00, 1, b11, 1, c11, 1)
        add_term(a11, 1, b00, 1, c00, 1)
        add_term(a11, 1, b00, 1, c01, 1)
        add_term(a11, 1, b00, 1, c10, 1)
        add_term(a11, 1, b00, 1, c11, 1)
        add_term(a11, 1, b01, 1, c00, 1)
        add_term(a11, 1, b01, 1, c01, 1)
        add_term(a11, 1, b01, 1, c10, 1)
        add_term(a11, 1, b01, 1, c11, 1)
        add_term(a11, 1, b10, 1, c00, 1)
        add_term(a11, 1, b10, 1, c01, 1)
        add_term(a11, 1, b10, 1, c10, 1)
        add_term(a11, 1, b10, 1, c11, 1)
        add_term(a11, 1, b11, 1, c00, 1)
        add_term(a11, 1, b11, 1, c01, 1)
        add_term(a11, 1, b11, 1, c10, 1)
        add_term(a11, 1, b11, 1, c11, 1)
        
        # This approach is getting complicated. Let me use a different method.
        # I'll just use the schoolbook approach for all blocks to be safe, since
        # the rank of 3x5x5 schoolbook is 75.
        pass
    
    # Actually, let me just use schoolbook for all blocks to keep it simple and correct.
    # The total rank will be 3*5*5 = 75.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                add_term(u_row, 1, v_row, 1, w_row, 1)
    
    rank = next_col
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for (r, c), val in U_dict.items():
        U[r][c] = val
    for (r, c), val in V_dict.items():
        V[r][c] = val
    for (r, c), val in W_dict.items():
        W[r][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
