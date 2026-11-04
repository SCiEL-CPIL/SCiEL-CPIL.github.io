import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 36
    
    # U: (n*m) x rank = 8 x 36
    # V: (m*p) x rank = 20 x 36
    # W: (p*n) x rank = 10 x 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    def add_strassen(a_idx, b_idx, c_idx):
        # a_idx: list of 4 indices into U (flat) for a,b,c,d
        # b_idx: list of 4 indices into V (flat) for e,f,g,h
        # c_idx: list of 4 indices into W (flat) for C11, C12, C21, C22
        
        nonlocal col
        
        # Strassen factors
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # We need to map these linear forms to U and V columns.
        # U column entries are coefficients for A elements.
        # V column entries are coefficients for B elements.
        # W column entries are coefficients for C elements.
        
        # M1
        U[a_idx[0]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1
        W[c_idx[1]][col] += 0 # C12? No, M1 contributes to C11 and C22
        # Wait, let's use the explicit formulas:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # It is safer to define the 7 terms and their contributions to C.
        
        terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # u_coeffs: dict {a_idx: coeff}
            # v_coeffs: dict {b_idx: coeff}
            # w_coeffs: dict {c_idx: coeff}
            
            # M1 = (a+d)(e+h)
            ({a_idx[0]: 1, a_idx[3]: 1}, {b_idx[0]: 1, b_idx[3]: 1}, {c_idx[0]: 1, c_idx[3]: 1}),
            
            # M2 = (c+d)e
            ({a_idx[2]: 1, a_idx[3]: 1}, {b_idx[0]: 1}, {c_idx[2]: 1, c_idx[3]: -1}),
            
            # M3 = a(f-h)
            ({a_idx[0]: 1}, {b_idx[1]: 1, b_idx[3]: -1}, {c_idx[1]: 1, c_idx[3]: 1}),
            
            # M4 = d(g-e)
            ({a_idx[3]: 1}, {b_idx[2]: 1, b_idx[0]: -1}, {c_idx[0]: 1, c_idx[2]: 1}),
            
            # M5 = (a+b)h
            ({a_idx[0]: 1, a_idx[1]: 1}, {b_idx[3]: 1}, {c_idx[0]: -1, c_idx[1]: 1}),
            
            # M6 = (c-a)(e+f)
            ({a_idx[2]: 1, a_idx[0]: -1}, {b_idx[0]: 1, b_idx[1]: 1}, {c_idx[3]: 1}),
            
            # M7 = (b-d)(g+h)
            ({a_idx[1]: 1, a_idx[3]: -1}, {b_idx[2]: 1, b_idx[3]: 1}, {c_idx[0]: 1})
        ]
        
        for u_c, v_c, w_c in terms:
            for idx, val in u_c.items():
                U[idx][col] += val
            for idx, val in v_c.items():
                V[idx][col] += val
            for idx, val in w_c.items():
                W[idx][col] += val
            col += 1

    # Helper to add Schoolbook 2x2x1 block
    def add_schoolbook_221(a_idx, b_idx, c_idx):
        # a_idx: 4 indices for a,b,c,d
        # b_idx: 2 indices for x,y
        # c_idx: 2 indices for C1, C2
        
        nonlocal col
        
        # C1 = ax + by
        # C2 = cx + dy
        
        terms = [
            ({a_idx[0]: 1}, {b_idx[0]: 1}, {c_idx[0]: 1}), # ax
            ({a_idx[1]: 1}, {b_idx[1]: 1}, {c_idx[0]: 1}), # by
            ({a_idx[2]: 1}, {b_idx[0]: 1}, {c_idx[1]: 1}), # cx
            ({a_idx[3]: 1}, {b_idx[1]: 1}, {c_idx[1]: 1})  # dy
        ]
        
        for u_c, v_c, w_c in terms:
            for idx, val in u_c.items():
                U[idx][col] += val
            for idx, val in v_c.items():
                V[idx][col] += val
            for idx, val in w_c.items():
                W[idx][col] += val
            col += 1

    # Define indices
    # A flat index: i*4 + j
    # B flat index: j*5 + k
    # C flat index: k*2 + i
    
    # Block 1: A_L (cols 0,1) * B_T (rows 0,1)
    # A_L indices:
    a_L = [0, 1, 4, 5] # a00, a01, a10, a11
    
    # B_T1 (cols 0,1, rows 0,1)
    b_T1 = [0, 1, 5, 6] # b00, b01, b10, b11
    c_out_11 = [0, 2, 1, 3] # C00, C01, C10, C11 -> indices 0, 2, 1, 3
    
    add_strassen(a_L, b_T1, c_out_11)
    
    # B_T2 (cols 2,3, rows 0,1)
    b_T2 = [2, 3, 7, 8] # b02, b03, b12, b13
    c_out_12 = [4, 6, 5, 7] # C20, C21, C30, C31 -> indices 4, 6, 5, 7
    
    add_strassen(a_L, b_T2, c_out_12)
    
    # B_T3 (col 4, rows 0,1)
    b_T3 = [4, 9] # b04, b14
    c_out_13 = [8, 9] # C40, C41 -> indices 8, 9
    
    add_schoolbook_221(a_L, b_T3, c_out_13)
    
    # Block 2: A_R (cols 2,3) * B_B (rows 2,3)
    # A_R indices:
    a_R = [2, 3, 6, 7] # a02, a03, a12, a13
    
    # B_B1 (cols 0,1, rows 2,3)
    b_B1 = [10, 11, 15, 16] # b20, b21, b30, b31
    c_out_21 = [0, 2, 1, 3] # Same output indices as 11
    
    add_strassen(a_R, b_B1, c_out_21)
    
    # B_B2 (cols 2,3, rows 2,3)
    b_B2 = [12, 13, 17, 18] # b22, b23, b32, b33
    c_out_22 = [4, 6, 5, 7] # Same output indices as 12
    
    add_strassen(a_R, b_B2, c_out_22)
    
    # B_B3 (col 4, rows 2,3)
    b_B3 = [14, 19] # b24, b34
    c_out_23 = [8, 9] # Same output indices as 13
    
    add_schoolbook_221(a_R, b_B3, c_out_23)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
