"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 36 decomposition using Strassen for 2x2x2 blocks and Schoolbook for 2x2x1
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs, col_idx):
        for idx, coeff in zip(u_indices, u_coeffs):
            U[idx][col_idx] += coeff
        for idx, coeff in zip(v_indices, v_coeffs):
            V[idx][col_idx] += coeff
        for idx, coeff in zip(w_indices, w_coeffs):
            W[idx][col_idx] += coeff
            
    # Strassen coefficients for 2x2x2
    # m1: (a+d)(e+h)
    # m2: (c+d)e
    # m3: a(f-h)
    # m4: d(g-f)
    # m5: (a+b)h
    # m6: (c-d)(e+f)
    # m7: (b-d)(g+h)
    
    # C = [[m1-m4+m5+m7, m3+m4-m6+m7],
    #      [m1-m2+m3+m5, -m2+m4+m5+m6]]
    
    def strassen_block(a_idx, v_e, v_f, v_g, v_h, c_00, c_01, c_10, c_11, start_col):
        # a_idx: list of 4 indices [a,b,c,d]
        # v_e, v_f, v_g, v_h: indices for B elements
        # c_00..c_11: indices for C elements
        
        a, b, c, d = a_idx
        
        # m1
        add_term([a, d], [1, 1], [v_e, v_h], [1, 1], 
                 [c_00, c_01, c_10, c_11], [1, 1, -1, -1], start_col + 0)
        # m2
        add_term([c, d], [1, 1], [v_e], [1], 
                 [c_10, c_11], [-1, 1], start_col + 1)
        # m3
        add_term([a], [1], [v_f, v_h], [1, -1], 
                 [c_01, c_10], [1, 1], start_col + 2)
        # m4
        add_term([d], [1], [v_g, v_f], [1, -1], 
                 [c_00, c_01, c_10, c_11], [-1, 1, 1, 1], start_col + 3)
        # m5
        add_term([a, b], [1, 1], [v_h], [1], 
                 [c_00, c_10, c_11], [1, 1, 1], start_col + 4)
        # m6
        add_term([c, d], [1, -1], [v_e, v_f], [1, 1], 
                 [c_01, c_11], [-1, 1], start_col + 5)
        # m7
        add_term([b, d], [1, -1], [v_g, v_h], [1, 1], 
                 [c_00, c_01, c_10], [1, 1, 1], start_col + 6)
                 
    def schoolbook_2x2x1(a_idx, v_y1, v_y2, c_0, c_1, start_col):
        a, b, c, d = a_idx
        # c0 = a*y1 + b*y2
        add_term([a], [1], [v_y1], [1], [c_0], [1], start_col + 0)
        add_term([b], [1], [v_y2], [1], [c_0], [1], start_col + 1)
        # c1 = c*y1 + d*y2
        add_term([c], [1], [v_y1], [1], [c_1], [1], start_col + 2)
        add_term([d], [1], [v_y2], [1], [c_1], [1], start_col + 3)
        
    # Block 1: A[0:2, 0:2] * B[0:2, 0:5]
    # A indices: 0,1,4,5
    # B indices: 
    #   Col 0,1: 0,1,5,6 (e,f,g,h)
    #   Col 2,3: 2,3,7,8
    #   Col 4: 4,9
    # C indices:
    #   Col 0,1: 0,2,1,3 (00,01,10,11)
    #   Col 2,3: 4,6,5,7
    #   Col 4: 8,9
    
    strassen_block([0,1,4,5], 0, 1, 5, 6, 0, 2, 1, 3, 0)
    strassen_block([0,1,4,5], 2, 3, 7, 8, 4, 6, 5, 7, 7)
    schoolbook_2x2x1([0,1,4,5], 4, 9, 8, 9, 14)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:5]
    # A indices: 2,3,6,7
    # B indices:
    #   Col 0,1: 10,11,15,16
    #   Col 2,3: 12,13,17,18
    #   Col 4: 14,19
    # C indices same as Block 1
    
    strassen_block([2,3,6,7], 10, 11, 15, 16, 0, 2, 1, 3, 18)
    strassen_block([2,3,6,7], 12, 13, 17, 18, 4, 6, 5, 7, 25)
    schoolbook_2x2x1([2,3,6,7], 14, 19, 8, 9, 32)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
