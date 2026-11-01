"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    # Helper to add a term
    # term_id: current column index
    # u_idx, coeff_u: list of (index, coeff)
    # v_idx, coeff_v: list of (index, coeff)
    # w_idx, coeff_w: list of (index, coeff)
    
    def add_term(col, u_terms, v_terms, w_terms):
        for idx, coeff in u_terms:
            U[idx][col] += coeff
        for idx, coeff in v_terms:
            V[idx][col] += coeff
        for idx, coeff in w_terms:
            W[idx][col] += coeff

    # Indices for U (A matrix)
    # A is 2x4
    # Row 0: 0, 1, 2, 3
    # Row 1: 4, 5, 6, 7
    # A_L (cols 0,1): 0, 1, 4, 5
    # A_R (cols 2,3): 2, 3, 6, 7
    
    # Indices for V (B matrix)
    # B is 4x5
    # Row 0: 0-4
    # Row 1: 5-9
    # Row 2: 10-14
    # Row 3: 15-19
    
    # Indices for W (C matrix)
    # C is 2x5 (output)
    # k=0..4, i=0..1
    # W[k*n + i]
    # k=0: 0, 1
    # k=1: 2, 3
    # k=2: 4, 5
    # k=3: 6, 7
    # k=4: 8, 9
    
    col = 0
    
    # --- Block 1: C_L = A_L * B_T ---
    # A_L indices: p11=0, p12=1, p21=4, p22=5
    # B_T indices:
    # B_T[0][0]=0, B_T[0][1]=1, B_T[0][2]=2, B_T[0][3]=3, B_T[0][4]=4
    # B_T[1][0]=5, B_T[1][1]=6, B_T[1][2]=7, B_T[1][3]=8, B_T[1][4]=9
    
    # Strassen B_T[0:2, 0:2] (cols 0,1)
    # M1
    add_term(col, [(0,1), (5,1)], [(0,1), (6,1)], [(0,1), (5,-1), (6,1), (7,-1)]); col += 1
    # M2
    add_term(col, [(4,1), (5,1)], [(0,1)], [(5,1), (6,-1)]); col += 1
    # M3
    add_term(col, [(0,1)], [(1,1), (6,-1)], [(1,1), (5,1)]); col += 1
    # M4
    add_term(col, [(5,1)], [(5,1), (0,-1)], [(5,-1), (6,1)]); col += 1
    # M5
    add_term(col, [(0,1), (1,1)], [(6,1)], [(0,-1), (1,-1), (5,1)]); col += 1
    # M6
    add_term(col, [(4,1), (0,-1)], [(0,1), (1,1)], [(6,1)]); col += 1
    # M7
    add_term(col, [(1,1), (5,-1)], [(5,1), (6,1)], [(0,-1), (5,1), (6,-1)]); col += 1
    
    # Strassen B_T[0:2, 2:4] (cols 2,3)
    # N1
    add_term(col, [(0,1), (5,1)], [(2,1), (8,1)], [(2,1), (7,-1), (8,1), (9,-1)]); col += 1
    # N2
    add_term(col, [(4,1), (5,1)], [(2,1)], [(7,1), (8,-1)]); col += 1
    # N3
    add_term(col, [(0,1)], [(3,1), (8,-1)], [(3,1), (7,1)]); col += 1
    # N4
    add_term(col, [(5,1)], [(7,1), (2,-1)], [(7,-1), (8,1)]); col += 1
    # N5
    add_term(col, [(0,1), (1,1)], [(8,1)], [(2,-1), (3,-1), (7,1)]); col += 1
    # N6
    add_term(col, [(4,1), (0,-1)], [(2,1), (3,1)], [(8,1)]); col += 1
    # N7
    add_term(col, [(1,1), (5,-1)], [(7,1), (8,1)], [(2,-1), (7,1), (8,-1)]); col += 1
    
    # Schoolbook B_T[0:2, 4:5] (col 4)
    # K1
    add_term(col, [(0,1)], [(4,1)], [(4,1)]); col += 1
    # K2
    add_term(col, [(1,1)], [(9,1)], [(4,1)]); col += 1
    # K3
    add_term(col, [(4,1)], [(4,1)], [(9,1)]); col += 1
    # K4
    add_term(col, [(5,1)], [(9,1)], [(9,1)]); col += 1

    # --- Block 2: C_R = A_R * B_B ---
    # A_R indices: p11=2, p12=3, p21=6, p22=7
    # B_B indices:
    # B_B[0][0]=10, B_B[0][1]=11, B_B[0][2]=12, B_B[0][3]=13, B_B[0][4]=14
    # B_B[1][0]=15, B_B[1][1]=16, B_B[1][2]=17, B_B[1][3]=18, B_B[1][4]=19
    
    # Strassen B_B[0:2, 0:2] (cols 0,1)
    # P1
    add_term(col, [(2,1), (7,1)], [(10,1), (16,1)], [(0,1), (5,-1), (6,1), (7,-1)]); col += 1
    # P2
    add_term(col, [(6,1), (7,1)], [(10,1)], [(5,1), (6,-1)]); col += 1
    # P3
    add_term(col, [(2,1)], [(11,1), (16,-1)], [(1,1), (5,1)]); col += 1
    # P4
    add_term(col, [(7,1)], [(15,1), (10,-1)], [(5,-1), (6,1)]); col += 1
    # P5
    add_term(col, [(2,1), (3,1)], [(16,1)], [(0,-1), (1,-1), (5,1)]); col += 1
    # P6
    add_term(col, [(6,1), (2,-1)], [(10,1), (11,1)], [(6,1)]); col += 1
    # P7
    add_term(col, [(3,1), (7,-1)], [(15,1), (16,1)], [(0,-1), (5,1), (6,-1)]); col += 1
    
    # Strassen B_B[0:2, 2:4] (cols 2,3)
    # Q1
    add_term(col, [(2,1), (7,1)], [(12,1), (18,1)], [(2,1), (7,-1), (8,1), (9,-1)]); col += 1
    # Q2
    add_term(col, [(6,1), (7,1)], [(12,1)], [(7,1), (8,-1)]); col += 1
    # Q3
    add_term(col, [(2,1)], [(13,1), (18,-1)], [(3,1), (7,1)]); col += 1
    # Q4
    add_term(col, [(7,1)], [(17,1), (12,-1)], [(7,-1), (8,1)]); col += 1
    # Q5
    add_term(col, [(2,1), (3,1)], [(18,1)], [(2,-1), (3,-1), (7,1)]); col += 1
    # Q6
    add_term(col, [(6,1), (2,-1)], [(12,1), (13,1)], [(8,1)]); col += 1
    # Q7
    add_term(col, [(3,1), (7,-1)], [(17,1), (18,1)], [(2,-1), (7,1), (8,-1)]); col += 1
    
    # Schoolbook B_B[0:2, 4:5] (col 4)
    # R1
    add_term(col, [(2,1)], [(14,1)], [(4,1)]); col += 1
    # R2
    add_term(col, [(3,1)], [(19,1)], [(4,1)]); col += 1
    # R3
    add_term(col, [(6,1)], [(14,1)], [(9,1)]); col += 1
    # R4
    add_term(col, [(7,1)], [(19,1)], [(9,1)]); col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
