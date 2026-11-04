"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (10, rank)
    
    # Decomposition strategy:
    # A (2x4) = [A_left (2x2), A_right (2x2)]
    # B (4x5) = [B_top (2x5); B_bottom (2x5)]
    # C = A_left @ B_top + A_right @ B_bottom
    
    # A_left indices in U: rows 0,1 (cols 0,1 of A) -> U indices 0,1,4,5
    # A_right indices in U: rows 0,1 (cols 2,3 of A) -> U indices 2,3,6,7
    # B_top indices in V: rows 0,1 of B -> V indices 0..4, 5..9
    # B_bottom indices in V: rows 2,3 of B -> V indices 10..14, 15..19
    # C indices in W: k*n + i. 
    # C is 2x5. i in 0..1, k in 0..4.
    # W indices: 0..9.

    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    # a_idx: [a00, a01, a10, a11] indices in U
    # b_idx: [b00, b01, b10, b11] indices in V
    # c_idx: [c00, c01, c10, c11] indices in W
    def add_strassen(a_idx, b_idx, c_idx, start_col):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        U[a_idx[0]][start_col] += 1
        U[a_idx[3]][start_col] += 1
        V[b_idx[0]][start_col] += 1
        V[b_idx[3]][start_col] += 1
        W[c_idx[0]][start_col] += 1
        W[c_idx[1]][start_col] += 0 # M1 contributes to C00, C11
        W[c_idx[2]][start_col] += 0
        W[c_idx[3]][start_col] += 1
        col += 1
        
        # M2 = (A10 + A11) B00
        U[a_idx[2]][start_col] += 1
        U[a_idx[3]][start_col] += 1
        V[b_idx[0]][start_col] += 1
        W[c_idx[1]][start_col] += 0
        W[c_idx[2]][start_col] += 1
        W[c_idx[3]][start_col] += -1
        col += 1
        
        # M3 = A00 (B10 - B11)
        U[a_idx[0]][start_col] += 1
        V[b_idx[2]][start_col] += 1
        V[b_idx[3]][start_col] += -1
        W[c_idx[1]][start_col] += 1
        W[c_idx[3]][start_col] += 1
        col += 1
        
        # M4 = A11 (B01 - B00)
        U[a_idx[3]][start_col] += 1
        V[b_idx[1]][start_col] += 1
        V[b_idx[0]][start_col] += -1
        W[c_idx[0]][start_col] += 1
        W[c_idx[2]][start_col] += 1
        col += 1
        
        # M5 = (A00 + A01) B11
        U[a_idx[0]][start_col] += 1
        U[a_idx[1]][start_col] += 1
        V[b_idx[3]][start_col] += 1
        W[c_idx[0]][start_col] += -1
        W[c_idx[1]][start_col] += 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[a_idx[2]][start_col] += 1
        U[a_idx[0]][start_col] += -1
        V[b_idx[0]][start_col] += 1
        V[b_idx[1]][start_col] += 1
        W[c_idx[3]][start_col] += 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[a_idx[1]][start_col] += 1
        U[a_idx[3]][start_col] += -1
        V[b_idx[2]][start_col] += 1
        V[b_idx[3]][start_col] += 1
        W[c_idx[0]][start_col] += 1
        col += 1
        
        return start_col + 7

    # Helper to add Schoolbook 2x2x1
    # a_idx: [a00, a01, a10, a11] indices in U
    # b_idx: [b00, b10] indices in V (col 0 of 2x1 B)
    # c_idx: [c00, c10] indices in W
    def add_schoolbook_221(a_idx, b_idx, c_idx, start_col):
        nonlocal col
        # C00 = A00*B00 + A01*B10
        # This is not quite right. B is 2x1. B00 is top, B10 is bottom.
        # C00 = A00*B00 + A01*B10? No.
        # A (2x2) * B (2x1) = C (2x1)
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        # We need 4 multiplications.
        # M1 = A00 * B0
        U[a_idx[0]][start_col] += 1
        V[b_idx[0]][start_col] += 1
        W[c_idx[0]][start_col] += 1
        col += 1
        
        # M2 = A01 * B1
        U[a_idx[1]][start_col] += 1
        V[b_idx[1]][start_col] += 1
        W[c_idx[0]][start_col] += 1
        col += 1
        
        # M3 = A10 * B0
        U[a_idx[2]][start_col] += 1
        V[b_idx[0]][start_col] += 1
        W[c_idx[1]][start_col] += 1
        col += 1
        
        # M4 = A11 * B1
        U[a_idx[3]][start_col] += 1
        V[b_idx[1]][start_col] += 1
        W[c_idx[1]][start_col] += 1
        col += 1
        
        return start_col + 4

    # 1. A_left * B_top_left (2x2 * 2x2)
    # A_left: U[0], U[1], U[4], U[5]
    # B_top_left: V[0], V[1], V[5], V[6] (Cols 0,1 of B_top)
    # C_part: W[0], W[1], W[5], W[6] (Cols 0,1 of C)
    a_left = [0, 1, 4, 5]
    b_tl = [0, 1, 5, 6]
    c_tl = [0, 1, 5, 6]
    col = add_strassen(a_left, b_tl, c_tl, col)
    
    # 2. A_left * B_top_mid (2x2 * 2x2)
    # B_top_mid: Cols 2,3 of B_top -> V[2], V[3], V[7], V[8]
    # C_part: Cols 2,3 of C -> W[2], W[3], W[7], W[8]
    b_tm = [2, 3, 7, 8]
    c_tm = [2, 3, 7, 8]
    col = add_strassen(a_left, b_tm, c_tm, col)
    
    # 3. A_left * B_top_right (2x2 * 2x1)
    # B_top_right: Col 4 of B_top -> V[4], V[9]
    # C_part: Col 4 of C -> W[4], W[9]
    b_tr = [4, 9]
    c_tr = [4, 9]
    col = add_schoolbook_221(a_left, b_tr, c_tr, col)
    
    # 4. A_right * B_bottom_left (2x2 * 2x2)
    # A_right: U[2], U[3], U[6], U[7]
    # B_bottom_left: Cols 0,1 of B_bottom -> V[10], V[11], V[15], V[16]
    # C_part: Cols 0,1 of C -> W[0], W[1], W[5], W[6]
    a_right = [2, 3, 6, 7]
    b_bl = [10, 11, 15, 16]
    c_bl = [0, 1, 5, 6]
    col = add_strassen(a_right, b_bl, c_bl, col)
    
    # 5. A_right * B_bottom_mid (2x2 * 2x2)
    # B_bottom_mid: Cols 2,3 of B_bottom -> V[12], V[13], V[17], V[18]
    # C_part: Cols 2,3 of C -> W[2], W[3], W[7], W[8]
    b_bm = [12, 13, 17, 18]
    c_bm = [2, 3, 7, 8]
    col = add_strassen(a_right, b_bm, c_bm, col)
    
    # 6. A_right * B_bottom_right (2x2 * 2x1)
    # B_bottom_right: Col 4 of B_bottom -> V[14], V[19]
    # C_part: Col 4 of C -> W[4], W[9]
    b_br = [14, 19]
    c_br = [4, 9]
    col = add_schoolbook_221(a_right, b_br, c_br, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
