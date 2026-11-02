import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 66 decomposition using block Winograd/Strassen approach.
    # A (3x4) split into A_top (2x4) and A_bot (1x4)
    # B (4x6) split into 3 blocks of 4x2: B1, B2, B3
    # B blocks split into B_top (2x2) and B_bot (2x2)
    # A_top split into A_top_left (2x2) and A_top_right (2x2)
    # A_bot split into A_bot_left (1x2) and A_bot_right (1x2)
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term for a 2x2x2 block
    # u_idx, v_idx, w_idx are lists of indices for the 4 elements in each block
    # The 7 terms for 2x2x2 (Strassen)
    # 1. a00 * (b11 - b01) -> c00, c11
    # 2. (a00 + a01) * b00 -> c00, c01
    # 3. (a10 + a11) * b11 -> c10, c11
    # 4. a11 * (b01 - b00) -> c00, c11
    # 5. (a00 + a10) * (b00 + b11) -> c00, c01, c10, c11
    # 6. (a01 - a11) * (b10 + b11) -> c01, c11
    # 7. (a10 - a00) * (b00 + b01) -> c00, c10
    
    # We need to map these to the global indices.
    # For a 2x2 block of A at (r0, c0), B at (r0, c0), C at (r0, c0)
    # Note: B indices are (row, col) in the B matrix, which corresponds to (j, k) in the product.
    # A indices are (i, j).
    # C indices are (i, k).
    
    def add_strassen_2x2(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        # Local indices:
        # A: 00, 01, 10, 11 -> (a_r0, a_c0), (a_r0, a_c0+1), (a_r0+1, a_c0), (a_r0+1, a_c0+1)
        # B: 00, 01, 10, 11 -> (b_r0, b_c0), (b_r0, b_c0+1), (b_r0+1, b_c0), (b_r0+1, b_c0+1)
        # C: 00, 01, 10, 11 -> (c_r0, c_c0), (c_r0, c_c0+1), (c_r0+1, c_c0), (c_r0+1, c_c0+1)
        
        # Global indices
        a00 = (a_r0) * m + a_c0
        a01 = (a_r0) * m + a_c0 + 1
        a10 = (a_r0 + 1) * m + a_c0
        a11 = (a_r0 + 1) * m + a_c0 + 1
        
        b00 = b_r0 * p + b_c0
        b01 = b_r0 * p + b_c0 + 1
        b10 = (b_r0 + 1) * p + b_c0
        b11 = (b_r0 + 1) * p + b_c0 + 1
        
        c00 = c_r0 * n + c_c0
        c01 = c_r0 * n + c_c0 + 1
        c10 = (c_r0 + 1) * n + c_c0
        c11 = (c_r0 + 1) * n + c_c0 + 1
        
        # Term 1: a00 * (b11 - b01)
        U[a00][col] = 1; V[b11][col] = 1; W[c00][col] = 1; W[c11][col] = 1; col += 1
        U[a00][col] = 1; V[b01][col] = -1; W[c00][col] = 1; W[c11][col] = 1; col += 1
        
        # Term 2: (a00 + a01) * b00
        U[a00][col] = 1; V[b00][col] = 1; W[c00][col] = 1; W[c01][col] = 1; col += 1
        U[a01][col] = 1; V[b00][col] = 1; W[c00][col] = 1; W[c01][col] = 1; col += 1
        
        # Term 3: (a10 + a11) * b11
        U[a10][col] = 1; V[b11][col] = 1; W[c10][col] = 1; W[c11][col] = 1; col += 1
        U[a11][col] = 1; V[b11][col] = 1; W[c10][col] = 1; W[c11][col] = 1; col += 1
        
        # Term 4: a11 * (b01 - b00)
        U[a11][col] = 1; V[b01][col] = 1; W[c00][col] = 1; W[c11][col] = 1; col += 1
        U[a11][col] = 1; V[b00][col] = -1; W[c00][col] = 1; W[c11][col] = 1; col += 1
        
        # Term 5: (a00 + a10) * (b00 + b11)
        U[a00][col] = 1; V[b00][col] = 1; W[c00][col] = 1; W[c01][col] = 1; W[c10][col] = 1; W[c11][col] = 1; col += 1
        U[a10][col] = 1; V[b00][col] = 1; W[c00][col] = 1; W[c01][col] = 1; W[c10][col] = 1; W[c11][col] = 1; col += 1
        U[a00][col] = 1; V[b11][col] = 1; W[c00][col] = 1; W[c01][col] = 1; W[c10][col] = 1; W[c11][col] = 1; col += 1
        U[a10][col] = 1; V[b11][col] = 1; W[c00][col] = 1; W[c01][col] = 1; W[c10][col] = 1; W[c11][col] = 1; col += 1
        
        # Term 6: (a01 - a11) * (b10 + b11)
        U[a01][col] = 1; V[b10][col] = 1; W[c01][col] = 1; W[c11][col] = 1; col += 1
        U[a11][col] = -1; V[b10][col] = 1; W[c01][col] = 1; W[c11][col] = 1; col += 1
        U[a01][col] = 1; V[b11][col] = 1; W[c01][col] = 1; W[c11][col] = 1; col += 1
        U[a11][col] = -1; V[b11][col] = 1; W[c01][col] = 1; W[c11][col] = 1; col += 1
        
        # Term 7: (a10 - a00) * (b00 + b01)
        U[a10][col] = 1; V[b00][col] = 1; W[c00][col] = 1; W[c10][col] = 1; col += 1
        U[a00][col] = -1; V[b00][col] = 1; W[c00][col] = 1; W[c10][col] = 1; col += 1
        U[a10][col] = 1; V[b01][col] = 1; W[c00][col] = 1; W[c10][col] = 1; col += 1
        U[a00][col] = -1; V[b01][col] = 1; W[c00][col] = 1; W[c10][col] = 1; col += 1

    def add_1x2x2(a_r, a_c0, b_r0, b_c0, c_r, c_c0):
        nonlocal col
        # A: 1x2 at (a_r, a_c0)
        # B: 2x2 at (b_r0, b_c0)
        # C: 1x2 at (c_r, c_c0)
        
        a0 = a_r * m + a_c0
        a1 = a_r * m + a_c0 + 1
        
        b00 = b_r0 * p + b_c0
        b01 = b_r0 * p + b_c0 + 1
        b10 = (b_r0 + 1) * p + b_c0
        b11 = (b_r0 + 1) * p + b_c0 + 1
        
        c0 = c_r * n + c_c0
        c1 = c_r * n + c_c0 + 1
        
        # c0 = a0*b00 + a1*b10
        # c1 = a0*b01 + a1*b11
        
        U[a0][col] = 1; V[b00][col] = 1; W[c0][col] = 1; col += 1
        U[a1][col] = 1; V[b10][col] = 1; W[c0][col] = 1; col += 1
        U[a0][col] = 1; V[b01][col] = 1; W[c1][col] = 1; col += 1
        U[a1][col] = 1; V[b11][col] = 1; W[c1][col] = 1; col += 1

    # Top block of C (rows 0,1)
    # k=0 (cols 0,1 of B)
    add_strassen_2x2(0, 0, 0, 0, 0, 0) # A_top_left * B1_top
    add_strassen_2x2(0, 2, 2, 0, 0, 0) # A_top_right * B1_bot
    
    # k=1 (cols 2,3 of B)
    add_strassen_2x2(0, 0, 0, 2, 0, 2) # A_top_left * B2_top
    add_strassen_2x2(0, 2, 2, 2, 0, 2) # A_top_right * B2_bot
    
    # k=2 (cols 4,5 of B)
    add_strassen_2x2(0, 0, 0, 4, 0, 4) # A_top_left * B3_top
    add_strassen_2x2(0, 2, 2, 4, 0, 4) # A_top_right * B3_bot
    
    # Bottom block of C (row 2)
    # k=0
    add_1x2x2(2, 0, 0, 0, 2, 0) # A_bot_left * B1_top
    add_1x2x2(2, 2, 2, 0, 2, 0) # A_bot_right * B1_bot
    
    # k=1
    add_1x2x2(2, 0, 0, 2, 2, 2) # A_bot_left * B2_top
    add_1x2x2(2, 2, 2, 2, 2, 2) # A_bot_right * B2_bot
    
    # k=2
    add_1x2x2(2, 0, 0, 4, 2, 4) # A_bot_left * B3_top
    add_1x2x2(2, 2, 2, 4, 2, 4) # A_bot_right * B3_bot

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
