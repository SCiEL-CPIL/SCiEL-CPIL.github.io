"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 36 decomposition for 2x4x5 matrix multiplication.
    # Strategy:
    # 1. Split B (4x5) into B_left (4x2) and B_right (4x3).
    # 2. C_left = A * B_left (2x2). Use two 2x2x2 Strassen decompositions.
    #    Split A (2x4) into A_top (2x2) and A_bot (2x2).
    #    B_left (4x2) into B_top (2x2) and B_bot (2x2).
    #    C_left = A_top * B_top + A_bot * B_bot.
    #    Rank = 7 (Strassen) + 7 (Strassen) = 14.
    # 3. C_right = A * B_right (2x3).
    #    Split B_right (4x3) into B_mid (4x1) and B_far (4x2).
    #    C_mid = A * B_mid (2x1). Schoolbook rank 8.
    #    C_far = A * B_far (2x2). Use two 2x2x2 Strassen decompositions.
    #    Rank = 7 + 7 = 14.
    # Total Rank = 14 + 8 + 14 = 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2x2(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        # Adds a rank-7 Strassen decomposition for 2x2x2 multiplication
        # A is at (a_r0, a_c0), B at (b_r0, b_c0), C at (c_r0, c_c0)
        
        # U indices: A[i][j] -> U[i*m + j]
        # V indices: B[i][j] -> V[i*p + j]
        # W indices: C[i][j] -> W[j*n + i]
        
        # A elements
        a00 = a_r0 * m + a_c0
        a01 = a_r0 * m + a_c0 + 1
        a10 = (a_r0 + 1) * m + a_c0
        a11 = (a_r0 + 1) * m + a_c0 + 1
        
        # B elements
        b00 = b_r0 * p + b_c0
        b01 = b_r0 * p + b_c0 + 1
        b10 = (b_r0 + 1) * p + b_c0
        b11 = (b_r0 + 1) * p + b_c0 + 1
        
        # C elements
        c00 = c_c0 * n + c_r0
        c01 = (c_c0 + 1) * n + c_r0
        c10 = c_c0 * n + c_r0 + 1
        c11 = (c_c0 + 1) * n + c_r0 + 1
        
        # Strassen Terms
        # M1 = (A00+A11)(B00+B11)
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[a00][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a01][col] += 1; U[a11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1
        col += 1

    def add_schoolbook_2x4x1(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        # Adds a rank-8 schoolbook decomposition for 2x4x1 multiplication
        # A is 2x4 at (a_r0, a_c0), B is 4x1 at (b_r0, b_c0), C is 2x1 at (c_r0, c_c0)
        
        # C[0] = A[0][0]B[0][0] + A[0][1]B[1][0] + A[0][2]B[2][0] + A[0][3]B[3][0]
        # C[1] = A[1][0]B[0][0] + A[1][1]B[1][0] + A[1][2]B[2][0] + A[1][3]B[3][0]
        
        c0_idx = c_c0 * n + c_r0
        c1_idx = c_c0 * n + c_r0 + 1
        
        for k in range(4):
            # Term for C[0]
            u_idx = a_r0 * m + a_c0 + k
            v_idx = (b_r0 + k) * p + b_c0
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[c0_idx][col] = 1
            col += 1
            
            # Term for C[1]
            u_idx = (a_r0 + 1) * m + a_c0 + k
            v_idx = (b_r0 + k) * p + b_c0
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[c1_idx][col] = 1
            col += 1

    # 1. C_left (2x2) from A (2x4) and B_left (4x2)
    # Split A into A_top (2x2) and A_bot (2x2)
    # Split B_left into B_top (2x2) and B_bot (2x2)
    
    # A_top * B_top -> C_left_top_part
    add_strassen_2x2x2(0, 0, 0, 0, 0, 0)
    
    # A_bot * B_bot -> C_left_bot_part
    add_strassen_2x2x2(0, 2, 2, 0, 0, 0)

    # 2. C_right (2x3) from A (2x4) and B_right (4x3)
    # Split B_right into B_mid (4x1) and B_far (4x2)
    
    # A * B_mid -> C_mid (2x1)
    add_schoolbook_2x4x1(0, 0, 0, 2, 0, 2)
    
    # A * B_far -> C_far (2x2)
    # Split A into A_top and A_bot
    # Split B_far into B_far_top (2x2) and B_far_bot (2x2)
    
    # A_top * B_far_top -> C_far_top_part
    add_strassen_2x2x2(0, 0, 0, 3, 0, 3)
    
    # A_bot * B_far_bot -> C_far_bot_part
    add_strassen_2x2x2(0, 2, 2, 3, 0, 3)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
