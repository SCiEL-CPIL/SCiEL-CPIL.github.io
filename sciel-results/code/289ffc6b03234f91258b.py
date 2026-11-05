"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    
    We use a block decomposition approach:
    - Split A (3x4) into two parts: A_top (2x4) and A_bottom (1x4)
    - Split B (4x8) into two parts: B_left (4x2) and B_right (4x6)
    - Split C (3x8) into four blocks: C_top_left (2x2), C_top_right (2x6), C_bot_left (1x2), C_bot_right (1x6)
    
    For C_top_left (2x2), we use Strassen's algorithm with rank 7.
    For the other blocks, we use schoolbook multiplication.
    
    Total rank = 7 (Strassen) + 2*4*6 (top_right) + 1*4*2 (bot_left) + 1*4*6 (bot_right)
              = 7 + 48 + 8 + 24 = 87
    """
    n, m, p = DIMENSIONS  # n=3, m=4, p=8
    
    # Total rank calculation
    strassen_rank = 7
    top_right_rank = 2 * 4 * 6  # 48
    bot_left_rank = 1 * 4 * 2   # 8
    bot_right_rank = 1 * 4 * 6  # 24
    total_rank = strassen_rank + top_right_rank + bot_left_rank + bot_right_rank
    
    # Initialize matrices
    U = [[0] * total_rank for _ in range(n * m)]  # shape (12, total_rank)
    V = [[0] * total_rank for _ in range(m * p)]  # shape (32, total_rank)
    W = [[0] * total_rank for _ in range(p * n)]  # shape (24, total_rank)
    
    col = 0
    
    # Part 1: Strassen for C_top_left (2x2) = A_top (2x2) * B_top_left (2x2)
    # A_top = A[0:2, 0:2], B_top_left = B[0:2, 0:2]
    # Strassen's 7 products for 2x2 matrix multiplication:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Strassen U vectors (for A part, 2x2 -> flattened to 4 elements)
    # Each U vector has 4 entries: [A00, A01, A10, A11]
    strassen_U = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [1, 0, -1, 0], # M6: A10 - A00
        [0, 1, 0, -1], # M7: A01 - A11
    ]
    
    # Strassen V vectors (for B part, 2x2 -> flattened to 4 elements)
    # Each V vector has 4 entries: [B00, B01, B10, B11]
    strassen_V = [
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [0, 0, 1, -1], # M4: B10 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1],  # M7: B10 + B11
    ]
    
    # Strassen W vectors (for C part, 2x2 -> flattened to 4 elements)
    # Each W vector has 4 entries: [C00, C01, C10, C11]
    strassen_W = [
        [1, 0, 0, 1],  # M1: contributes to C00 and C11
        [0, 0, 1, -1], # M2: contributes to C10 and C11
        [0, 1, 0, 0],  # M3: contributes to C01 and C11
        [1, 0, 1, 0],  # M4: contributes to C00 and C10
        [-1, 1, 0, 0], # M5: contributes to C00 and C01
        [0, 0, 0, 1],  # M6: contributes to C11
        [1, 0, 0, 0],  # M7: contributes to C00
    ]
    
    for s in range(7):
        # U part: A[0:2, 0:2] -> U[i*m + j] for i=0,1; j=0,1
        U[0 * m + 0][col] = strassen_U[s][0]  # A00
        U[0 * m + 1][col] = strassen_U[s][1]  # A01
        U[1 * m + 0][col] = strassen_U[s][2]  # A10
        U[1 * m + 1][col] = strassen_U[s][3]  # A11
        
        # V part: B[0:2, 0:2] -> V[j*p + k] for j=0,1; k=0,1
        V[0 * p + 0][col] = strassen_V[s][0]  # B00
        V[0 * p + 1][col] = strassen_V[s][1]  # B01
        V[1 * p + 0][col] = strassen_V[s][2]  # B10
        V[1 * p + 1][col] = strassen_V[s][3]  # B11
        
        # W part: C[0:2, 0:2] -> W[k*n+i] for k=0,1; i=0,1
        W[0 * n + 0][col] = strassen_W[s][0]  # C00: k=0, i=0
        W[0 * n + 1][col] = strassen_W[s][1]  # C01: k=1, i=0
        W[1 * n + 0][col] = strassen_W[s][2]  # C10: k=0, i=1
        W[1 * n + 1][col] = strassen_W[s][3]  # C11: k=1, i=1
        
        col += 1
    
    # Part 2: Schoolbook for C_top_right (2x6) = A_top (2x4) * B_right (4x6)
    # C_top_right: rows i=0,1; cols k=2,3,4,5,6,7
    # A part: i=0,1; j=0,1,2,3
    # B part: j=0,1,2,3; k=2,3,4,5,6,7
    for i in range(2):
        for j in range(4):
            for k in range(2, 8):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Part 3: Schoolbook for C_bot_left (1x2) = A_bottom (1x4) * B_top_left (4x2)
    # C_bot_left: row i=2; cols k=0,1
    # A part: i=2; j=0,1,2,3
    # B part: j=0,1,2,3; k=0,1
    for j in range(4):
        for k in range(2):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    # Part 4: Schoolbook for C_bot_right (1x6) = A_bottom (1x4) * B_right (4x6)
    # C_bot_right: row i=2; cols k=2,3,4,5,6,7
    # A part: i=2; j=0,1,2,3
    # B part: j=0,1,2,3; k=2,3,4,5,6,7
    for j in range(4):
        for k in range(2, 8):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    assert col == total_rank, f"col={col}, expected {total_rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
