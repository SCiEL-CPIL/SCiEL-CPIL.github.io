"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen-like rank-7 decomposition factors for 2x2 by 2x2 matrix multiplication
    # Local indices for A, B, C are 0..3 (row-major: 00, 01, 10, 11)
    u_factors = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0], # P6
        [0, 1, 0, -1]  # P7
    ]
    v_factors = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1], # P3
        [-1, 0, 1, 0], # P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    w_factors = [
        [1, 0, 0, 1], # P1 -> C00, C11
        [0, 0, 1, -1], # P2 -> C10, C11
        [0, 1, 0, 1], # P3 -> C01, C11
        [1, 0, 1, 0], # P4 -> C00, C10
        [-1, 1, 0, 0], # P5 -> C00, C01
        [0, 0, 0, 1], # P6 -> C11
        [1, 0, 0, 0]  # P7 -> C00
    ]
    
    # We have 6 blocks: 2 row-blocks of A (size 2 cols) x 3 col-blocks of B (size 2 cols)
    # Actually, A is 2x4. Split into A1 (cols 0,1) and A2 (cols 2,3).
    # B is 4x6. Split into 6 blocks of 2x2:
    # B11 (rows 0,1; cols 0,1), B12 (rows 0,1; cols 2,3), B13 (rows 0,1; cols 4,5)
    # B21 (rows 2,3; cols 0,1), B22 (rows 2,3; cols 2,3), B23 (rows 2,3; cols 4,5)
    # C = A1 B1* + A2 B2*
    # Block 0: A1, B11 -> C11 part
    # Block 1: A1, B12 -> C12 part
    # Block 2: A1, B13 -> C13 part
    # Block 3: A2, B21 -> C11 part
    # Block 4: A2, B22 -> C12 part
    # Block 5: A2, B23 -> C13 part
    
    for b in range(6):
        a_block_idx = b // 3  # 0 for A1, 1 for A2
        c_block_idx = b % 3   # 0, 1, 2 for C col groups
        
        base_col = 7 * b
        
        for c in range(7):
            col = base_col + c
            
            # Update U
            for r_local in range(4):
                val = u_factors[c][r_local]
                if val != 0:
                    i_local = r_local // 2
                    j_local = r_local % 2
                    # Global A index: row i_local, col (2*a_block_idx + j_local)
                    g_row = i_local * m + (2 * a_block_idx + j_local)
                    U[g_row][col] = val
            
            # Update V
            for r_local in range(4):
                val = v_factors[c][r_local]
                if val != 0:
                    r_b_local = r_local // 2
                    c_b_local = r_local % 2
                    # Global B index: row (2*a_block_idx + r_b_local), col (2*c_block_idx + c_b_local)
                    g_row = (2 * a_block_idx + r_b_local) * p + (2 * c_block_idx + c_b_local)
                    V[g_row][col] = val
            
            # Update W
            for r_local in range(4):
                val = w_factors[c][r_local]
                if val != 0:
                    i_c_local = r_local // 2
                    c_c_local = r_local % 2
                    # Global C index: row i_c_local, col (2*c_block_idx + c_c_local)
                    g_row = i_c_local * p + (2 * c_block_idx + c_c_local)
                    W[g_row][col] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
