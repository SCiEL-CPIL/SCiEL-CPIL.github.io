"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    
    # Rank calculation:
    # Decomposition:
    # A (2x4) = [A1 (2x2), A2 (2x2)]
    # B (4x6) = [B1 (2x6); B2 (2x6)]
    # C = A1*B1 + A2*B2
    # Each A_i (2x2) * B_i (2x6) is split into 3 (2x2)*(2x2) multiplications.
    # Each (2x2)*(2x2) uses Strassen (rank 7).
    # Total rank = 2 * 3 * 7 = 42.
    
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42
    
    col = 0
    
    # Strassen coefficients
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    # C indices: 00, 01, 10, 11
    
    # M1: (a00+a11)(b00+b11)
    # M2: (a10+a11)b00
    # M3: a00(b01-b11)
    # M4: a11(b10-b00)
    # M5: (a00+a01)b11
    # M6: (a10-a00)(b00+b01)
    # M7: (a01-a11)(b10+b11)
    
    strassen_U = [
        [1, 0, 1, 0, 1, -1, 0], # a00
        [0, 0, 0, 0, 1, 0, 1],  # a01
        [0, 1, 0, 0, 0, 1, 0],  # a10
        [1, 1, 0, 1, 0, 0, -1]  # a11
    ]
    
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0], # b00
        [0, 0, 1, 0, 0, 1, 0],  # b01
        [0, 0, 0, 1, 0, 0, 1],  # b10
        [1, 0, -1, 0, 1, 0, 1]  # b11
    ]
    
    strassen_W = [
        [1, 0, 0, 1, 1, 0, 1],  # c00
        [0, 0, 1, 0, 1, 0, 0],  # c01
        [0, 1, 0, 1, 0, 0, 0],  # c10
        [1, -1, 1, 0, 0, 1, 0]  # c11
    ]
    
    # Loop over A blocks (2 blocks: A1, A2)
    for a_block in range(2):
        a_col_offset = a_block * 2
        
        # Loop over B blocks corresponding to A_block
        # B1 (rows 0-1) for A1, B2 (rows 2-3) for A2
        b_row_offset = a_block * 2
        
        # B_block is 2x6. Split into 3 2x2 blocks.
        for b_col_block in range(3):
            b_col_offset = b_col_block * 2
            
            # Add 7 columns for this 2x2 by 2x2 multiplication
            for s in range(7):
                # Update U
                for i_local in range(2):
                    for j_local in range(2):
                        val = strassen_U[i_local * 2 + j_local][s]
                        if val != 0:
                            u_idx = i_local * m + (a_col_offset + j_local)
                            U[u_idx][col] += val
                
                # Update V
                for k_local in range(2):
                    for l_local in range(2):
                        val = strassen_V[k_local * 2 + l_local][s]
                        if val != 0:
                            v_idx = (b_row_offset + k_local) * p + (b_col_offset + l_local)
                            V[v_idx][col] += val
                
                # Update W
                for k_local in range(2):
                    for i_local in range(2):
                        val = strassen_W[k_local * 2 + i_local][s]
                        if val != 0:
                            w_idx = (b_col_offset + k_local) * n + i_local
                            W[w_idx][col] += val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
