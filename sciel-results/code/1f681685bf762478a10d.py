"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen's algorithm on 2x2 blocks
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen multipliers definitions
    # Each entry is (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {local_idx: coeff} for A (2x2)
    # V_coeffs: dict {local_idx: coeff} for B (2x2)
    # W_coeffs: dict {local_idx: coeff} for C (2x2)
    
    strassen_terms = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    col = 0
    # 8 blocks: 4 for Left (A_L, B_L) and 4 for Right (A_R, B_R)
    for block in range(8):
        is_left = block < 4
        block_idx = block % 4
        
        # Determine global index offsets for A (2x2 part)
        if is_left:
            a_i_offset = 0
            a_j_offset = 0
        else:
            a_i_offset = 0
            a_j_offset = 2
            
        # Determine global index offsets for B (2x2 part)
        # B is 4x8. Rows 0,1 for Left, 2,3 for Right.
        if is_left:
            b_i_offset = 0
        else:
            b_i_offset = 2
        b_j_offset = block_idx * 2
        
        # Determine global index offsets for C (2x2 part)
        # C is 2x8. Columns match B's column offsets.
        c_i_offset = 0
        c_j_offset = block_idx * 2
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            # Fill U
            for (li, lj), val in u_coeffs.items():
                gi = a_i_offset + li
                gj = a_j_offset + lj
                U[gi * m + gj][col] = val
            
            # Fill V
            for (li, lj), val in v_coeffs.items():
                gi = b_i_offset + li
                gj = b_j_offset + lj
                V[gi * p + gj][col] = val
            
            # Fill W
            for (li, lj), val in w_coeffs.items():
                gi = c_i_offset + li
                gj = c_j_offset + lj
                # W index is k*n + i, so gj is k, gi is i
                W[gj * n + gi][col] = val
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
