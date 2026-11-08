"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use 4 independent Strassen decompositions for 2x2 blocks.
    # Rank will be 4 * 7 = 28.
    rank = 28
    
    # Initialize matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Define Strassen terms for 2x2 multiplication
    # X is 2x2, Y is 2x2.
    # M1: (X00+X11) * (Y00+Y11)
    # M2: (X10+X11) * Y00
    # M3: X00 * (Y01-Y11)
    # M4: X11 * (Y10-Y00)
    # M5: (X00+X01) * Y11
    # M6: (X10-X00) * (Y00+Y01)
    # M7: (X01-X11) * (Y10+Y11)
    
    # Coefficients for U (X part)
    # Format: [ [coeff for X00, coeff for X10, coeff for X01, coeff for X11], ... for each M ]
    # Note: X indices in 2x2 local:
    # X00: i=0, j=0
    # X10: i=1, j=0
    # X01: i=0, j=1
    # X11: i=1, j=1
    U_strassen = [
        [1, 0, 0, 1],   # M1: X00+X11
        [0, 1, 0, 1],   # M2: X10+X11
        [1, 0, 0, 0],   # M3: X00
        [0, 0, 0, 1],   # M4: X11
        [1, 0, 1, 0],   # M5: X00+X01
        [-1, 1, 0, 0],  # M6: X10-X00
        [0, 0, 1, -1]   # M7: X01-X11
    ]
    
    # Coefficients for V (Y part)
    # Y indices in 2x2 local:
    # Y00: j=0, k=0
    # Y01: j=0, k=1
    # Y10: j=1, k=0
    # Y11: j=1, k=1
    V_strassen = [
        [1, 0, 0, 1],   # M1: Y00+Y11
        [1, 0, 0, 0],   # M2: Y00
        [0, 1, 0, -1],  # M3: Y01-Y11
        [0, 0, 1, -1],  # M4: Y10-Y00
        [0, 0, 0, 1],   # M5: Y11
        [1, 1, 0, 0],   # M6: Y00+Y01
        [0, 0, 1, 1]    # M7: Y10+Y11
    ]
    
    # Coefficients for W (Z part)
    # Z indices in 2x2 local:
    # Z00: i=0, k=0
    # Z01: i=0, k=1
    # Z10: i=1, k=0
    # Z11: i=1, k=1
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    W_strassen = [
        [1, 0, 0, 1],   # M1: Z00+=1, Z11+=1
        [0, 1, -1, 0],  # M2: Z10+=1, Z11-=-1
        [0, -1, 1, 1],  # M3: Z01+=1, Z11+=1
        [1, 0, 1, 0],   # M4: Z00+=1, Z10+=1
        [-1, 1, 0, 0],  # M5: Z00-=-1, Z01+=1
        [0, 0, 0, 1],   # M6: Z11+=1
        [1, 0, 0, 0]    # M7: Z00+=1
    ]
    
    # We have 4 blocks to process:
    # Block 1: A_left (cols 0,1) * B_top_left (rows 0,1, cols 0,1) -> C_left (cols 0,1)
    # Block 2: A_right (cols 2,3) * B_bottom_left (rows 2,3, cols 0,1) -> C_left (cols 0,1)
    # Block 3: A_left (cols 0,1) * B_top_right (rows 0,1, cols 2,3) -> C_right (cols 2,3)
    # Block 4: A_right (cols 2,3) * B_bottom_right (rows 2,3, cols 2,3) -> C_right (cols 2,3)
    
    # Define offsets for each block
    # (A_col_offset, B_row_offset, B_col_offset, C_col_offset)
    blocks = [
        (0, 0, 0, 0),  # Block 1
        (2, 2, 0, 0),  # Block 2
        (0, 0, 2, 2),  # Block 3
        (2, 2, 2, 2)   # Block 4
    ]
    
    for b_idx, (a_off, b_r_off, b_c_off, c_off) in enumerate(blocks):
        for s_idx in range(7):
            col = b_idx * 7 + s_idx
            
            # Update U
            # U indices: i (0,1), j_local (0,1) -> j_global = a_off + j_local
            for i_local in range(2):
                for j_local in range(2):
                    u_val = U_strassen[s_idx][i_local * 2 + j_local]
                    if u_val != 0:
                        i_global = i_local
                        j_global = a_off + j_local
                        U[i_global * m + j_global][col] += u_val
                        
            # Update V
            # V indices: j_local (0,1), k_local (0,1) -> j_global = b_r_off + j_local, k_global = b_c_off + k_local
            for j_local in range(2):
                for k_local in range(2):
                    v_val = V_strassen[s_idx][j_local * 2 + k_local]
                    if v_val != 0:
                        j_global = b_r_off + j_local
                        k_global = b_c_off + k_local
                        V[j_global * p + k_global][col] += v_val
                        
            # Update W
            # W indices: k_local (0,1), i_local (0,1) -> k_global = c_off + k_local, i_global = i_local
            # Note: W_strassen is indexed by [i_local, k_local] in the Z matrix?
            # My W_strassen definition above:
            # W_strassen[s_idx][i_local * 2 + k_local]
            # Let's re-verify the W_strassen construction.
            # Z00 (i=0, k=0): M1, M4, -M5, M7
            # Z01 (i=0, k=1): M3, M5
            # Z10 (i=1, k=0): M2, M4
            # Z11 (i=1, k=1): M1, -M2, M3, M6
            
            # My W_strassen list:
            # M1: [1, 0, 0, 1] -> Z00+=1, Z11+=1. Correct.
            # M2: [0, 1, -1, 0] -> Z10+=1, Z11-=-1. Correct.
            # M3: [0, -1, 1, 1] -> Z01+=1, Z11+=1. Correct.
            # M4: [1, 0, 1, 0] -> Z00+=1, Z10+=1. Correct.
            # M5: [-1, 1, 0, 0] -> Z00-=-1, Z01+=1. Correct.
            # M6: [0, 0, 0, 1] -> Z11+=1. Correct.
            # M7: [1, 0, 0, 0] -> Z00+=1. Correct.
            
            for i_local in range(2):
                for k_local in range(2):
                    w_val = W_strassen[s_idx][i_local * 2 + k_local]
                    if w_val != 0:
                        i_global = i_local
                        k_global = c_off + k_local
                        W[k_global * n + i_global][col] += w_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
