"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using Strassen's algorithm.
    
    This applies Strassen's 7-multiplication algorithm for 2x2 matrices recursively.
    The 4x4 matrices are viewed as 2x2 blocks of 2x2 matrices.
    Each of the 7 block-level multiplications is expanded into 7 scalar multiplications
    using Strassen again, resulting in 7 * 7 = 49 scalar multiplications.
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Helper to define a Strassen step on 2x2 blocks
    # Inputs: indices for A blocks (4 indices for each of the 4 sub-blocks? No, we pass linear combos)
    # Actually, it's easier to define the 7 top-level linear combinations, 
    # and then for each, define the 7 bottom-level linear combinations.
    
    # Top-level Strassen on 4x4 matrices (viewed as 2x2 blocks of 2x2)
    # Let A = [[A00, A01], [A10, A11]] and B = [[B00, B01], [B10, B11]]
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
    
    # We need to expand each Mk (which is a 2x2 matrix product) into 7 scalar products.
    # For a 2x2 matrix product X * Y, the Strassen components are:
    # S1 = (X00 + X11) * (Y00 + Y11)
    # S2 = (X10 + X11) * Y00
    # S3 = X00 * (Y01 - Y11)
    # S4 = X11 * (Y10 - Y00)
    # S5 = (X00 + X01) * Y11
    # S6 = (X10 - X00) * (Y00 + Y01)
    # S7 = (X01 - X11) * (Y10 + Y11)
    
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    
    # We will iterate through the 7 top-level terms (M1..M7).
    # For each, we determine the linear combinations of A-blocks and B-blocks.
    # Then for each, we iterate through the 7 bottom-level terms (S1..S7).
    # For each S, we determine the linear combinations of scalar elements of the A-sub-blocks and B-sub-blocks.
    # And we add the contribution to the output C-blocks.
    
    # Indices for 4x4 matrix (row-major):
    # Block (0,0): 0, 1, 4, 5
    # Block (0,1): 2, 3, 6, 7
    # Block (1,0): 8, 9, 12, 13
    # Block (1,1): 10, 11, 14, 15
    
    # Helper to get indices for a 2x2 block at (br, bc) in a 4x4 matrix
    def get_block_indices(br, bc):
        # br, bc in {0,1}
        # Returns list of 4 indices [00, 01, 10, 11] within the block
        # Global row start = br * 2, col start = bc * 2
        r_start = br * 2
        c_start = bc * 2
        return [
            r_start * 4 + c_start,       # (0,0)
            r_start * 4 + c_start + 1,   # (0,1)
            (r_start + 1) * 4 + c_start, # (1,0)
            (r_start + 1) * 4 + c_start + 1 # (1,1)
        ]
    
    # Define the 7 top-level Strassen terms for the 4x4 level
    # Each term is (A_coeff_dict, B_coeff_dict)
    # Keys are (br, bc) for the 2x2 blocks
    top_terms = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
    ]
    
    # Define the 7 bottom-level Strassen terms for a 2x2 matrix product X*Y
    # Each term is (X_coeff_dict, Y_coeff_dict)
    # Keys are (r,c) in {0,1} for the 2x2 sub-matrix
    bot_terms = [
        # S1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # S2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        # S3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        # S4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        # S5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # S6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # S7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
    ]
    
    # Output block contributions for top-level terms
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    out_contrib = [
        [(0,0, 1), (0,0, 0), (0,0, 0), (0,0, 0), (0,0, 0), (0,0, 0), (0,0, 0)], # M1 -> C00 (+1)
        [(0,0, 0), (0,0, 0), (0,0, 0), (0,0, 0), (0,0, 0), (0,0, 0), (0,0, 0)], # Placeholder, we'll handle mapping differently
    ]
    
    # It is easier to map which top-term contributes to which output block with what sign
    top_to_out = {
        0: [(0,0, 1)],   # M1 -> C00 (+1)
        1: [(1,0, 1)],   # M2 -> C10 (+1)
        2: [(0,1, 1)],   # M3 -> C01 (+1)
        3: [(0,0, 1), (1,0, 1)], # M4 -> C00 (+1), C10 (+1) -- Wait, C00=M1+M4-M5+M7, C10=M2+M4
        4: [(0,0, -1), (0,1, 1)], # M5 -> C00 (-1), C01 (+1)
        5: [(1,1, 1)],   # M6 -> C11 (+1)
        6: [(0,0, 1), (1,1, 1)], # M7 -> C00 (+1), C11 (+1) -- Wait, C11=M1-M2+M3+M6. M7 is in C00.
        # Let's re-verify Strassen output:
        # C11 (top left) = M1 + M4 - M5 + M7
        # C12 (top right) = M3 + M5
        # C21 (bottom left) = M2 + M4
        # C22 (bottom right) = M1 - M2 + M3 + M6
        
        # So:
        # M1 (idx 0): C11(+1), C22(+1)
        # M2 (idx 1): C21(+1), C22(-1)
        # M3 (idx 2): C12(+1), C22(+1)
        # M4 (idx 3): C11(+1), C21(+1)
        # M5 (idx 4): C11(-1), C12(+1)
        # M6 (idx 5): C22(+1)
        # M7 (idx 6): C11(+1)
    }
    
    # Correct mapping:
    # M1: C00(+1), C11(+1)
    # M2: C10(+1), C11(-1)
    # M3: C01(+1), C11(+1)
    # M4: C00(+1), C10(+1)
    # M5: C00(-1), C01(+1)
    # M6: C11(+1)
    # M7: C00(+1)
    
    top_contrib_map = [
        [(0,0, 1), (1,1, 1)], # M1
        [(1,0, 1), (1,1, -1)], # M2
        [(0,1, 1), (1,1, 1)], # M3
        [(0,0, 1), (1,0, 1)], # M4
        [(0,0, -1), (0,1, 1)], # M5
        [(1,1, 1)], # M6
        [(0,0, 1)], # M7
    ]
    
    # Bottom-level output mapping for 2x2 Strassen (X*Y -> Z)
    # Z00 = S1 + S4 - S5 + S7
    # Z01 = S3 + S5
    # Z10 = S2 + S4
    # Z11 = S1 - S2 + S3 + S6
    
    bot_contrib_map = [
        [(0,0, 1), (1,1, 1)], # S1
        [(1,0, 1), (1,1, -1)], # S2
        [(0,1, 1), (1,1, 1)], # S3
        [(0,0, 1), (1,0, 1)], # S4
        [(0,0, -1), (0,1, 1)], # S5
        [(1,1, 1)], # S6
        [(0,0, 1)], # S7
    ]
    
    col = 0
    for t_idx, (a_blocks, b_blocks) in enumerate(top_terms):
        for b_idx, (x_sub, y_sub) in enumerate(bot_terms):
            # For this scalar multiplication, we need to sum up coefficients
            # for each scalar index in U (A), V (B), and W (C)
            
            u_coeffs = {}
            v_coeffs = {}
            w_coeffs = {}
            
            # Expand A part
            for (abr, abc), ac in a_blocks.items():
                # ac is the coefficient for the block (abr, abc) in the top-level term
                # We need to combine this with the bottom-level term x_sub
                # x_sub is a dict of (r,c) -> coeff for the 2x2 sub-matrix
                # The global index for A is determined by abr, abc and r, c
                for (sr, sc), xc in x_sub.items():
                    # Get global index
                    block_indices = get_block_indices(abr, abc)
                    # sr, sc are local indices in 2x2 block
                    # block_indices[0] is (0,0), [1] is (0,1), [2] is (1,0), [3] is (1,1)
                    local_idx = sr * 2 + sc
                    global_idx = block_indices[local_idx]
                    coeff = ac * xc
                    if coeff != 0:
                        u_coeffs[global_idx] = u_coeffs.get(global_idx, 0) + coeff
            
            # Expand B part
            for (bbr, bbc), bc in b_blocks.items():
                for (sr, sc), yc in y_sub.items():
                    block_indices = get_block_indices(bbr, bbc)
                    local_idx = sr * 2 + sc
                    global_idx = block_indices[local_idx]
                    coeff = bc * yc
                    if coeff != 0:
                        v_coeffs[global_idx] = v_coeffs.get(global_idx, 0) + coeff
            
            # Expand C part
            # The output of the top-level term t_idx contributes to specific C blocks
            # The output of the bottom-level term b_idx contributes to specific sub-positions within those C blocks
            for (cbr, cbc, c_sign) in top_contrib_map[t_idx]:
                # cbr, cbc is the top-level block index in C
                # c_sign is the sign
                for (sr, sc, s_sign) in bot_contrib_map[b_idx]:
                    # sr, sc is the local index in the 2x2 block
                    # s_sign is the sign
                    block_indices = get_block_indices(cbr, cbc)
                    local_idx = sr * 2 + sc
                    global_idx = block_indices[local_idx]
                    coeff = c_sign * s_sign
                    if coeff != 0:
                        w_coeffs[global_idx] = w_coeffs.get(global_idx, 0) + coeff
            
            # Add to U, V, W
            for idx, coeff in u_coeffs.items():
                if coeff != 0:
                    U[idx][col] += coeff
            for idx, coeff in v_coeffs.items():
                if coeff != 0:
                    V[idx][col] += coeff
            for idx, coeff in w_coeffs.items():
                if coeff != 0:
                    W[idx][col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
