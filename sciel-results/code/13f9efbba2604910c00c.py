"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 49 decomposition for 4x4x4 using Strassen
    # Strassen 2x2x2 terms
    # U, V, W are lists of dicts: index -> coeff
    # Indices for 2x2: 0,1,2,3
    strassen_2x2 = [
        # M1
        {"U": {(0,0): 1, (1,1): 1}, "V": {(0,0): 1, (1,1): 1}, "W": {(0,0): 1, (1,1): 1}},
        # M2
        {"U": {(1,0): 1, (1,1): 1}, "V": {(0,0): 1}, "W": {(1,0): 1}},
        # M3
        {"U": {(0,0): 1}, "V": {(0,1): 1, (1,1): -1}, "W": {(0,1): 1}},
        # M4
        {"U": {(1,1): 1}, "V": {(1,0): 1, (0,0): -1}, "W": {(0,0): 1, (1,0): 1}},
        # M5
        {"U": {(0,0): 1, (0,1): 1}, "V": {(1,1): 1}, "W": {(0,0): -1, (0,1): 1}},
        # M6
        {"U": {(1,0): 1, (0,0): -1}, "V": {(0,0): 1, (0,1): 1}, "W": {(1,1): 1}},
        # M7
        {"U": {(0,1): 1, (1,1): -1}, "V": {(1,0): 1, (1,1): 1}, "W": {(0,0): 1}}
    ]
    
    # High-level Strassen terms for 4x4x4
    # Each term defines A_blocks, B_blocks, C_blocks
    # Block 0: (0,0), Block 1: (0,1), Block 2: (1,0), Block 3: (1,1)
    # A_blocks is list of (block_idx, coeff)
    high_level_terms = [
        # M1: (A11+A22)(B11+B22) -> C11, C22
        {"A": [(0, 1), (3, 1)], "B": [(0, 1), (3, 1)], "C": [(0, 1), (3, 1)]},
        # M2: (A21+A22)B11 -> C21
        {"A": [(2, 1), (3, 1)], "B": [(0, 1)], "C": [(2, 1)]},
        # M3: A11(B12-B22) -> C12
        {"A": [(0, 1)], "B": [(1, 1), (3, -1)], "C": [(1, 1)]},
        # M4: A22(B21-B11) -> C11, C21
        {"A": [(3, 1)], "B": [(2, 1), (0, -1)], "C": [(0, 1), (2, 1)]},
        # M5: (A11+A12)B22 -> -C11, C12
        {"A": [(0, 1), (1, 1)], "B": [(3, 1)], "C": [(0, -1), (1, 1)]},
        # M6: (A21-A11)(B11+B12) -> C22
        {"A": [(2, 1), (0, -1)], "B": [(0, 1), (1, 1)], "C": [(3, 1)]},
        # M7: (A12-A22)(B21+B22) -> C11
        {"A": [(1, 1), (3, -1)], "B": [(2, 1), (3, 1)], "C": [(0, 1)]}
    ]
    
    rank_49 = 49
    U_49 = [[0] * rank_49 for _ in range(16)]
    V_49 = [[0] * rank_49 for _ in range(16)]
    W_49 = [[0] * rank_49 for _ in range(16)]
    
    for h_idx, term in enumerate(high_level_terms):
        for l_idx, s_term in enumerate(strassen_2x2):
            c_idx = h_idx * 7 + l_idx
            
            # Process U
            for (i_low, j_low), coeff_low in s_term["U"].items():
                coeff_total = 0
                for b_idx, coeff_block in term["A"]:
                    # Block b_idx has row offset 2*(b_idx//2), col offset 2*(b_idx%2)
                    i_glob = 2 * (b_idx // 2) + i_low
                    j_glob = 2 * (b_idx % 2) + j_low
                    # Add to U[i_glob*4 + j_glob]
                    U_49[i_glob * 4 + j_glob][c_idx] += coeff_block * coeff_low
            
            # Process V
            for (j_low, k_low), coeff_low in s_term["V"].items():
                for b_idx, coeff_block in term["B"]:
                    j_glob = 2 * (b_idx // 2) + j_low
                    k_glob = 2 * (b_idx % 2) + k_low
                    V_49[j_glob * 4 + k_glob][c_idx] += coeff_block * coeff_low
            
            # Process W
            for (k_low, i_low), coeff_low in s_term["W"].items():
                for b_idx, coeff_block in term["C"]:
                    k_glob = 2 * (b_idx // 2) + k_low
                    i_glob = 2 * (b_idx % 2) + i_low
                    W_49[k_glob * 4 + i_glob][c_idx] += coeff_block * coeff_low

    # Schoolbook for 4x4x1 (last 1 column of B)
    rank_schoolbook = 16
    U_sb = [[0] * rank_schoolbook for _ in range(16)]
    V_sb = [[0] * rank_schoolbook for _ in range(4)] # m*p = 4*1 = 4
    W_sb = [[0] * rank_schoolbook for _ in range(4)] # p*n = 1*4 = 4
    
    col = 0
    for i in range(4):
        for j in range(4):
            U_sb[i * 4 + j][col] = 1
            V_sb[j * 1 + 0][col] = 1
            W_sb[0 * 4 + i][col] = 1
            col += 1

    # Combine
    total_rank = rank_49 + rank_schoolbook
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(36)] # m*p = 4*9 = 36
    W = [[0] * total_rank for _ in range(36)] # p*n = 9*4 = 36
    
    # Copy 4x4x4 part
    # U is 16x16, V is 16x16, W is 16x16 for the first 4 columns of B
    # B is 4x9. First 4 columns are B1, B2.
    # Wait, I split B into B1 (4x4), B2 (4x4), B3 (4x1).
    # My 49-rank decomposition handles A * B_4x4.
    # But B is 4x9.
    # I need to handle A * B1 and A * B2 separately?
    # No, the 49-rank decomposition is for A (4x4) * B (4x4).
    # If I use it for B1 (cols 0-3), it works.
    # If I use it for B2 (cols 4-7), I need another 49-rank decomposition.
    # And B3 (col 8) uses schoolbook.
    
    # So total rank = 49 (for B1) + 49 (for B2) + 16 (for B3) = 114.
    
    # Let's re-allocate
    total_rank = 49 + 49 + 16
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(36)]
    W = [[0] * total_rank for _ in range(36)]
    
    # Part 1: A * B1 (cols 0-3 of B)
    # U_49 is 16x49, V_49 is 16x49, W_49 is 16x49
    # V_49 corresponds to B1 (4x4). Indices j in 0..3, k in 0..3.
    # In global V, j is 0..3, k is 0..3.
    # So V_49[j*4+k, c] maps to V[j*9+k, c]
    for r in range(16):
        for c in range(49):
            U[r][c] = U_49[r][c]
            # V_49 index r is j*4+k. Global V index is j*9+k.
            j = r // 4
            k = r % 4
            V[j * 9 + k][c] = V_49[r][c]
            # W_49 index r is k*4+i. Global W index is k*4+i.
            W[r][c] = W_49[r][c]
            
    # Part 2: A * B2 (cols 4-7 of B)
    # Same U_49, but V_49 maps to cols 4-7 of B.
    # V_49[j*4+k, c] maps to V[j*9 + (4+k), c]
    offset_c = 49
    for r in range(16):
        for c in range(49):
            U[r][c + offset_c] = U_49[r][c]
            j = r // 4
            k = r % 4
            V[j * 9 + (4 + k)][c + offset_c] = V_49[r][c]
            W[r][c + offset_c] = W_49[r][c]
            
    # Part 3: A * B3 (col 8 of B)
    offset_c = 98
    for r in range(16):
        for c in range(16):
            U[r][c + offset_c] = U_sb[r][c]
            # V_sb index r is j*1+0 = j. Global V index is j*9 + 8.
            j = r
            V[j * 9 + 8][c + offset_c] = V_sb[r][c]
            # W_sb index r is 0*4+i = i. Global W index is 8*4 + i.
            i = r
            W[8 * 4 + i][c + offset_c] = W_sb[r][c]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
