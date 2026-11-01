import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _strassen_4x4_decomposition():
    """
    Constructs the rank-49 Strassen decomposition for 4x4x4 matrix multiplication.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Top-level Strassen combinations for 4x4 blocks (2x2 blocks of 2x2 matrices)
    # Each entry is (linear_combination_of_A_blocks, linear_combination_of_B_blocks)
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B11, B10, B01 (note order in some formulas)
    
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B01 - B11)
    # M3 = A11(B10 - B00)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    
    top_level_products = [
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]), # M0
        ([ (1,0,1), (1,1,1) ], [ (0,0,1) ]),           # M1
        ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ]),          # M2
        ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ]),          # M3
        ([ (0,0,1), (0,1,1) ], [ (1,1,1) ]),           # M4
        ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ]), # M5
        ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ])  # M6
    ]
    
    # Bottom-level Strassen combinations for 2x2 blocks
    # S0 = (x00 + x11)(y00 + y11)
    # S1 = (x10 + x11)y00
    # S2 = x00(y01 - y11)
    # S3 = x11(y10 - y00)
    # S4 = (x00 + x01)y11
    # S5 = (x10 - x00)(y00 + y01)
    # S6 = (x01 - x11)(y10 + y11)
    
    bottom_level_products = [
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]), # S0
        ([ (1,0,1), (1,1,1) ], [ (0,0,1) ]),           # S1
        ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ]),          # S2
        ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ]),          # S3
        ([ (0,0,1), (0,1,1) ], [ (1,1,1) ]),           # S4
        ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ]), # S5
        ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ])  # S6
    ]
    
    # How the bottom-level products S0..S6 combine to form the 2x2 result Z
    # Z00 = S0 + S3 - S4 + S6
    # Z01 = S2 + S4
    # Z10 = S1 + S3
    # Z11 = S0 - S1 + S2 + S5
    bottom_level_combinations = [
        [1, 0, 0, 1, -1, 0, 1], # Z00 coefficients for S0..S6
        [0, 0, 1, 0, 1, 0, 0],  # Z01
        [0, 1, 0, 1, 0, 0, 0],  # Z10
        [1, -1, 1, 0, 0, 1, 0]  # Z11
    ]
    
    # How the top-level products M0..M6 combine to form the 4x4 result C
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    top_level_combinations = [
        [1, 0, 0, 1, -1, 0, 1], # C00 (block) coefficients for M0..M6
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Helper to add a term to U, V, W
    def add_term(rank_idx, a_coeffs, b_coeffs, c_coeffs):
        for (br, bc, coeff) in a_coeffs:
            # br, bc are block indices (0 or 1)
            # We need to expand this into individual matrix entries
            pass # Placeholder, will do it in the main loop
    
    # Iterate over each top-level product M_i
    for i, (a_blocks, b_blocks) in enumerate(top_level_products):
        # a_blocks is a list of (block_row, block_col, coeff)
        # b_blocks is a list of (block_row, block_col, coeff)
        
        # For each bottom-level product S_j within this M_i
        for j, (a_local, b_local) in enumerate(bottom_level_products):
            rank_idx = i * 7 + j
            
            # Construct the linear form for U (input A)
            # U entries are indexed by (row, col) in the 4x4 matrix
            # A block (br, bc) corresponds to rows 2*br..2*br+1, cols 2*bc..2*bc+1
            # a_local is a list of (local_row, local_col, coeff)
            # The actual A entry is A[2*br + local_row, 2*bc + local_col]
            
            # We need to combine the coefficients from a_blocks and a_local
            # The linear form for A in this rank-1 term is:
            # sum_{(br,bc,ca) in a_blocks} sum_{(lr,lc,cl) in a_local} ca*cl * A[2*br+lr, 2*bc+lc]
            
            a_terms = {}
            for (br, bc, ca) in a_blocks:
                for (lr, lc, cl) in a_local:
                    row = 2 * br + lr
                    col = 2 * bc + lc
                    a_terms[(row, col)] = a_terms.get((row, col), 0) + ca * cl
            
            b_terms = {}
            for (br, bc, cb) in b_blocks:
                for (lr, lc, cl) in b_local:
                    row = 2 * br + lr
                    col = 2 * bc + lc
                    b_terms[(row, col)] = b_terms.get((row, col), 0) + cb * cl
            
            # Fill U and V
            for (row, col), coeff in a_terms.items():
                if coeff != 0:
                    U[row * m + col][rank_idx] += coeff
            
            for (row, col), coeff in b_terms.items():
                if coeff != 0:
                    V[row * p + col][rank_idx] += coeff
            
            # Now determine the contribution to W
            # The product M_i contributes to C blocks according to top_level_combinations[i]
            # The product S_j (within M_i) contributes to the local 2x2 result according to bottom_level_combinations[j]
            
            # C block (cbr, cbc) gets coefficient top_level_combinations[cbr*2+cbc][i]
            # Local entry (lrow, lcol) within that block gets coefficient bottom_level_combinations[lrow*2+lcol][j]
            
            for cbr in range(2):
                for cbc in range(2):
                    c_block_coeff = top_level_combinations[cbr * 2 + cbc][i]
                    if c_block_coeff == 0:
                        continue
                    
                    for lrow in range(2):
                        for lcol in range(2):
                            c_local_coeff = bottom_level_combinations[lrow * 2 + lcol][j]
                            if c_local_coeff == 0:
                                continue
                            
                            c_row = 2 * cbr + lrow
                            c_col = 2 * cbc + lcol
                            total_coeff = c_block_coeff * c_local_coeff
                            
                            if total_coeff != 0:
                                # W index is k*n + i, where k is col of C, i is row of C
                                W[c_col * n + c_row][rank_idx] += total_coeff
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS

    # Generate the rank-49 Strassen decomposition
    result = _strassen_4x4_decomposition()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])

    # Verify correctness
    if _verify_decomposition(U, V, W, n, m, p, rank):
        return result

    # Fallback to schoolbook if verification fails (should not happen)
    return _schoolbook_decomposition(n, m, p)

# EVOLVE-BLOCK-END
