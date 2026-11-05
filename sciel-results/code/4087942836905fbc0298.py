"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 98 using Strassen for 4x4x4 twice
    rank = 98
    
    # Initialize U, V, W
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Define the 7 rank-1 terms for 2x2x2 Strassen
    # u, v, w are length 4
    # Order: [00, 01, 10, 11]
    u2 = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    v2 = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    w2 = [
        [1, 0, 0, 1], # M1 -> C00, C11
        [0, 0, 1, -1],# M2 -> C10, C11
        [0, 1, 0, 1], # M3 -> C01, C11
        [1, 0, 1, 0], # M4 -> C00, C10
        [-1, 1, 0, 0],# M5 -> C00, C01
        [0, 0, 0, 1], # M6 -> C11
        [1, 0, 0, 0]  # M7 -> C00
    ]
    
    # Define the 7 Strassen combinations for 4x4x4
    # Each is a dict mapping block index (0..3) to coefficient
    # Block 0: 00, Block 1: 01, Block 2: 10, Block 3: 11
    # A combinations
    sA = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {0: -1, 2: 1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]
    
    # B combinations for Left Half (cols 0-3)
    sB_left = [
        {0: 1, 3: 1},       # M1: B00 + B11
        {0: 1},             # M2: B00
        {1: 1, 3: -1},      # M3: B01 - B11
        {2: 1, 0: -1},      # M4: B10 - B00
        {3: 1},             # M5: B11
        {0: 1, 1: 1},       # M6: B00 + B01
        {2: 1, 3: 1}        # M7: B10 + B11
    ]
    
    # B combinations for Right Half (cols 4-7)
    # The blocks are the same structure, just shifted in column index
    sB_right = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    
    col = 0
    for half in range(2):
        sB = sB_left if half == 0 else sB_right
        for s in range(7):
            # Get coefficients for A blocks and B blocks
            cA = sA[s]
            cB = sB[s]
            
            for t in range(7):
                # Iterate over the 4 elements of the 2x2 matrix
                for r in range(4):
                    # r corresponds to element in 2x2 block
                    # row = r // 2, col_in_block = r % 2
                    
                    # Add to U
                    # U index: global row, global col
                    # The 2x2 block is at (block_row, block_col)
                    # For A, the blocks are 00, 01, 10, 11
                    # We need to map the local 2x2 element to global 4x4 element
                    
                    # Wait, the Strassen sum P_A is a 2x2 matrix.
                    # P_A = sum cA[b] * A_b
                    # The t-th term uses u2[t][r] for the r-th element of P_A.
                    # So the coefficient for the r-th element of block b is cA[b] * u2[t][r].
                    # This element is at global position:
                    # block_row = b // 2
                    # block_col = b % 2
                    # local_row = r // 2
                    # local_col = r % 2
                    # global_row = block_row * 2 + local_row
                    # global_col = block_col * 2 + local_col
                    # U index = global_row * 4 + global_col
                    
                    for b in range(4):
                        if b in cA:
                            coeff_A = cA[b]
                            block_row = b // 2
                            block_col = b % 2
                            local_row = r // 2
                            local_col = r % 2
                            global_row = block_row * 2 + local_row
                            global_col = block_col * 2 + local_col
                            U_idx = global_row * 4 + global_col
                            U[U_idx][col] += coeff_A * u2[t][r]
                            
                    # Add to V
                    # V index: global row, global col
                    # For B, the blocks are 00, 01, 10, 11
                    # The 2x2 block is at (block_row, block_col)
                    # But B is 4x8.
                    # Left half: cols 0-3. Right half: cols 4-7.
                    # Block col 0 -> global cols 0,1 (if half 0) or 4,5 (if half 1)
                    # Block col 1 -> global cols 2,3 (if half 0) or 6,7 (if half 1)
                    
                    for b in range(4):
                        if b in cB:
                            coeff_B = cB[b]
                            block_row = b // 2
                            block_col = b % 2
                            local_row = r // 2
                            local_col = r % 2
                            global_row = block_row * 2 + local_row
                            # global_col depends on half
                            if half == 0:
                                global_col = block_col * 2 + local_col
                            else:
                                global_col = 4 + block_col * 2 + local_col
                            V_idx = global_row * 8 + global_col
                            V[V_idx][col] += coeff_B * v2[t][r]
                            
                    # Add to W
                    # W index: global row, global col
                    # W is 8x4.
                    # The output C is 4x4 for this half.
                    # The 2x2 block of C is at (block_row, block_col)
                    # local_row, local_col from w2[t][r]
                    # global_row_C = block_row * 2 + local_row
                    # global_col_C = block_col * 2 + local_col
                    # W index: W[k*n+i] -> row is k (0..7), col is i (0..3)
                    # Wait, W shape is (p*n, rank) = (32, rank).
                    # W[k*n+i, r]
                    # k is 0..7, i is 0..3.
                    # For left half, k is 0..3. For right half, k is 4..7.
                    # So k = half * 4 + global_row_C?
                    # No, C is 4x8.
                    # C_left is 4x4 (cols 0-3). C_right is 4x4 (cols 4-7).
                    # So for half 0, k is 0..3. For half 1, k is 4..7.
                    # i is 0..3.
                    
                    for b in range(4):
                        if b in cB: # Wait, W depends on C, which is sum of products.
                            # The w2[t] vector gives coefficients for the 4 elements of the 2x2 C block.
                            # The C block is at (block_row, block_col).
                            # We don't iterate over B blocks for W.
                            # We iterate over the 4 elements of the C block.
                            pass # Logic below is better

                # Correct W logic:
                # The term contributes to C via w2[t].
                # w2[t] is a vector of 4 coefficients for the 2x2 C block.
                # The C block is determined by the Strassen term s?
                # No, the Strassen term s computes a 2x2 matrix M_s.
                # M_s contributes to the final C blocks via the Strassen reconstruction.
                # My previous derivation of w2 was for the 2x2 case.
                # For the 4x4 case, the "w" vector for the 4x4 Strassen term s is not just w2[t].
                # The 4x4 Strassen formula is:
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                # Here M1..M7 are 2x2 matrices.
                # So for a fixed s, the contribution to the C blocks is:
                # If s=0 (M1): C00 += M1, C11 += M1
                # If s=1 (M2): C10 += M2, C11 -= M2
                # etc.
                
                # So for each s, we have a mapping from the 2x2 C blocks to the 4x4 C blocks.
                # Let's define this mapping for each s.
                
                col += 1

# The logic above was getting mixed up. Let's restart the W construction clearly.

def construct():
    n, m, p = DIMENSIONS
    rank = 98
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen vectors
    u2 = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    v2 = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, -1],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    
    # 4x4 Strassen A coefficients
    sA = [
        {0: 1, 3: 1}, {2: 1, 3: 1}, {0: 1}, {3: 1},
        {0: 1, 1: 1}, {0: -1, 2: 1}, {1: 1, 3: -1}
    ]
    
    # 4x4 Strassen B coefficients (same for left and right, just shifted cols)
    sB = [
        {0: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}, {2: 1, 0: -1},
        {3: 1}, {0: 1, 1: 1}, {2: 1, 3: 1}
    ]
    
    # 4x4 Strassen C mapping: for each s, which C blocks get which coefficient
    # C blocks: 00, 01, 10, 11
    sC = [
        {0: 1, 3: 1},   # M1 -> C00, C11
        {2: 1, 3: -1},  # M2 -> C10, C11
        {1: 1, 3: 1},   # M3 -> C01, C11
        {0: 1, 2: 1},   # M4 -> C00, C10
        {0: -1, 1: 1},  # M5 -> C00, C01
        {3: 1},         # M6 -> C11
        {0: 1}          # M7 -> C00
    ]
    
    col = 0
    for half in range(2):
        for s in range(7):
            cA = sA[s]
            cB = sB[s]
            cC = sC[s]
            
            for t in range(7):
                # For each element r in the 2x2 matrix
                for r in range(4):
                    # U part
                    for b in range(4):
                        if b in cA:
                            coeff = cA[b] * u2[t][r]
                            br, bc = b // 2, b % 2
                            lr, lc = r // 2, r % 2
                            gr, gc = br * 2 + lr, bc * 2 + lc
                            U[gr * 4 + gc][col] += coeff
                            
                    # V part
                    for b in range(4):
                        if b in cB:
                            coeff = cB[b] * v2[t][r]
                            br, bc = b // 2, b % 2
                            lr, lc = r // 2, r % 2
                            gr = br * 2 + lr
                            gc = bc * 2 + lc
                            if half == 1:
                                gc += 4
                            V[gr * 8 + gc][col] += coeff
                            
                    # W part
                    # The C block is at (br, bc) where br, bc come from the C mapping?
                    # No, the C mapping sC[s] tells us which C blocks are affected.
                    # But the 2x2 matrix M_s is placed in those blocks.
                    # So for each C block b_c in cC:
                    for b_c in range(4):
                        if b_c in cC:
                            coeff = cC[b_c] * w2[t][r]
                            # Wait, w2 is not defined yet in this scope, but I have it above.
                            # The r-th element of the 2x2 matrix M_s corresponds to
                            # local_row = r // 2, local_col = r % 2
                            # This element is added to the C block b_c.
                            # C block b_c is at global row br_c * 2 + lr, col bc_c * 2 + lc
                            br_c, bc_c = b_c // 2, b_c % 2
                            lr, lc = r // 2, r % 2
                            gr_c = br_c * 2 + lr
                            gc_c = bc_c * 2 + lc
                            # W index: k * n + i
                            # k is global row of C (0..3 for left, 4..7 for right)
                            # i is global col of C (0..3)
                            k = gr_c + (half * 4)
                            i = gc_c
                            W[k * 4 + i][col] += coeff
                            
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
