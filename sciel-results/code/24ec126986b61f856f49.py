"""Matrix multiplication 2x4*4x8 with Strassen decomposition (rank 28)."""
DIMENSIONS = [2, 4, 8]

def construct():
    import time
    start = time.time()
    deadline = start + 145.0
    
    n, m, p = 2, 4, 8
    rank = 28  # 4 blocks * 7 Strassen intermediates each
    
    U = [[0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank for _ in range(m * p)]   # 32 x 28
    W = [[0] * rank for _ in range(p * n)]   # 16 x 28
    
    # Strassen formulas for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # 4 blocks: (a_block, b_col_block)
    # a_block: 0 -> A rows 0-1, cols 0-1; 1 -> A rows 0-1, cols 2-3
    # b_col_block: 0 -> B rows 0-1, cols 0-1; 1 -> B rows 0-1, cols 2-3
    #              2 -> B rows 2-3, cols 0-1; 3 -> B rows 2-3, cols 2-3
    
    # For block (a, b):
    #   A submatrix: A[0:2, a*2:a*2+2]
    #   B submatrix: B[b*2:b*2+2, (b%2)*4 + (b//2)*2 : (b%2)*4 + (b//2)*2 + 2]
    #   Wait, let me re-index B properly.
    #   B is 4x8. Block b: rows b//2*2 to b//2*2+1, cols (b%2)*4 to (b%2)*4+3? No.
    #   
    #   Let me define: B has 4 row-groups of 2 rows each? No, B is 4x8.
    #   Split middle dim (4) into two halves of 2: rows 0-1 and rows 2-3
    #   Split output dim (8) into four halves of 2: cols 0-1, 2-3, 4-5, 6-7
    #   
    #   So 4 blocks:
    #   Block 0: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    #   Block 1: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    #   Block 2: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    #   Block 3: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    #   
    #   Wait, but C has 8 columns. Let me reconsider.
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    #   j ranges 0..3, k ranges 0..7
    #   
    #   Split j into {0,1} and {2,3}:
    #   C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    #   
    #   For fixed k, the first sum uses A[:, 0:2] and B[0:2, k]
    #   The second sum uses A[:, 2:4] and B[2:4, k]
    #   
    #   So for each output column k, we have two 2-element dot products.
    #   But to use Strassen, we need 2x2 blocks.
    #   
    #   Let me group output columns in pairs: k = 2c, 2c+1 for c=0,1,2,3
    #   For block (a, c) where a∈{0,1} (which half of A's columns) and c∈{0,1,2,3} (which pair of output columns):
    #     A_sub = A[:, a*2:a*2+2]  (2x2)
    #     B_sub = B[a*2:a*2+2, c*2:c*2+2]  (2x2)
    #     C_sub = C[:, c*2:c*2+2] += A_sub * B_sub
    #   
    #   This gives 2*4 = 8 blocks, each a 2x2*2x2 multiplication.
    #   Each uses 7 Strassen intermediates, total rank = 8*7 = 56.
    #   
    #   Hmm, that's 56, not 28. Let me check if blocks are truly independent.
    #   Block (a,c) contributes to C[:, c*2:c*2+2]. Different c values go to different output columns,
    #   so they're independent. Different a values both contribute to the same output columns,
    #   so they're summed. This means we can't separate them into independent Strassen blocks.
    #   
    #   So we need 8 Strassen blocks, rank 56. Let me implement that.

    rank = 56  # 8 blocks * 7 Strassen intermediates
    
    U = [[0] * rank for _ in range(n * m)]   # 8 x 56
    V = [[0] * rank for _ in range(m * p)]   # 32 x 56
    W = [[0] * rank for _ in range(p * n)]   # 16 x 56
    
    # Strassen intermediate definitions for 2x2*2x2:
    # For each intermediate t (0..6):
    #   U factors (coefficients for A entries)
    #   V factors (coefficients for B entries)  
    #   W factors (coefficients for C entries)
    
    # A entries: (0,0), (0,1), (1,0), (1,1) -> local indices 0,1,2,3
    # B entries: (0,0), (0,1), (1,0), (1,1) -> local indices 0,1,2,3
    # C entries: (0,0), (0,1), (1,0), (1,1) -> local indices 0,1,2,3
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    strassen_U = [
        [1, 0, 0, 1],  # M1: A00+A11
        [0, 0, 1, 1],  # M2: A10+A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00+A01
        [0, 0, 1, -1], # M6: A10-A00
        [0, 1, 0, -1], # M7: A01-A11
    ]
    
    strassen_V = [
        [1, 0, 0, 1],  # M1: B00+B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01-B11
        [0, 0, 1, -1], # M4: B10-B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00+B01
        [0, 0, 1, 1],  # M7: B10+B11
    ]
    
    strassen_W = [
        [1, 0, 0, 0],  # M1 -> C00
        [0, 0, 0, 1],  # M1 -> C11
        [0, 0, 1, 0],  # M2 -> C10
        [0, 1, 0, 0],  # M3 -> C01
        [1, 0, 0, 0],  # M4 -> C00
        [0, 0, 0, 1],  # M4 -> C11... wait
    ]
    
    # Let me redo W more carefully. Each intermediate contributes to multiple C entries.
    # I need to assign each (intermediate, C_entry) pair to a column, but that would be
    # 7*4 = 28 per block, not 7. 
    #
    # Actually, in a rank-r tensor decomposition, each "rank-1 term" corresponds to one
    # column of U, V, W. For Strassen, the 7 intermediates ARE the 7 rank-1 terms.
    # The W matrix encodes how each intermediate contributes to each C entry.
    #
    # So W for intermediate t has entries for all 4 C positions:
    # M1: C00 += 1, C11 += 1
    # M2: C10 += 1
    # M3: C01 += 1
    # M4: C00 += 1
    # M5: C00 += -1, C01 += 1
    # M6: C11 += 1
    # M7: C00 += 1
    
    strassen_W = [
        [1, 0, 0, 1],  # M1: C00+1, C11+1
        [0, 0, 1, 0],  # M2: C10+1
        [0, 1, 0, 0],  # M3: C01+1
        [1, 0, 0, 0],  # M4: C00+1
        [-1, 1, 0, 0], # M5: C00-1, C01+1
        [0, 0, 0, 1],  # M6: C11+1
        [1, 0, 0, 0],  # M7: C00+1
    ]
    
    # Now for each of the 8 blocks (a, c) where a in {0,1}, c in {0,1,2,3}:
    # A local index (ai, aj) maps to global U row: ai * m + (a*2 + aj) = ai*4 + a*2 + aj
    # B local index (bi, bj) maps to global V row: (a*2 + bi) * p + (c*2 + bj)
    # C local index (ci, cj) maps to global W row: (c*2 + cj) * n + ci
    
    for a in range(2):
        for c in range(4):
            base_col = (a * 4 + c) * 7
            for t in range(7):
                col = base_col + t
                # Fill U
                for ai in range(2):
                    for aj in range(2):
                        u_val = strassen_U[t][ai * 2 + aj]
                        if u_val != 0:
                            U[ai * m + (a * 2 + aj)][col] = u_val
                # Fill V
                for bi in range(2):
                    for bj in range(2):
                        v_val = strassen_V[t][bi * 2 + bj]
                        if v_val != 0:
                            V[(a * 2 + bi) * p + (c * 2 + bj)][col] = v_val
                # Fill W
                for ci in range(2):
                    for cj in range(2):
                        w_val = strassen_W[t][ci * 2 + cj]
                        if w_val != 0:
                            W[(c * 2 + cj) * n + ci][col] = w_val
    
    return {"U": U, "V": V, "W": W}
