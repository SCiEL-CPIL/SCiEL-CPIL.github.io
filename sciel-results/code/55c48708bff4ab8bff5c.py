import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We'll use a modified Strassen approach for 2x4x4
    # Blocks:
    # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 1: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Block 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    # Use 7 Strassen multiplications for blocks 0,1,2 (21 total)
    # Use 6 multiplications for block 3, sharing 1 with block 1 or 2
    
    # Helper to add a Strassen block with given base column offset
    def add_strassen_block(block_id, A_rows, A_cols, B_rows, B_cols, C_rows, C_cols, base_col):
        # M1: (X11+X22)(Y11+Y22)
        col = base_col + 0
        U[A_rows[0] * m + A_cols[0]][col] += 1
        U[A_rows[1] * m + A_cols[1]][col] += 1
        V[B_rows[0] * p + B_cols[0]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] += 1
        
        # M2: (X21+X22)Y11
        col = base_col + 1
        U[A_rows[1] * m + A_cols[0]][col] += 1
        U[A_rows[1] * m + A_cols[1]][col] += 1
        V[B_rows[0] * p + B_cols[0]][col] += 1
        
        # M3: X11(Y12-Y22)
        col = base_col + 2
        U[A_rows[0] * m + A_cols[0]][col] += 1
        V[B_rows[0] * p + B_cols[1]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] -= 1
        
        # M4: X22(Y21-Y11)
        col = base_col + 3
        U[A_rows[1] * m + A_cols[1]][col] += 1
        V[B_rows[1] * p + B_cols[0]][col] += 1
        V[B_rows[0] * p + B_cols[0]][col] -= 1
        
        # M5: (X11+X12)Y22
        col = base_col + 4
        U[A_rows[0] * m + A_cols[0]][col] += 1
        U[A_rows[0] * m + A_cols[1]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] += 1
        
        # M6: (X21-X11)(Y11+Y12)
        col = base_col + 5
        U[A_rows[1] * m + A_cols[0]][col] += 1
        U[A_rows[0] * m + A_cols[0]][col] -= 1
        V[B_rows[0] * p + B_cols[0]][col] += 1
        V[B_rows[0] * p + B_cols[1]][col] += 1
        
        # M7: (X12-X22)(Y21+Y22)
        col = base_col + 6
        U[A_rows[0] * m + A_cols[1]][col] += 1
        U[A_rows[1] * m + A_cols[1]][col] -= 1
        V[B_rows[1] * p + B_cols[0]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] += 1
        
        # Z11 = M1 + M4 - M5 + M7
        w_idx = C_cols[0] * n + C_rows[0]
        W[w_idx][base_col + 0] += 1
        W[w_idx][base_col + 3] += 1
        W[w_idx][base_col + 4] -= 1
        W[w_idx][base_col + 6] += 1
        
        # Z12 = M3 + M5
        w_idx = C_cols[1] * n + C_rows[0]
        W[w_idx][base_col + 2] += 1
        W[w_idx][base_col + 4] += 1
        
        # Z21 = M2 + M4
        w_idx = C_cols[0] * n + C_rows[1]
        W[w_idx][base_col + 1] += 1
        W[w_idx][base_col + 3] += 1
        
        # Z22 = M1 - M2 + M3 + M6
        w_idx = C_cols[1] * n + C_rows[1]
        W[w_idx][base_col + 0] += 1
        W[w_idx][base_col + 1] -= 1
        W[w_idx][base_col + 2] += 1
        W[w_idx][base_col + 5] += 1

    # Block 0: A_L * B_LL -> C_L (cols 0-6)
    add_strassen_block(0, [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], 0)
    
    # Block 1: A_R * B_RL -> C_L (cols 7-13)
    add_strassen_block(1, [0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1], 7)
    
    # Block 2: A_L * B_LR -> C_R (cols 14-20)
    add_strassen_block(2, [0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3], 14)
    
    # Block 3: A_R * B_RR -> C_R (cols 21-26, only 6 multiplications)
    # We need to find a 6-term decomposition for this block
    # that shares one term with block 1 or 2
    
    # For now, let's try a 6-term approach for block 3
    # We'll use a modified Strassen that omits one term and compensates
    # This is tricky; let me try a different approach
    
    # Actually, let me just use full Strassen for block 3 too but with rank 28
    # and see if we can trim one term
    
    # For block 3, use 6 terms by sharing M1 with block 1's M1
    # Block 3: A_R * B_RR -> C_R
    A_rows3 = [0, 1]
    A_cols3 = [2, 3]
    B_rows3 = [2, 3]
    B_cols3 = [2, 3]
    C_rows3 = [0, 1]
    C_cols3 = [2, 3]
    
    base3 = 21
    
    # Share M1 with block 1's M1 (col 7)
    # M1 for block 3 would be (A[0,2]+A[1,3])(B[2,2]+B[3,3])
    # Block 1's M1 is (A[0,2]+A[1,3])(B[2,0]+B[3,1])
    # These are different, so we can't directly share
    
    # Let me try a different sharing strategy
    # Share a term between block 2 and block 3
    
    # For now, let's just do full Strassen for block 3 (7 terms, cols 21-27)
    # But that gives rank 28. We need rank 27.
    
    # Alternative: use a known rank-27 identity for 2x4x4
    # Let me try to construct it from scratch
    
    # I'll use 7+7+7+6 = 27 by finding a valid 6-term decomposition for one block
    # combined with adjustments to other blocks
    
    # This is complex. Let me try a simpler approach:
    # Use the fact that some Strassen terms can be combined
    
    # For block 3, let's use 6 terms:
    # N1 = (A[0,2]+A[1,3])(B[2,2]+B[3,3])  -- same as M1
    # N2 = (A[1,2]+A[1,3])B[2,2]              -- same as M2
    # N3 = A[0,2](B[2,3]-B[3,3])               -- same as M3
    # N4 = A[1,3](B[3,2]-B[2,2])               -- same as M4
    # N5 = (A[0,2]+A[0,3])B[3,3]               -- same as M5
    # N6 = (A[1,2]-A[0,2])(B[2,2]+B[2,3])      -- same as M6
    # We omit M7: (A[0,3]-A[1,3])(B[3,2]+B[3,3])
    
    # But we need M7 for the output. Let's see if we can express M7 in terms of others
    # M7 = (A[0,3]-A[1,3])(B[3,2]+B[3,3])
    #     = A[0,3]*B[3,2] + A[0,3]*B[3,3] - A[1,3]*B[3,2] - A[1,3]*B[3,3]
    
    # From N3 = A[0,2]*B[2,3] - A[0,2]*B[3,3]
    # From N4 = A[1,3]*B[3,2] - A[1,3]*B[2,2]
    # From N5 = A[0,2]*B[3,3] + A[0,3]*B[3,3]
    # From N6 = A[1,2]*B[2,2] + A[1,2]*B[2,3] - A[0,2]*B[2,2] - A[0,2]*B[2,3]
    
    # This doesn't directly give us M7. The approach of omitting one term doesn't work simply.
    
    # Let me try a completely different strategy: use a known algebraic identity
    # for 2x4x4 matrix multiplication with rank 27
    
    # After research, I'll try to construct a valid rank-27 decomposition
    # by carefully choosing the 27 rank-1 tensors
    
    # For now, let me fall back to rank 28 with the current approach
    # and add the 7th term for block 3
    add_strassen_block(3, [0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3], 21)
    
    # This gives rank 28, not 27. Let me try to actually find rank 27.
    
    # Reset and try a proper rank-27 construction
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # I'll use a specific known decomposition for 2x4x4 with rank 27
    # Based on the algebraic structure, here's a candidate:
    
    # Use 7 Strassen for blocks 0, 1, 2 (21 terms)
    add_strassen_block(0, [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], 0)
    add_strassen_block(1, [0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1], 7)
    add_strassen_block(2, [0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3], 14)
    
    # For block 3, use 6 terms (cols 21-26)
    # We'll use a modified approach where we combine some terms
    
    A_r3, A_c3 = [0, 1], [2, 3]
    B_r3, B_c3 = [2, 3], [2, 3]
    C_r3, C_c3 = [0, 1], [2, 3]
    base3 = 21
    
    # Term 21: (A[0,2]+A[1,3])(B[2,2]+B[3,3])
    U[A_r3[0]*m + A_c3[0]][base3] += 1
    U[A_r3[1]*m + A_c3[1]][base3] += 1
    V[B_r3[0]*p + B_c3[0]][base3] += 1
    V[B_r3[1]*p + B_c3[1]][base3] += 1
    
    # Term 22: (A[1,2]+A[1,3])B[2,2]
    U[A_r3[1]*m + A_c3[0]][base3+1] += 1
    U[A_r3[1]*m + A_c3[1]][base3+1] += 1
    V[B_r3[0]*p + B_c3[0]][base3+1] += 1
    
    # Term 23: A[0,2](B[2,3]-B[3,3])
    U[A_r3[0]*m + A_c3[0]][base3+2] += 1
    V[B_r3[0]*p + B_c3[1]][base3+2] += 1
    V[B_r3[1]*p + B_c3[1]][base3+2] -= 1
    
    # Term 24: A[1,3](B[3,2]-B[2,2])
    U[A_r3[1]*m + A_c3[1]][base3+3] += 1
    V[B_r3[1]*p + B_c3[0]][base3+3] += 1
    V[B_r3[0]*p + B_c3[0]][base3+3] -= 1
    
    # Term 25: (A[0,2]+A[0,3])B[3,3]
    U[A_r3[0]*m + A_c3[0]][base3+4] += 1
    U[A_r3[0]*m + A_c3[1]][base3+4] += 1
    V[B_r3[1]*p + B_c3[1]][base3+4] += 1
    
    # Term 26: (A[1,2]-A[0,2])(B[2,2]+B[2,3])
    U[A_r3[1]*m + A_c3[0]][base3+5] += 1
    U[A_r3[0]*m + A_c3[0]][base3+5] -= 1
    V[B_r3[0]*p + B_c3[0]][base3+5] += 1
    V[B_r3[0]*p + B_c3[1]][base3+5] += 1
    
    # Now we need to compute C_R = A_R * B_RR using these 6 terms
    # C_R[0,0] = A[0,2]*B[2,2] + A[0,3]*B[3,2]
    # C_R[0,1] = A[0,2]*B[2,3] + A[0,3]*B[3,3]
    # C_R[1,0] = A[1,2]*B[2,2] + A[1,3]*B[3,2]
    # C_R[1,1] = A[1,2]*B[2,3] + A[1,3]*B[3,3]
    
    # Let's express these in terms of our 6 terms:
    # T21 = A[0,2]*B[2,2] + A[0,2]*B[3,3] + A[1,3]*B[2,2] + A[1,3]*B[3,3]
    # T22 = A[1,2]*B[2,2] + A[1,3]*B[2,2]
    # T23 = A[0,2]*B[2,3] - A[0,2]*B[3,3]
    # T24 = A[1,3]*B[3,2] - A[1,3]*B[2,2]
    # T25 = A[0,2]*B[3,3] + A[0,3]*B[3,3]
    # T26 = A[1,2]*B[2,2] + A[1,2]*B[2,3] - A[0,2]*B[2,2] - A[0,2]*B[2,3]
    
    # C_R[0,0] = A[0,2]*B[2,2] + A[0,3]*B[3,2]
    # We have A[0,2]*B[2,2] in T21 and T26, but A[0,3]*B[3,2] is not directly available
    # This shows that 6 terms are not sufficient for this block alone
    
    # The issue is that we can't decompose a 2x2x2 multiplication in 6 terms
    # So we need a different strategy
    
    # Let me try a global approach: instead of 4 independent blocks, 
    # use a joint decomposition that shares terms across blocks
    
    # This is getting complex. Let me try to verify if rank 27 is actually achievable
    # and if so, find a valid construction
    
    # For now, return the rank-28 solution as a fallback
    # The above code has already set up blocks 0,1,2 with 21 terms
    # and block 3 with 6 terms, but the 6-term decomposition for block 3 is incomplete
    
    # Let me just use full Strassen for all 4 blocks (rank 28)
    U = [[0] * 28 for _ in range(n * m)]
    V = [[0] * 28 for _ in range(m * p)]
    W = [[0] * 28 for _ in range(p * n)]
    
    add_strassen_block(0, [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], 0)
    add_strassen_block(1, [0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1], 7)
    add_strassen_block(2, [0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3], 14)
    add_strassen_block(3, [0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3], 21)
    
    return {"U": U, "V": V, "W": W}
