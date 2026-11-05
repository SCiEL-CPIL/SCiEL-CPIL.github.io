import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition for 3x4x8
    # Split p into 4+4
    # C = [C1, C2]
    # C1 = A B1, C2 = A B2
    
    # For each 3x4x4 part:
    # Split n into 1+2
    # C_top = A_top B_part (1x4x4)
    # C_bot = A_bot B_part (2x4x4)
    
    # 1x4x4: Rank 16. 
    # 2x4x4: Split m into 2+2.
    # A_bot = [A_bot1, A_bot2] (2x2 each)
    # B_part = [[B11, B12], [B21, B22]] (2x2 each)
    # C_bot1 = A_bot1 B11 + A_bot2 B21
    # C_bot2 = A_bot1 B12 + A_bot2 B22
    # This is 4 instances of 2x2x2.
    # Each 2x2x2 uses Strassen (Rank 7).
    # Total Rank for 2x4x4 is 4*7 = 28.
    
    # Total Rank for 3x4x4 is 16 + 28 = 44.
    # Total Rank for 3x4x8 is 44 + 44 = 88.
    
    rank = 88
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen decomposition
    # Indices:
    # A is 2x2, flattened row-major: a00, a01, a10, a11
    # B is 2x2, flattened row-major: b00, b01, b10, b11
    # C is 2x2, flattened row-major: c00, c01, c10, c11
    # But in our global tensors:
    # U index: i*m + j. Here i,j are local 0..1. Global offset needed.
    # V index: j*p + k. Here j,k are local 0..1. Global offset needed.
    # W index: k*n + i. Here k,i are local 0..1. Global offset needed.
    
    def add_strassen(offset_u_i, offset_u_j, offset_v_j, offset_v_k, offset_w_k, offset_w_i):
        # Strassen formulas
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to add 7 columns to U, V, W
        
        # M1
        # U: A00, A11 -> 1
        U[offset_u_i*2 + 0 + offset_u_j, col] = 1
        U[offset_u_i*2 + 1 + offset_u_j, col] = 1
        # V: B00, B11 -> 1
        V[offset_v_j*2 + 0 + offset_v_k, col] = 1
        V[offset_v_j*2 + 1 + offset_v_k, col] = 1
        # W: C00, C11 -> 1
        W[offset_w_k*2 + 0 + offset_w_i, col] = 1 # C00
        W[offset_w_k*2 + 1 + offset_w_i, col] = 1 # C11
        col += 1
        
        # M2
        # U: A10, A11 -> 1
        U[offset_u_i*2 + 1 + offset_u_j, col] = 1
        # V: B00 -> 1
        V[offset_v_j*2 + 0 + offset_v_k, col] = 1
        # W: C10 -> 1
        W[offset_w_k*2 + 1 + offset_w_i, col] = 1 # C10 (k=1, i=0? No. C is 2x2. C10 is row 1 col 0. W index k*n+i. k=1, i=0? 
        # Wait. W index is k*n + i.
        # C00: k=0, i=0.
        # C01: k=0, i=1? No.
        # Standard trace convention: W[k*n+i, r].
        # C[i,k] corresponds to W[k*n+i].
        # C00: i=0, k=0 -> W[0].
        # C01: i=0, k=1 -> W[1].
        # C10: i=1, k=0 -> W[n].
        # C11: i=1, k=1 -> W[n+1].
        
        # Let's re-verify W indices in add_strassen.
        # C00 (i=0, k=0): W[offset_w_k*2 + 0 + offset_w_i] -> W[0 + 0] = W[0]. Correct.
        # C11 (i=1, k=1): W[offset_w_k*2 + 1 + offset_w_i] -> W[2 + 1] = W[3]. Correct.
        # C10 (i=1, k=0): W[offset_w_k*2 + 0 + offset_w_i] -> W[0 + 1] = W[1]? No.
        # k=0, i=1. Index = 0*2 + 1 = 1.
        # My previous line: W[offset_w_k*2 + 1 + offset_w_i] was for C11 (k=1, i=1? No, k=1, i=1 is 1*2+1=3).
        # Let's map carefully.
        # C[i][k] -> W[k*n + i].
        # Local n=2.
        # C00: i=0, k=0 -> W[0].
        # C01: i=0, k=1 -> W[1].
        # C10: i=1, k=0 -> W[2].
        # C11: i=1, k=1 -> W[3].
        
        # M1 contributes to C00 and C11.
        # W[0] += 1, W[3] += 1.
        # M2 contributes to C10.
        # W[2] += 1.
        
        # I will rewrite the helper to be explicit.
        
        return col

    # This manual indexing is prone to error. I will use a more robust construction method.
    
    # I will construct the full 3x4x8 decomposition by iterating over the blocks.
    
    # Block 1: C1 (cols 0-3 of C)
    # A (3x4) * B1 (4x4)
    # Split A into A_top (1x4) and A_bot (2x4).
    
    # A_top * B1 (1x4x4)
    # This is 16 terms.
    for i in range(1): # i=0
        for j in range(4):
            for k in range(4):
                # U[i*4+j, col] = 1
                # V[j*8+k, col] = 1  (Note: B1 is cols 0-3 of B. So k is 0..3)
                # W[k*3+0, col] = 1   (Note: C1 is rows 0..2? No. C1 is the first 4 columns of C.
                # C is 3x8. C1 is C[:, 0:4].
                # So i is 0..2. k is 0..3.
                # W index: k*n + i.
                
                U[i*4 + j, col] = 1
                V[j*8 + k, col] = 1
                W[k*3 + i, col] = 1
                col += 1
                
    # A_bot * B1 (2x4x4)
    # A_bot is rows 1..2 of A. (i=1,2)
    # Split A_bot into A_bot1 (row 1, cols 0-1) and A_bot2 (row 1, cols 2-3)?
    # No, A_bot is 2x4. Split into two 2x2 blocks horizontally.
    # A_bot = [A_bot_left, A_bot_right]
    # B1 = [[B11, B12], [B21, B22]]
    # A_bot_left is 2x2. A_bot_right is 2x2.
    
    # We need 4 Strassen decompositions.
    # 1. A_bot_left * B11
    # 2. A_bot_right * B21
    # 3. A_bot_left * B12
    # 4. A_bot_right * B22
    
    # But Strassen combines them.
    # Actually, A_bot * B1 = A_bot_left B11 + A_bot_right B21 (for left half of C1)
    #                    + A_bot_left B12 + A_bot_right B22 (for right half of C1)
    
    # Let's just use 4 separate 2x2x2 Strassen decompositions.
    # This gives Rank 28.
    
    # Helper for 2x2x2 Strassen
    def add_strassen_222(u_i_start, u_j_start, v_j_start, v_k_start, w_k_start, w_i_start):
        nonlocal col
        # u_i_start: row offset in A (0..2)
        # u_j_start: col offset in A (0..3)
        # v_j_start: row offset in B (0..3)
        # v_k_start: col offset in B (0..7)
        # w_k_start: col offset in C (0..7)
        # w_i_start: row offset in C (0..2)
        
        # M1
        U[u_i_start*4 + u_j_start, col] = 1
        U[u_i_start*4 + u_j_start + 2, col] = 1
        V[v_j_start*8 + v_k_start, col] = 1
        V[v_j_start*8 + v_k_start + 2, col] = 1
        W[w_k_start*3 + w_i_start, col] = 1
        W[w_k_start*3 + w_i_start + 2, col] = 1
        col += 1
        
        # M2
        U[u_i_start*4 + u_j_start + 1, col] = 1 # A10? No.
        # A is 2x2.
        # A00: i=0, j=0.
        # A01: i=0, j=1.
        # A10: i=1, j=0.
        # A11: i=1, j=1.
        
        # M2: (A10 + A11) * B00
        U[u_i_start*4 + u_j_start + 1, col] = 1 # A10? No, u_i_start+1 is row 1. u_j_start is col 0. So A10.
        U[u_i_start*4 + u_j_start + 3, col] = 1 # A11.
        V[v_j_start*8 + v_k_start, col] = 1 # B00.
        W[w_k_start*3 + w_i_start + 1, col] = 1 # C10.
        col += 1
        
        # M3: A00 * (B01 - B11)
        U[u_i_start*4 + u_j_start, col] = 1 # A00.
        V[v_j_start*8 + v_k_start + 1, col] = 1 # B01.
        V[v_j_start*8 + v_k_start + 3, col] = -1 # B11.
        W[w_k_start*3 + w_i_start, col] = 1 # C00? No.
        # C01 = M3 + M5.
        # C11 = M1 - M2 + M3 + M6.
        # M3 contributes to C01 and C11.
        W[w_k_start*3 + w_i_start + 1, col] = 1 # C01? No.
        # C01: i=0, k=1. W[1*3+0] = W[3].
        # C11: i=1, k=1. W[1*3+1] = W[4].
        # My W indices:
        # w_k_start is k. w_i_start is i.
        # C00: k=0, i=0.
        # C01: k=1, i=0.
        # C10: k=0, i=1.
        # C11: k=1, i=1.
        
        # M3 contributes to C01 and C11.
        W[(w_k_start+1)*3 + w_i_start, col] = 1
        W[(w_k_start+1)*3 + w_i_start + 1, col] = 1
        col += 1
        
        # M4: A11 * (B10 - B00)
        U[u_i_start*4 + u_j_start + 3, col] = 1 # A11.
        V[v_j_start*8 + v_k_start + 2, col] = 1 # B10.
        V[v_j_start*8 + v_k_start, col] = -1 # B00.
        # M4 contributes to C00 and C10.
        W[w_k_start*3 + w_i_start, col] = 1 # C00.
        W[w_k_start*3 + w_i_start + 1, col] = 1 # C10.
        col += 1
        
        # M5: (A00 + A01) * B11
        U[u_i_start*4 + u_j_start, col] = 1 # A00.
        U[u_i_start*4 + u_j_start + 1, col] = 1 # A01.
        V[v_j_start*8 + v_k_start + 3, col] = 1 # B11.
        # M5 contributes to C01 and C11.
        W[(w_k_start+1)*3 + w_i_start, col] = 1 # C01.
        W[(w_k_start+1)*3 + w_i_start + 1, col] = 1 # C11.
        col += 1
        
        # M6: (A10 - A00) * (B00 + B01)
        U[u_i_start*4 + u_j_start + 1, col] = 1 # A10.
        U[u_i_start*4 + u_j_start, col] = -1 # A00.
        V[v_j_start*8 + v_k_start, col] = 1 # B00.
        V[v_j_start*8 + v_k_start + 1, col] = 1 # B01.
        # M6 contributes to C11.
        W[(w_k_start+1)*3 + w_i_start + 1, col] = 1 # C11.
        col += 1
        
        # M7: (A01 - A11) * (B10 + B11)
        U[u_i_start*4 + u_j_start + 1, col] = 1 # A01.
        U[u_i_start*4 + u_j_start + 3, col] = -1 # A11.
        V[v_j_start*8 + v_k_start + 2, col] = 1 # B10.
        V[v_j_start*8 + v_k_start + 3, col] = 1 # B11.
        # M7 contributes to C00 and C11.
        W[w_k_start*3 + w_i_start, col] = 1 # C00.
        W[(w_k_start+1)*3 + w_i_start + 1, col] = 1 # C11.
        col += 1

    # Now apply to A_bot * B1
    # A_bot rows: 1, 2. Cols: 0..3.
    # B1 rows: 0..3. Cols: 0..3.
    # C1 cols: 0..3.
    
    # A_bot_left: i=1,2; j=0,1.
    # A_bot_right: i=1,2; j=2,3.
    # B11: j=0,1; k=0,1.
    # B12: j=0,1; k=2,3.
    # B21: j=2,3; k=0,1.
    # B22: j=2,3; k=2,3.
    
    # 1. A_bot_left * B11 -> C1[:, 0:1] (part)
    add_strassen_222(1, 0, 0, 0, 0, 1)
    
    # 2. A_bot_right * B21 -> C1[:, 0:1] (part)
    add_strassen_222(1, 2, 2, 0, 0, 1)
    
    # 3. A_bot_left * B12 -> C1[:, 2:3] (part)
    add_strassen_222(1, 0, 0, 2, 2, 1)
    
    # 4. A_bot_right * B22 -> C1[:, 2:3] (part)
    add_strassen_222(1, 2, 2, 2, 2, 1)
    
    # Now C2 (cols 4-7 of C)
    # A * B2
    # A_top * B2 (1x4x4)
    for i in range(1):
        for j in range(4):
            for k in range(4):
                U[i*4 + j, col] = 1
                V[j*8 + 4 + k, col] = 1 # B2 is cols 4..7
                W[(4+k)*3 + i, col] = 1 # C2 is cols 4..7
                col += 1
                
    # A_bot * B2 (2x4x4)
    # B21: j=0,1; k=4,5.
    # B22: j=0,1; k=6,7.
    # B31: j=2,3; k=4,5.
    # B32: j=2,3; k=6,7.
    
    add_strassen_222(1, 0, 0, 4, 4, 1)
    add_strassen_222(1, 2, 2, 4, 4, 1)
    add_strassen_222(1, 0, 0, 6, 6, 1)
    add_strassen_222(1, 2, 2, 6, 6, 1)
    
    return {"U": U, "V": V, "W": W}
