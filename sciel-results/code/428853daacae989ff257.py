import numpy as np

def construct():
    # Dimensions
    n, m, p = 5, 5, 5
    
    # We will build the factors for the 5x5x5 tensor decomposition
    # using the block decomposition described above.
    
    # Helper to create zero matrices
    def zero_mat(rows, cols):
        return [[0] * cols for _ in range(rows)]
    
    # We will accumulate U, V, W
    # Total rank is 110
    total_rank = 110
    U = zero_mat(25, total_rank)
    V = zero_mat(25, total_rank)
    W = zero_mat(25, total_rank)
    
    current_rank = 0
    
    # 1. C_44 = A_44 B_44 (Strassen M4, Rank 49)
    # We need factors for M4 using Strassen.
    # M4 is composed of 7 M2 multiplications.
    # M2 has rank 7.
    # So M4 has rank 7*7 = 49.
    
    # First, define M2 Strassen factors.
    # Indices for 2x2 matrix: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # U2 is 4x7, V2 is 4x7, W2 is 4x7
    
    U2 = zero_mat(4, 7)
    V2 = zero_mat(4, 7)
    W2 = zero_mat(4, 7)
    
    # M1 = (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1 # C00 gets M1
    
    # M2 = (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1 # C10 gets M2
    W2[3][1] = -1 # C11 gets -M2
    
    # M3 = A00(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1 # C01 gets M3
    W2[3][2] = 1 # C11 gets M3
    
    # M4 = A11(B10-B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1 # C00 gets M4
    W2[2][3] = 1 # C10 gets M4
    
    # M5 = (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1 # C00 gets -M5
    W2[1][4] = 1 # C01 gets M5
    
    # M6 = (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1 # C11 gets M6
    
    # M7 = (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1 # C00 gets M7
    
    # Now build M4 factors from M2 factors.
    # M4 is 4x4. Blocks are 2x2.
    # Block indices: 00, 01, 10, 11.
    # Each block is 2x2.
    # We iterate over the 7 Strassen steps for M4.
    # For each step, we use the M2 factors.
    
    # U4 will be 16x49.
    # The rows of U4 correspond to the 16 elements of A (4x4).
    # Row index in U4: i*4 + j, where i,j in 0..3.
    # Block (bi, bj) contains rows (2*bi .. 2*bi+1, 2*bj .. 2*bj+1).
    
    U4 = zero_mat(16, 49)
    V4 = zero_mat(16, 49)
    W4 = zero_mat(16, 49)
    
    for step in range(7):
        # The U2, V2, W2 for this step define the linear combination
        # for the 4 elements of the 2x2 block.
        # We need to map these to the 16 elements of the 4x4 matrix.
        
        # For U4:
        # The coefficient for A[bi][bj] in step 'step' is given by U2[bi*2+bj][step] ?
        # No, U2 is 4x7. The rows of U2 correspond to the 4 elements of the 2x2 matrix.
        # Let's say the 2x2 matrix elements are indexed 0,1,2,3.
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1).
        # In the 4x4 matrix, the block (bi, bj) has elements:
        # (2*bi, 2*bj), (2*bi, 2*bj+1), (2*bi+1, 2*bj), (2*bi+1, 2*bj+1).
        # These correspond to local indices 0,1,2,3.
        
        for bi in range(2):
            for bj in range(2):
                # For each block (bi, bj), the contribution to U4 is
                # U2[local_idx][step] for the corresponding row.
                for local_idx in range(4):
                    val = U2[local_idx][step]
                    if val != 0:
                        # Global row index in 4x4 matrix
                        gi = 2*bi + (local_idx // 2)
                        gj = 2*bj + (local_idx % 2)
                        U4[gi*4 + gj][step*7 + (local_idx // 2)*2 + (local_idx % 2)] = val
                        # Wait, the column index in U4 for M4 should be step*7 + local_idx?
                        # Yes, because each of the 7 steps involves a full 2x2 multiplication (rank 7).
                        # So the columns are grouped by step.
                        # Actually, the rank of M4 is 49.
                        # The columns 0..6 correspond to step 0, 7..13 to step 1, etc.
                        # Within a step, the 7 columns correspond to the 7 Strassen multiplications of the 2x2 blocks.
                        # So for step s, and local Strassen index l (0..6), the column is s*7 + l.
                        # But U2 has 7 columns. So for a given step s, we use U2[:, l] for l in 0..6?
                        # No, U2 is the factor for ONE 2x2 multiplication.
                        # So for step s, we perform 7 multiplications of 2x2 matrices?
                        # No, Strassen for M4 uses 7 multiplications of 2x2 MATRICES.
                        # Each 2x2 matrix multiplication has rank 7.
                        # So total rank is 7*7=49.
                        # The structure is:
                        # For each of the 7 high-level Strassen steps (s=0..6):
                        #   We compute a 2x2 matrix product.
                        #   This product is decomposed into 7 scalar products (low-level Strassen).
                        #   So for step s, we have 7 columns in U4/V4/W4.
                        #   The values in these columns come from U2/V2/W2.
                        
                        # Let's refine the mapping.
                        # High-level step s.
                        # Low-level step l (0..6).
                        # Column index = s*7 + l.
                        # Row index for A: determined by bi, bj, and local_idx.
                        # U2[local_idx][l] is the coefficient.
                        
                        # So:
                        # U4[global_row][s*7 + l] = U2[local_idx][l]
                        # This is what I wrote above, but I need to be careful with the column index.
                        # My previous line: U4[gi*4 + gj][step*7 + (local_idx // 2)*2 + (local_idx % 2)] = val
                        # This is WRONG. The column index should be step*7 + l, where l is the column in U2.
                        # But I am iterating over local_idx (rows of U2).
                        # I need to iterate over l (columns of U2) as well?
                        # No, U2 is 4x7.
                        # For a fixed high-level step s, and a fixed low-level step l,
                        # the coefficient for A[global_row] is U2[local_idx][l].
                        # So for each s, l, and local_idx:
                        # U4[global_row][s*7 + l] += U2[local_idx][l]
                        
                        # Let's rewrite the loop correctly.
        
    # Let's restart the M4 construction with the correct logic.
    U4 = zero_mat(16, 49)
    V4 = zero_mat(16, 49)
    W4 = zero_mat(16, 49)
    
    for s in range(7): # High-level Strassen step
        for l in range(7): # Low-level Strassen step
            col_idx = s * 7 + l
            for bi in range(2):
                for bj in range(2):
                    for local_idx in range(4):
                        val_u = U2[local_idx][l]
                        val_v = V2[local_idx][l]
                        val_w = W2[local_idx][l]
                        
                        gi = 2*bi + (local_idx // 2)
                        gj = 2*bj + (local_idx % 2)
                        g_row = gi*4 + gj
                        
                        U4[g_row][col_idx] += val_u
                        V4[g_row][col_idx] += val_v
                        W4[g_row][col_idx] += val_w
                        
    # Now we have U4, V4, W4 for the A_44 B_44 term.
    # We need to add these to the main U, V, W.
    # A_44 occupies rows 0..3, cols 0..3 of A.
    # So U4 rows 0..15 correspond to A[0..3][0..3].
    # U main rows 0..15 correspond to A[0..3][0..3].
    
    for i in range(16):
        for j in range(49):
            U[i][current_rank + j] = U4[i][j]
            V[i][current_rank + j] = V4[i][j]
            # W4 corresponds to C_44, which is C[0..3][0..3].
            # W main rows 0..15 correspond to C[0..3][0..3].
            W[i][current_rank + j] = W4[i][j]
            
    current_rank += 49
    
    # 2. C_44 += A_41 B_14 (Rank 16)
    # A_41 is 4x1 (rows 0..3, col 4).
    # B_14 is 1x4 (row 4, cols 0..3).
    # Product is 4x4.
    # Rank 16.
    # U: A_41 elements. Rows 0..3, col 4. Indices 4, 9, 14, 19 in A (25 elements).
    # V: B_14 elements. Row 4, cols 0..3. Indices 20, 21, 22, 23 in B.
    # W: C_44 elements. Rows 0..3, cols 0..3. Indices 0..15 in C.
    
    for i in range(4):
        for j in range(4):
            col_idx = current_rank + i*4 + j
            # A[i][4]
            U[i*5 + 4][col_idx] = 1
            # B[4][j]
            V[4*5 + j][col_idx] = 1
            # C[i][j]
            W[i*5 + j][col_idx] = 1
            
    current_rank += 16
    
    # 3. C_41 = A_44 B_41 (Rank 16)
    # A_44 is 4x4. B_41 is 4x1.
    # Product is 4x1.
    # Rank 16.
    # For each col j in 0..3 of C_41 (which is C[i][4] for i=0..3):
    # C[i][4] = sum_k A[i][k] B[k][4]
    # This is 4 terms for each i. Total 16 terms.
    
    for i in range(4):
        for k in range(4):
            col_idx = current_rank + i*4 + k
            # A[i][k]
            U[i*5 + k][col_idx] = 1
            # B[k][4]
            V[k*5 + 4][col_idx] = 1
            # C[i][4]
            W[i*5 + 4][col_idx] = 1
            
    current_rank += 16
    
    # 4. C_41 += A_41 B_11 (Rank 4)
    # A_41 is 4x1. B_11 is 1x1.
    # Product is 4x1.
    # Rank 4.
    
    for i in range(4):
        col_idx = current_rank + i
        # A[i][4]
        U[i*5 + 4][col_idx] = 1
        # B[4][4]
        V[4*5 + 4][col_idx] = 1
        # C[i][4]
        W[i*5 + 4][col_idx] = 1
        
    current_rank += 4
    
    # 5. C_14 = A_14 B_44 (Rank 16)
    # A_14 is 1x4. B_44 is 4x4.
    # Product is 1x4.
    # Rank 16.
    
    for j in range(4):
        for k in range(4):
            col_idx = current_rank + j*4 + k
            # A[4][k]
            U[4*5 + k][col_idx] = 1
            # B[k][j]
            V[k*5 + j][col_idx] = 1
            # C[4][j]
            W[4*5 + j][col_idx] = 1
            
    current_rank += 16
    
    # 6. C_14 += A_11 B_14 (Rank 4)
    # A_11 is 1x1. B_14 is 1x4.
    # Product is 1x4.
    # Rank 4.
    
    for j in range(4):
        col_idx = current_rank + j
        # A[4][4]
        U[4*5 + 4][col_idx] = 1
        # B[4][j]
        V[4*5 + j][col_idx] = 1
        # C[4][j]
        W[4*5 + j][col_idx] = 1
        
    current_rank += 4
    
    # 7. C_11 = A_14 B_41 (Rank 4)
    # A_14 is 1x4. B_41 is 4x1.
    # Product is 1x1.
    # Rank 4.
    
    for k in range(4):
        col_idx = current_rank + k
        # A[4][k]
        U[4*5 + k][col_idx] = 1
        # B[k][4]
        V[k*5 + 4][col_idx] = 1
        # C[4][4]
        W[4*5 + 4][col_idx] = 1
        
    current_rank += 4
    
    # 8. C_11 += A_11 B_11 (Rank 1)
    # A_11 is 1x1. B_11 is 1x1.
    # Product is 1x1.
    # Rank 1.
    
    col_idx = current_rank
    # A[4][4]
    U[4*5 + 4][col_idx] = 1
    # B[4][4]
    V[4*5 + 4][col_idx] = 1
    # C[4][4]
    W[4*5 + 4][col_idx] = 1
    
    current_rank += 1
    
    return {"U": U, "V": V, "W": W}
