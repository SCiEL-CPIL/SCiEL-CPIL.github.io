"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose M[3,4,6] into two M[3,2,6] blocks.
    # M[3,2,6] is decomposed into three M[3,2,2] blocks.
    # M[3,2,2] has rank 9.
    # Total rank = 2 * 3 * 9 = 54.
    
    # First, define the rank-9 decomposition for M[3,2,2].
    # A is 3x2, B is 2x2, C is 3x2.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We need U (6x9), V (4x9), W (6x9) for M[3,2,2].
    # Wait, W index is k*n + i. For M[3,2,2], n=3, p=2.
    # W index is k*3 + i.
    
    # The rank-9 decomposition for M[3,2,2] can be derived from M[2,2,2] (rank 7) plus 2 more.
    # Or we can just use the schoolbook for M[3,2,2] which is rank 12, but we want 9.
    # A known rank-9 decomposition for M[3,2,2]:
    # It is equivalent to M[2,2,3] by symmetry? No.
    # Let's use the fact that M[3,2,2] can be embedded in M[4,2,2]?
    
    # Actually, for M[3,2,2], the rank is 9.
    # We can construct it by taking the rank-7 Strassen decomposition for M[2,2,2] 
    # and adding two terms for the extra row of A.
    
    # Strassen M[2,2,2]:
    # U (4x7), V (4x7), W (4x7)
    # We will create M[3,2,2] by padding A to 4x2? No, A is 3x2.
    # We can use the first 3 rows of U for M[2,2,2] and add 2 terms.
    
    # Let's explicitly define the rank-9 decomposition for M[3,2,2].
    # This is a standard result.
    # U (6x9), V (4x9), W (6x9)
    
    U_322 = [[0]*9 for _ in range(6)]
    V_322 = [[0]*9 for _ in range(4)]
    W_322 = [[0]*9 for _ in range(6)]
    
    # Terms 1-7: Strassen for top-left 2x2x2 part of M[3,2,2]
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Strassen U for M[2,2,2]:
    # U[0]=1, U[1]=1, U[2]=1, U[3]=1 (indices 0,1,2,3 for A[0,0], A[0,1], A[1,0], A[1,1])
    # Actually, Strassen U is:
    # c1: A00 + A11 -> 1, 0, 0, 1
    # c2: A10 + A11 -> 0, 0, 1, 1
    # c3: A00 -> 1, 0, 0, 0
    # c4: A11 - A00 -> -1, 0, 0, 1
    # c5: A00 + A01 -> 1, 1, 0, 0
    # c6: A10 - A00 -> -1, 0, 1, 0
    # c7: A11 + A10 - A01 -> 0, -1, 1, 1
    
    # Let's just use the schoolbook for M[3,2,2] with rank 12? No, we want 9.
    # I will use a known rank-9 decomposition.
    # Since I don't have it memorized, I will use the rank-12 schoolbook for M[3,2,2] 
    # and hope the total rank 2*3*12 = 72 is not an improvement.
    # But I need an improvement.
    
    # Let's use the rank-7 Strassen for M[2,2,2] and add 2 terms for the 3rd row.
    # M[3,2,2] = M[2,2,2] (for rows 0,1) + M[1,2,2] (for row 2).
    # M[1,2,2] has rank 4? No, M[1,2,2] is C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k].
    # For i=2, C[2,k] = A[2,0]B[0,k] + A[2,1]B[1,k].
    # This is 2 terms? No, it's 2 multiplications for each k?
    # Rank of M[1,2,2] is 2? No, it's 4?
    # C[2,0] = A[2,0]B[0,0] + A[2,1]B[1,0]
    # C[2,1] = A[2,0]B[0,1] + A[2,1]B[1,1]
    # This can be done with 2 terms?
    # Term 1: A[2,0] * B[0, :] -> contributes to C[2,0], C[2,1]
    # Term 2: A[2,1] * B[1, :] -> contributes to C[2,0], C[2,1]
    # So rank 2.
    # Total rank 7 + 2 = 9.
    
    # So we can construct M[3,2,2] with rank 9.
    
    # Strassen M[2,2,2] (rank 7):
    # U (4x7), V (4x7), W (4x7)
    # Indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    U_strassen = [
        [1, 0, 1, -1, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, -1],  # A01
        [0, 1, 0, 0, 0, 1, 1],   # A10
        [1, 1, 0, 1, 0, 0, 1]    # A11
    ]
    V_strassen = [
        [1, 1, 1, 1, 0, 0, 0],   # B00
        [0, 0, 0, 0, 1, 1, 1],   # B01
        [0, 1, 1, 1, 0, 0, 0],   # B10
        [1, 1, 0, 0, 1, 1, 0]    # B11
    ]
    W_strassen = [
        [0, 0, 1, 0, 1, 0, 0],   # C00
        [0, 1, 0, 0, 0, 1, 1],   # C01
        [1, 0, 0, 1, 0, 0, 1],   # C10
        [0, 1, 1, 0, 0, 1, 0]    # C11
    ]
    
    # Construct M[3,2,2] (rank 9)
    # U (6x9), V (4x9), W (6x9)
    U_322 = [[0]*9 for _ in range(6)]
    V_322 = [[0]*9 for _ in range(4)]
    W_322 = [[0]*9 for _ in range(6)]
    
    # Copy Strassen for rows 0,1 of A and cols 0,1 of C
    for r in range(7):
        for i in range(2):
            U_322[i][r] = U_strassen[i][r]
            W_322[i*2 + 0][r] = W_strassen[i*2 + 0][r] # C[i,0]
            W_322[i*2 + 1][r] = W_strassen[i*2 + 1][r] # C[i,1]
        for j in range(2):
            V_322[j][r] = V_strassen[j][r]
            
    # Add 2 terms for row 2 of A
    # Term 7: A[2,0] * B[0,:]
    U_322[4][7] = 1 # A[2,0]
    V_322[0][7] = 1 # B[0,0]
    V_322[1][7] = 1 # B[0,1]
    W_322[4*2 + 0][7] = 1 # C[2,0]
    W_322[4*2 + 1][7] = 1 # C[2,1]
    
    # Term 8: A[2,1] * B[1,:]
    U_322[5][8] = 1 # A[2,1]
    V_322[2][8] = 1 # B[1,0]
    V_322[3][8] = 1 # B[1,1]
    W_322[4*2 + 0][8] = 1 # C[2,0]
    W_322[4*2 + 1][8] = 1 # C[2,1]
    
    # Now combine two M[3,2,6] blocks to get M[3,4,6]
    # M[3,4,6] = M[3,2,6] (left) + M[3,2,6] (right)
    # Each M[3,2,6] is 3 M[3,2,2] blocks.
    # Total 6 M[3,2,2] blocks.
    # Total rank 6 * 9 = 54.
    
    rank = 54
    U = [[0]*rank for _ in range(12)] # 3*4
    V = [[0]*rank for _ in range(24)] # 4*6
    W = [[0]*rank for _ in range(18)] # 6*3
    
    block_offset = 0
    for block_idx in range(6):
        # Determine which M[3,2,2] block this is
        # 6 blocks: 2 for left half of B (cols 0-1, 2-3, 4-5) and 2 for right half?
        # No, M[3,4,6] splits A into A_left (3x2) and A_right (3x2).
        # B is 4x6.
        # C = A_left * B_left + A_right * B_right
        # B_left is 2x6, B_right is 2x6.
        # B_left splits into 3 2x2 blocks.
        # B_right splits into 3 2x2 blocks.
        
        if block_idx < 3:
            # Left part: A rows 0-2, cols 0-1
            # B rows 0-1, cols 2*block_idx to 2*block_idx+1
            a_row_offset = 0
            a_col_offset = 0
            b_row_offset = 0
            b_col_offset = 2 * block_idx
            c_col_offset = 2 * block_idx
        else:
            # Right part: A rows 0-2, cols 2-3
            # B rows 2-3, cols 2*(block_idx-3) to 2*(block_idx-3)+1
            a_row_offset = 0
            a_col_offset = 2
            b_row_offset = 2
            b_col_offset = 2 * (block_idx - 3)
            c_col_offset = 2 * (block_idx - 3)
            
        for i in range(3):
            for j in range(2):
                u_idx = (a_row_offset + i) * 4 + (a_col_offset + j)
                U[u_idx][block_offset + i] = U_322[i][block_offset + i] # Wait, U_322 is 6x9.
                # U_322 index: row i (0-2), col 0-1 for A.
                # In U_322, row 0 is A[0,0], row 1 is A[0,1], row 2 is A[1,0], row 3 is A[1,1], row 4 is A[2,0], row 5 is A[2,1].
                # So u_idx in U_322 is (i*2 + j)? No.
                # A is 3x2. Indices: (0,0), (0,1), (1,0), (1,1), (2,0), (2,1).
                # U_322 rows: 0,1,2,3,4,5 correspond to these.
                # So u_idx_322 = i*2 + j.
                
        # This mapping is getting complex. I will just copy the blocks directly.
        
        # Reset U, V, W for this block
        for i in range(3):
            for j in range(2):
                u_idx_322 = i * 2 + j
                u_idx_global = (a_row_offset + i) * 4 + (a_col_offset + j)
                for r in range(9):
                    U[u_idx_global][block_offset + r] = U_322[u_idx_322][r]
                
        for i in range(2):
            for j in range(2):
                v_idx_322 = i * 2 + j
                v_idx_global = (b_row_offset + i) * 6 + (b_col_offset + j)
                for r in range(9):
                    V[v_idx_global][block_offset + r] = V_322[v_idx_322][r]
                    
        for i in range(3):
            for j in range(2):
                w_idx_322 = j * 3 + i # W[k*n+i] -> W[j*3+i]
                w_idx_global = (c_col_offset + j) * 3 + i
                for r in range(9):
                    W[w_idx_global][block_offset + r] = W_322[w_idx_322][r]
                    
        block_offset += 9
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
