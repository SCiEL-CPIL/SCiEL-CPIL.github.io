import numpy as np

def construct():
    n, m, p = 3, 4, 4
    
    def get_solution():
        rank = 44
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Local 3x2x2 block decomposition (rank 11)
        U_loc = np.zeros((6, 11))
        V_loc = np.zeros((4, 11))
        W_loc = np.zeros((6, 11))
        
        # Strassen part (cols 0-6) for the top 2x2 sub-blocks
        # Indices in local 3x2 A: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # Indices in local 2x2 B: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # Indices in local 3x2 C: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        
        # M1 = (A11+A22)(B11+B22)
        U_loc[0, 0] = 1; U_loc[3, 0] = 1
        V_loc[0, 0] = 1; V_loc[3, 0] = 1
        W_loc[0, 0] = 1; W_loc[3, 0] = 1
        
        # M2 = (A21+A22)(B11)
        U_loc[2, 1] = 1; U_loc[3, 1] = 1
        V_loc[0, 1] = 1
        W_loc[2, 1] = 1; W_loc[3, 1] = -1
        
        # M3 = (A11)(B21-B11)
        U_loc[0, 2] = 1
        V_loc[1, 2] = 1; V_loc[3, 2] = -1
        W_loc[1, 2] = 1; W_loc[3, 2] = 1
        
        # M4 = (A22)(B12-B22)
        U_loc[3, 3] = 1
        V_loc[2, 3] = 1; V_loc[0, 3] = -1
        W_loc[0, 3] = 1; W_loc[2, 3] = 1
        
        # M5 = (A11+A12)(B22)
        U_loc[0, 4] = 1; U_loc[1, 4] = 1
        V_loc[3, 4] = 1
        W_loc[0, 4] = -1; W_loc[1, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U_loc[2, 5] = 1; U_loc[0, 5] = -1
        V_loc[0, 5] = 1; V_loc[1, 5] = 1
        W_loc[3, 5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U_loc[1, 6] = 1; U_loc[3, 6] = -1
        V_loc[2, 6] = 1; V_loc[3, 6] = 1
        W_loc[0, 6] = 1
        
        # Extra part (cols 7-10) for the 3rd row of A
        # C20 = A20 * B00 + A21 * B10  (Wait, row index 2 in local A is global row 2? No.)
        # Local A row 2 is the 3rd row of the 3x2 block.
        # Global A row 2 corresponds to the 3rd row of the 3x4 matrix.
        # The block is 3x2. So local row 2 is global row 2.
        # We need to compute C[2, :] = A[2, :] * B[:, :].
        # A[2, 0] * B[0, 0] + A[2, 1] * B[1, 0] -> C[2, 0]
        # A[2, 0] * B[0, 1] + A[2, 1] * B[1, 1] -> C[2, 1]
        
        # But wait, the Strassen part above only used local rows 0 and 1 of A (indices 0,1,2,3).
        # It did NOT use local row 2 (indices 4,5).
        # So we must add terms for local row 2.
        
        # Term for A[2,0] * B[0,0] -> C[2,0]
        # Local A idx 4, Local B idx 0, Local C idx 2 (row 1, col 0)
        U_loc[4, 7] = 1; V_loc[0, 7] = 1; W_loc[2, 7] = 1
        
        # Term for A[2,1] * B[1,0] -> C[2,0]
        # Local A idx 5, Local B idx 2, Local C idx 2
        U_loc[5, 8] = 1; V_loc[2, 8] = 1; W_loc[2, 8] = 1
        
        # Term for A[2,0] * B[0,1] -> C[2,1]
        # Local A idx 4, Local B idx 1, Local C idx 3 (row 1, col 1)
        U_loc[4, 9] = 1; V_loc[1, 9] = 1; W_loc[3, 9] = 1
        
        # Term for A[2,1] * B[1,1] -> C[2,1]
        # Local A idx 5, Local B idx 3, Local C idx 3
        U_loc[5, 10] = 1; V_loc[3, 10] = 1; W_loc[3, 10] = 1
        
        def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
            col_offset = block_idx * 11
            for r in range(3):
                for c in range(2):
                    local_a_idx = r * 2 + c
                    global_a_idx = (a_row_start + r) * m + (a_col_start + c)
                    U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
            for r in range(2):
                for c in range(2):
                    local_b_idx = r * 2 + c
                    global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                    V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
            for r in range(3):
                for c in range(2):
                    local_c_idx = r * 2 + c
                    global_c_idx = (c_col_start + c) * n + (c_row_start + r)
                    W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]

        # Tile the 3x4 by 4x4 matrix into four 3x2 by 2x2 blocks
        # Block 0: A[0:3, 0:2] * B[0:2, 0:2] -> C[0:3, 0:2]
        add_block(0, 0, 0, 0, 0, 0, 0)
        # Block 1: A[0:3, 2:4] * B[0:2, 2:4] -> C[0:3, 2:4]
        add_block(1, 0, 2, 0, 2, 0, 2)
        # Block 2: A[0:3, 0:2] * B[2:4, 0:2] -> C[0:3, 0:2] (Wait, B rows 2:4, C rows 0:3? No.)
        # C[i, k] = sum_j A[i, j] B[j, k]
        # Block 2: A[0:3, 0:2] * B[2:4, 0:2] -> C[0:3, 0:2] is wrong.
        # It contributes to C[0:3, 0:2] but B rows are 2:4.
        # So C[0:3, 0:2] gets contribution from A[0:3, 0:2] * B[0:2, 0:2] (Block 0)
        # and A[0:3, 2:4] * B[2:4, 0:2]? No.
        # Let's stick to the standard tiling:
        # A is 3x4, B is 4x4.
        # Split A into left 3x2 and right 3x2.
        # Split B into top 2x4 and bottom 2x4.
        # C = A_left * B_top + A_right * B_bottom.
        # This is 2 blocks of 3x2x4? No, the local block is 3x2x2.
        # So we split B_top into 2x2 and 2x2.
        # B_bottom into 2x2 and 2x2.
        
        # Block 0: A[0:3, 0:2] * B[0:2, 0:2] -> C[0:3, 0:2]
        add_block(0, 0, 0, 0, 0, 0, 0)
        
        # Block 1: A[0:3, 0:2] * B[0:2, 2:4] -> C[0:3, 2:4]
        add_block(1, 0, 0, 0, 2, 0, 2)
        
        # Block 2: A[0:3, 2:4] * B[2:4, 0:2] -> C[0:3, 0:2]
        add_block(2, 0, 2, 2, 0, 0, 0)
        
        # Block 3: A[0:3, 2:4] * B[2:4, 2:4] -> C[0:3, 2:4]
        add_block(3, 0, 2, 2, 2, 0, 2)
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return get_solution()
