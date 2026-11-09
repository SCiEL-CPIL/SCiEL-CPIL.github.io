import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We will construct a rank-45 decomposition using 2+2+1 partition.
    # U shape: (10, 45)
    # V shape: (25, 45)
    # W shape: (10, 45)
    
    rank = 45
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Partition indices
    # A rows: 0, 1
    # A cols: 0, 1 (block 1), 2, 3 (block 2), 4 (block 3)
    # B rows: 0, 1 (block 1), 2, 3 (block 2), 4 (block 3)
    # B cols: 0, 1 (block 1), 2, 3 (block 2), 4 (block 3)
    
    # Block definitions:
    # (A_block, B_row_block, B_col_block) -> C_block
    # A_blocks: 0 (rows 0-1, cols 0-1), 1 (rows 0-1, cols 2-3), 2 (rows 0-1, cols 4)
    # B_row_blocks: 0 (rows 0-1), 1 (rows 2-3), 2 (row 4)
    # B_col_blocks: 0 (cols 0-1), 1 (cols 2-3), 2 (col 4)
    
    # We need to map global indices to local indices for each sub-problem.
    
    def get_strassen_2x2x2(offset):
        """
        Returns U, V, W factors for 2x2x2 Strassen, placed at columns offset..offset+6.
        U: 4x7, V: 4x7, W: 4x7
        """
        # Strassen algorithm for 2x2
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We define U, V, W such that sum_c U[i,j] V[j,k] W[k,i] = C[i,k]
        # Note: The task convention is W[k*n+i, r].
        # For a 2x2 sub-block, local indices i, k in 0..1.
        # Global W index is k_global * n + i_global.
        
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # M1
        # U: A11, A22 -> 1
        # V: B11, B22 -> 1
        # W: C11, C22 -> 1
        u[0][0] = 1; u[3][0] = 1
        v[0][0] = 1; v[3][0] = 1
        w[0][0] = 1; w[3][0] = 1
        
        # M2
        # U: A21, A22 -> 1
        # V: B11 -> 1
        # W: C21 -> 1
        u[2][1] = 1; u[3][1] = 1
        v[0][1] = 1
        w[2][1] = 1
        
        # M3
        # U: A11 -> 1
        # V: B21 -> 1, B22 -> -1
        # W: C12 -> 1, C22 -> 1
        u[0][2] = 1
        v[2][2] = 1; v[3][2] = -1
        w[1][2] = 1; w[3][2] = 1
        
        # M4
        # U: A22 -> 1
        # V: B12 -> 1, B11 -> -1
        # W: C11 -> 1, C21 -> 1
        u[3][3] = 1
        v[1][3] = 1; v[0][3] = -1
        w[0][3] = 1; w[2][3] = 1
        
        # M5
        # U: A11, A12 -> 1
        # V: B22 -> 1
        # W: C11 -> -1, C12 -> 1
        u[0][4] = 1; u[1][4] = 1
        v[3][4] = 1
        w[0][4] = -1; w[1][4] = 1
        
        # M6
        # U: A21, A11 -> 1, -1
        # V: B11, B12 -> 1
        # W: C22 -> 1
        u[2][5] = 1; u[0][5] = -1
        v[0][5] = 1; v[1][5] = 1
        w[3][5] = 1
        
        # M7
        # U: A12, A22 -> 1, -1
        # V: B21, B22 -> 1
        # W: C11 -> 1
        u[1][6] = 1; u[3][6] = -1
        v[2][6] = 1; v[3][6] = 1
        w[0][6] = 1
        
        return u, v, w

    def get_rank4_2x1x2(offset):
        """
        2x1x2 multiplication. Rank 4.
        A is 2x1, B is 1x2. C is 2x2.
        C11 = A1 B1
        C12 = A1 B2
        C21 = A2 B1
        C22 = A2 B2
        """
        u = [[0]*4 for _ in range(2)]
        v = [[0]*4 for _ in range(1)]
        w = [[0]*4 for _ in range(4)]
        
        # Term 1: A1 B1
        u[0][0] = 1
        v[0][0] = 1
        w[0][0] = 1 # C11
        
        # Term 2: A1 B2
        u[0][1] = 1
        v[0][1] = 1
        w[1][1] = 1 # C12
        
        # Term 3: A2 B1
        u[1][2] = 1
        v[0][2] = 1
        w[2][2] = 1 # C21
        
        # Term 4: A2 B2
        u[1][3] = 1
        v[0][3] = 1
        w[3][3] = 1 # C22
        
        return u, v, w

    def get_rank4_1x2x2(offset):
        """
        1x2x2 multiplication. Rank 4.
        A is 1x2, B is 2x2. C is 1x2.
        C1 = A1 B11 + A2 B21
        C2 = A1 B12 + A2 B22
        """
        u = [[0]*4 for _ in range(2)]
        v = [[0]*4 for _ in range(4)]
        w = [[0]*4 for _ in range(2)]
        
        # Term 1: A1 B11
        u[0][0] = 1
        v[0][0] = 1
        w[0][0] = 1 # C1
        
        # Term 2: A2 B21
        u[1][1] = 1
        v[2][1] = 1
        w[0][1] = 1 # C1
        
        # Term 3: A1 B12
        u[0][2] = 1
        v[1][2] = 1
        w[1][2] = 1 # C2
        
        # Term 4: A2 B22
        u[1][3] = 1
        v[3][3] = 1
        w[1][3] = 1 # C2
        
        return u, v, w

    def get_rank4_2x2x1(offset):
        """
        2x2x1 multiplication. Rank 4.
        A is 2x2, B is 2x1. C is 2x1.
        C1 = A11 B1 + A12 B2
        C2 = A21 B1 + A22 B2
        """
        u = [[0]*4 for _ in range(4)]
        v = [[0]*4 for _ in range(2)]
        w = [[0]*4 for _ in range(2)]
        
        # Term 1: A11 B1
        u[0][0] = 1
        v[0][0] = 1
        w[0][0] = 1 # C1
        
        # Term 2: A12 B2
        u[1][1] = 1
        v[1][1] = 1
        w[0][1] = 1 # C1
        
        # Term 3: A21 B1
        u[2][2] = 1
        v[0][2] = 1
        w[1][2] = 1 # C2
        
        # Term 4: A22 B2
        u[3][3] = 1
        v[1][3] = 1
        w[1][3] = 1 # C2
        
        return u, v, w

    def get_rank1_1x1x1(offset):
        u = [[1]]
        v = [[1]]
        w = [[1]]
        return u, v, w

    # Mapping of global indices to local indices
    # A global row i (0..1), col j (0..4)
    # B global row j (0..4), col k (0..4)
    # C global row i (0..1), col k (0..4)
    
    # We iterate over the 3x3 grid of blocks
    # A_block_idx: 0 (cols 0-1), 1 (cols 2-3), 2 (col 4)
    # B_row_block_idx: 0 (rows 0-1), 1 (rows 2-3), 2 (row 4)
    # B_col_block_idx: 0 (cols 0-1), 1 (cols 2-3), 2 (col 4)
    
    current_rank = 0
    
    for ab in range(3):
        for brb in range(3):
            for bcb in range(3):
                # Determine dimensions
                # A block: 2 rows x (2 or 1) cols
                # B block: (2 or 1) rows x (2 or 1) cols
                # C block: 2 rows x (2 or 1) cols
                
                a_rows = 2
                a_cols = 2 if ab < 2 else 1
                b_rows = 2 if brb < 2 else 1
                b_cols = 2 if bcb < 2 else 1
                c_rows = 2
                c_cols = 2 if bcb < 2 else 1
                
                # Get factors
                if a_rows == 2 and a_cols == 2 and b_rows == 2 and b_cols == 2:
                    # 2x2x2
                    u_loc, v_loc, w_loc = get_strassen_2x2x2(current_rank)
                    r_sub = 7
                elif a_rows == 2 and a_cols == 1 and b_rows == 2 and b_cols == 2:
                    # 2x1x2? No, A is 2x1, B is 2x2.
                    # This is 2x1x2? No, middle dim is 1.
                    # A (2x1) * B (1x2) is 2x1x2.
                    # Here B is 2x2. So middle dim is 2?
                    # Wait, the partition is on the *multiplication* index.
                    # A cols = B rows.
                    # If ab=2 (1 col), then B row block must be size 1.
                    # So brb must be 2.
                    # So this case (ab=2, brb!=2) is invalid.
                    # My loop allows invalid combinations.
                    # I should only iterate valid combinations.
                    # ab=0 (2 cols) -> brb=0 (2 rows) or brb=1 (2 rows)
                    # ab=1 (2 cols) -> brb=0 or brb=1
                    # ab=2 (1 col) -> brb=2 (1 row)
                    
                    # So I should skip invalid ones.
                    if (a_cols != b_rows):
                        continue
                    
                    # Valid combinations:
                    # ab=0, brb=0: 2x2x2
                    # ab=0, brb=1: 2x2x2
                    # ab=1, brb=0: 2x2x2
                    # ab=1, brb=1: 2x2x2
                    # ab=2, brb=2: 2x1x1? No, A is 2x1, B is 1x(2 or 1).
                    
                    pass
                
                # Let's restart the loop structure to be cleaner.
                pass

    # Re-implement with explicit valid blocks
    
    current_rank = 0
    
    # Block (0,0): A[0:2, 0:2] * B[0:2, :]
    # B cols 0:2 (2x2), 2:4 (2x2), 4 (2x1)
    
    # Sub-block (0,0,0): 2x2x2
    u, v, w = get_strassen_2x2x2(current_rank)
    # Map to global
    for i in range(2):
        for j in range(2):
            U[i*5 + j][current_rank + i*2 + j] = u[i*2+j] # Wait, U is (n*m, r). Index i*m+j.
            # U index: i*5 + j
            # Local u index: i*2 + j
            # This mapping is not 1-to-1 if we just copy.
            # We need to embed the local factor into the global factor.
            
    # This manual embedding is error-prone. Let's use a helper function.
    
    def embed(U_global, V_global, W_global, u_loc, v_loc, w_loc, 
              a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset,
              a_rows, a_cols, b_rows, b_cols, c_rows, c_cols,
              rank_offset):
        
        # U: (n*m, r)
        # Local U: (a_rows * a_cols, r_sub)
        for i in range(a_rows):
            for j in range(a_cols):
                U_global[(a_row_offset + i) * m + (a_col_offset + j)][rank_offset + i*a_cols + j] = u_loc[i*a_cols + j]
        
        # V: (m*p, r)
        # Local V: (b_rows * b_cols, r_sub)
        for j in range(b_rows):
            for k in range(b_cols):
                V_global[(b_row_offset + j) * p + (b_col_offset + k)][rank_offset + j*b_cols + k] = v_loc[j*b_cols + k]
        
        # W: (p*n, r)
        # Local W: (c_rows * c_cols, r_sub)
        # Global W index: k*n + i
        for i in range(c_rows):
            for k in range(c_cols):
                W_global[(c_col_offset + k) * n + (c_row_offset + i)][rank_offset + i*c_cols + k] = w_loc[i*c_cols + k]

    # Reset U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Block 0: A cols 0-1
    # Block 1: A cols 2-3
    # Block 2: A col 4
    
    # B rows 0-1, 2-3, 4
    # B cols 0-1, 2-3, 4
    
    # Iterate
    for ab in range(3):
        for brb in range(3):
            for bcb in range(3):
                
                a_cols = 2 if ab < 2 else 1
                b_rows = 2 if brb < 2 else 1
                b_cols = 2 if bcb < 2 else 1
                
                if a_cols != b_rows:
                    continue
                
                a_row_offset = 0
                a_col_offset = 0 if ab == 0 else (2 if ab == 1 else 4)
                
                b_row_offset = 0 if brb == 0 else (2 if brb == 1 else 4)
                b_col_offset = 0 if bcb == 0 else (2 if bcb == 1 else 4)
                
                c_row_offset = 0
                c_col_offset = 0 if bcb == 0 else (2 if bcb == 1 else 4)
                
                a_rows = 2
                c_rows = 2
                c_cols = b_cols
                
                if a_rows == 2 and a_cols == 2 and b_rows == 2 and b_cols == 2:
                    u_loc, v_loc, w_loc = get_strassen_2x2x2(0)
                    r_sub = 7
                elif a_rows == 2 and a_cols == 1 and b_rows == 1 and b_cols == 2:
                    # 2x1x2
                    u_loc, v_loc, w_loc = get_rank4_2x1x2(0)
                    r_sub = 4
                elif a_rows == 2 and a_cols == 1 and b_rows == 1 and b_cols == 1:
                    # 2x1x1
                    # Rank 2
                    u_loc = [[1, 0], [0, 1]]
                    v_loc = [[1], [1]]
                    w_loc = [[1], [1]]
                    r_sub = 2
                else:
                    continue
                
                embed(U, V, W, u_loc, v_loc, w_loc,
                      a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset,
                      a_rows, a_cols, b_rows, b_cols, c_rows, c_cols,
                      current_rank)
                
                current_rank += r_sub

    return {"U": U, "V": V, "W": W}
