import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary with 'U', 'V', 'W' matrices representing the tensor decomposition.
    Uses a hybrid decomposition:
    - Split B (4x6) into B1 (4x4) and B2 (4x2).
    - Compute A*B1 using Strassen's algorithm (Rank 49).
    - Compute A*B2 using 4 independent Strassen 2x2x2 multiplications (Rank 28).
    Total Rank: 77.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS # 4, 4, 6
    
    # Helper: Generate Strassen 2x2x2 decomposition (Rank 7)
    def strassen_2x2():
        # U: 4x7, V: 4x7, W: 4x7
        # Indices: 0:A00, 1:A01, 2:A10, 3:A11
        #           0:B00, 1:B01, 2:B10, 3:B11
        #           0:C00, 1:C01, 2:C10, 3:C11
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # M3 = A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1
        
        # M4 = A11(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1
        W[2][3] = 1
        
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1
        W[1][4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        W[3][6] = 1
        
        return U, V, W

    # Helper: Generate 4x4x4 Strassen decomposition (Rank 49)
    def strassen_4x4():
        U_2, V_2, W_2 = strassen_2x2()
        rank = 49
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        
        # Block row indices for 4x4 matrix
        # Block 0 (00): 0,1,4,5
        # Block 1 (01): 2,3,6,7
        # Block 2 (10): 8,9,12,13
        # Block 3 (11): 10,11,14,15
        block_rows = [
            [0, 1, 4, 5],
            [2, 3, 6, 7],
            [8, 9, 12, 13],
            [10, 11, 14, 15]
        ]
        
        # Define the 7 terms of Strassen
        # Each term is a list of (block_id, local_row, coeff) for U, V, W
        # local_row is 0..3 in the 2x2 block matrix
        
        # Term 0: (A00+A11)(B00+B11)
        # U: A00 (block 0, row 0), A11 (block 3, row 3)
        # V: B00 (block 0, row 0), B11 (block 3, row 3)
        # W: C00 (block 0, row 0), C11 (block 3, row 3)
        terms = [
            # (U_blocks, V_blocks, W_blocks)
            # U_blocks: list of (block_id, local_row, coeff)
            # local_row corresponds to the 2x2 position in the block matrix
            # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            # But in strassen_2x2, U[0] is A00, U[1] is A01, U[2] is A10, U[3] is A11.
            # So local_row 0 -> A00 (block 0), local_row 1 -> A01 (block 1), etc.
            
            # Term 0
            ([(0, 0, 1), (3, 3, 1)], [(0, 0, 1), (3, 3, 1)], [(0, 0, 1), (3, 3, 1)]),
            # Term 1: (A10+A11)B00
            ([(2, 2, 1), (3, 3, 1)], [(0, 0, 1)], [(2, 2, 1)]),
            # Term 2: A00(B01-B11)
            ([(0, 0, 1)], [(1, 1, 1), (3, 3, -1)], [(1, 1, 1)]),
            # Term 3: A11(B10-B00)
            ([(3, 3, 1)], [(2, 2, 1), (0, 0, -1)], [(0, 0, 1), (2, 2, 1)]),
            # Term 4: (A00+A01)B11
            ([(0, 0, 1), (1, 1, 1)], [(3, 3, 1)], [(0, 0, -1), (1, 1, 1)]),
            # Term 5: (A10-A00)(B00+B01)
            ([(2, 2, 1), (0, 0, -1)], [(0, 0, 1), (1, 1, 1)], [(3, 3, 1)]),
            # Term 6: (A01-A11)(B10+B11)
            ([(1, 1, 1), (3, 3, -1)], [(2, 2, 1), (3, 3, 1)], [(0, 0, 1), (3, 3, 1)]),
        ]
        
        for k in range(7):
            u_blocks, v_blocks, w_blocks = terms[k]
            col_offset = k * 7
            
            # Fill U
            for b_id, l_row, coeff in u_blocks:
                # l_row is the index in the 2x2 block matrix (0..3)
                # This corresponds to the local row in U_2
                # We add coeff * U_2[l_row][j] to all 4 rows of block b_id
                for j in range(7):
                    val = coeff * U_2[l_row][j]
                    if val != 0:
                        for r in block_rows[b_id]:
                            U[r][col_offset + j] += val
            
            # Fill V
            for b_id, l_row, coeff in v_blocks:
                for j in range(7):
                    val = coeff * V_2[l_row][j]
                    if val != 0:
                        for r in block_rows[b_id]:
                            V[r][col_offset + j] += val
            
            # Fill W
            for b_id, l_row, coeff in w_blocks:
                for j in range(7):
                    val = coeff * W_2[l_row][j]
                    if val != 0:
                        for r in block_rows[b_id]:
                            W[r][col_offset + j] += val
        
        return U, V, W

    # Helper: Generate 4x4x2 decomposition (Rank 28)
    def strassen_4x4x2():
        U_2, V_2, W_2 = strassen_2x2()
        rank = 28
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(8)] # B is 4x2, so 8 rows
        W = [[0]*rank for _ in range(8)] # C is 4x2, so 8 rows
        
        # Block row indices for A (4x4)
        block_rows_A = [
            [0, 1, 4, 5],   # A00
            [2, 3, 6, 7],   # A01
            [8, 9, 12, 13], # A10
            [10, 11, 14, 15] # A11
        ]
        
        # Block row indices for B (4x2)
        # B is split into B00 (top 2 rows, 2 cols) and B01 (bottom 2 rows, 2 cols)
        # B00: rows 0,1; cols 0,1 -> indices 0,1,4,5
        # B01: rows 2,3; cols 0,1 -> indices 2,3,6,7
        block_rows_B = [
            [0, 1, 4, 5],   # B00
            [2, 3, 6, 7]    # B01
        ]
        
        # Block row indices for C (4x2)
        # C_top (rows 0,1): indices 0,1,4,5
        # C_bot (rows 2,3): indices 2,3,6,7
        block_rows_C = [
            [0, 1, 4, 5],   # C_top
            [2, 3, 6, 7]    # C_bot
        ]
        
        # 4 terms:
        # 1. A00 B00 -> C_top
        # 2. A01 B01 -> C_top
        # 3. A10 B00 -> C_bot
        # 4. A11 B01 -> C_bot
        
        terms = [
            (0, 0, 0), # A00, B00, C_top
            (1, 1, 0), # A01, B01, C_top
            (2, 0, 1), # A10, B00, C_bot
            (3, 1, 1), # A11, B01, C_bot
        ]
        
        for k, (a_block, b_block, c_block) in enumerate(terms):
            col_offset = k * 7
            
            # Fill U
            for j in range(7):
                val = U_2[0][j] # All 4 local rows of U_2 are used?
                # Wait, U_2 is 4x7.
                # We are multiplying A_block (2x2) by B_block (2x2).
                # So we use the full U_2, V_2, W_2.
                # The local rows 0..3 of U_2 correspond to the 4 elements of A_block.
                pass
            
            # Correct logic:
            for j in range(7):
                # U
                for l_row in range(4):
                    val = U_2[l_row][j]
                    if val != 0:
                        for r in block_rows_A[a_block]:
                            U[r][col_offset + j] += val
                
                # V
                for l_row in range(4):
                    val = V_2[l_row][j]
                    if val != 0:
                        for r in block_rows_B[b_block]:
                            V[r][col_offset + j] += val
                
                # W
                for l_row in range(4):
                    val = W_2[l_row][j]
                    if val != 0:
                        for r in block_rows_C[c_block]:
                            W[r][col_offset + j] += val
        
        return U, V, W

    # Combine
    U_44, V_44, W_44 = strassen_4x4()
    U_42, V_42, W_42 = strassen_4x4x2()
    
    total_rank = 49 + 28
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(24)]
    W = [[0]*total_rank for _ in range(24)]
    
    # Copy 4x4 part
    for i in range(16):
        for j in range(49):
            U[i][j] = U_44[i][j]
            V[i][j] = V_44[i][j]
            W[i][j] = W_44[i][j]
    
    # Copy 4x2 part
    for i in range(16):
        for j in range(28):
            U[i][49 + j] = U_42[i][j]
    
    for i in range(8):
        for j in range(28):
            V[16 + i][49 + j] = V_42[i][j]
            W[16 + i][49 + j] = W_42[i][j]
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
