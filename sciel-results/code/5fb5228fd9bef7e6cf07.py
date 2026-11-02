import time

DIMENSIONS = [3, 5, 6]

def construct():
    # Rank 78 decomposition using block decomposition with Strassen
    # Split A (3x5) into A1 (1x5) and A2 (2x5)
    # A1: rank 30 (schoolbook for 1x5x6)
    # A2: rank 48 (split B into three 5x2 blocks, each uses Strassen on 2x2x2 + schoolbook for 2x1x2)
    
    n, m, p = DIMENSIONS
    rank = 78
    U = [[0] * rank for _ in range(n * m)]  # 15 x 78
    V = [[0] * rank for _ in range(m * p)]  # 30 x 78
    W = [[0] * rank for _ in range(p * n)]  # 18 x 78
    
    col = 0
    
    # Part 1: A1 (1x5) * B (5x6) -> 1x6
    # Schoolbook: rank = 1*5*6 = 30
    for j in range(5):
        for k in range(6):
            U[0 * m + j][col] = 1  # row 0 of A, col j
            V[j * p + k][col] = 1  # row j of B, col k
            W[k * n + 0][col] = 1  # row k of C, col 0
            col += 1
    
    # Part 2: A2 (2x5) * B (5x6) -> 2x6
    # Split B into three 5x2 blocks: B1 (cols 0-1), B2 (cols 2-3), B3 (cols 4-5)
    for block_idx in range(3):
        # For each 2x5 * 5x2 -> 2x2 subproblem
        # Split the 5x2 block into (4x2) and (1x2)
        # 2x4*4x2: use Strassen on two 2x2 blocks -> rank 14
        # 2x1*1x2: schoolbook -> rank 2
        # Total per block: 16
        
        # Strassen part: 2x2 * 2x2 for the first 4 rows of B block
        # Split B_block (4x2) into B_top (2x2) and B_bot (2x2)
        # A2 (2x2 effective) * B_top (2x2) -> use Strassen (rank 7)
        # A2 (2x2 effective) * B_bot (2x2) -> use Strassen (rank 7)
        
        # Strassen for 2x2x2:
        # M1 = (A11-A22)(B21-B22)
        # M2 = (A11+A12)B22
        # M3 = A11(A22-A21) -- wait, standard Strassen:
        # M1 = (A11-A22)*(B21-B22)
        # M2 = (A11+A12)*B22
        # M3 = A11*(A22-A21) -- no
        # Standard Strassen for C = A*B where A,B are 2x2:
        # M1 = (A11 - A22) * (B21 - B22)
        # M2 = (A11 + A12) * B22
        # M3 = A11 * (A22 - A21) -- no, let me recall properly
        # M1 = (A11 - A22) * (B21 - B22)
        # M2 = (A11 + A12) * B22
        # M3 = A11 * (A22 - A21) -- incorrect
        # Correct Strassen:
        # M1 = (A11 - A22) * (B21 - B22)
        # M2 = (A11 + A12) * B22
        # M3 = A11 * (A22 - A21) -- still wrong
        # Let me use the standard formulation:
        # M1 = (A11 - A22) * (B21 - B22)
        # M2 = (A11 + A12) * B22
        # M3 = A11 * (A22 - A21) -- I'm confusing myself
        # Standard Strassen multiplies:
        # P1 = (A11 - A22) * (B21 - B22)
        # P2 = (A11 + A12) * B22
        # P3 = A11 * (A22 - A21) -- no
        # P3 = (A21 - A22) * (B11 + B12)
        # P4 = A22 * (B12 - B11)
        # P5 = (A11 + A22) * (B11 + B22)
        # P6 = (A12 - A22) * (B21 + B22)
        # P7 = (A11 - A21) * (B11 + B12)
        # C11 = P5 + P4 - P2 + P6
        # C12 = P1 + P2
        # C21 = P3 + P5
        # C22 = P1 - P3 + P4 + P7
        
        # For 2x2 * 2x2 with A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]:
        # In our case, A2 is 2x5, so the "2x2" part means we treat columns 0-1 of A2 as the A matrix
        # But A2 is 2x5, and B_block is 5x2. We need to decompose this 2x5 * 5x2.
        
        # Actually, let me reconsider. The subproblem is A2 (2x5) * B_block (5x2) -> 2x2
        # This is a 2x5x2 tensor. We can split the 5 into 4+1.
        # A2 * B_block = A2_top * B_block_top + A2_bot * B_block_bot
        # where A2_top is 2x4 (first 4 columns of A2), B_block_top is 4x2 (first 4 rows of B_block)
        # and A2_bot is 2x1 (last column of A2), B_block_bot is 1x2 (last row of B_block)
        
        # For 2x4 * 4x2: split 4 into 2+2
        # A2_top * B_block_top = A2_top_left * B_block_top_left + A2_top_right * B_block_top_right
        # where A2_top_left is 2x2 (cols 0-1 of A2), B_block_top_left is 2x2 (rows 0-1 of B_block)
        # A2_top_right is 2x2 (cols 2-3 of A2), B_block_top_right is 2x2 (rows 2-3 of B_block)
        
        # So we have two 2x2 * 2x2 subproblems, each with Strassen rank 7.
        
        # Strassen factors for 2x2 * 2x2:
        # For A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
        # C = A*B
        # P1 = (a11-a22)*(b21-b22)
        # P2 = (a11+a12)*b22
        # P3 = (a21-a22)*(b11+b12)
        # P4 = a22*(b12-b11)
        # P5 = (a11+a22)*(b11+b22)
        # P6 = (a12-a22)*(b21+b22)
        # P7 = (a11-a21)*(b11+b12)
        # c11 = p5+p4-p2+p6
        # c12 = p1+p2
        # c21 = p3+p5
        # c22 = p1-p3+p4+p7
        
        # The U, V, W entries for each Strassen term:
        # U has rows for A (2x2), V has rows for B (2x2), W has rows for C (2x2)
        # For each of the 7 products, we set entries in U, V, W.
        
        strassen_entries = [
            # (u_indices, v_indices, w_indices, signs)
            # P1: (a11-a22)*(b21-b22)
            [([0, 2], [1, 3], [1, 1, 3, 3], [1, -1, 1, -1])],
        ]
        
        # Let me just hardcode the Strassen decomposition properly.
        # For a 2x2 * 2x2 = 2x2:
        # U is 4x7, V is 4x7, W is 4x7
        # Row indices: U: [a11,a12,a21,a22] -> 0,1,2,3
        # V: [b11,b12,b21,b22] -> 0,1,2,3
        # W: [c11,c12,c21,c22] -> 0,1,2,3
        
        strassen_U = [
            [1, 1, 0, 0, 1, 0, 1],   # a11
            [0, 1, 0, 0, 0, -1, 0],  # a12
            [0, 0, 1, -1, 0, -1, -1],# a21
            [-1, 0, 1, 1, 1, 0, 0],  # a22
        ]
        strassen_V = [
            [0, 0, 1, 1, 1, 0, 1],   # b11
            [0, 0, 1, -1, 0, 0, 1],  # b12
            [-1, 1, 0, 0, 0, 1, 0],  # b21
            [1, 1, 0, 0, 1, 1, 0],   # b22
        ]
        strassen_W = [
            [0, 0, 0, 1, 1, 1, 0],   # c11
            [1, 1, 0, 0, 0, 0, 0],   # c12
            [0, 0, 1, 0, 1, 0, 0],   # c21
            [1, 0, -1, 1, 0, 0, 1],  # c22
        ]
        
        # First Strassen: A2_top_left (2x2) * B_block_top_left (2x2)
        # A2_top_left is rows 1-2 of A (since A1 is row 0), cols 0-1
        # B_block_top_left is rows 0-1 of B, cols 2*block_idx to 2*block_idx+1
        for s in range(7):
            for u_row in range(4):
                val = strassen_U[u_row][s]
                if val != 0:
                    # u_row 0->a11 (A row 0, col 0), 1->a12 (A row 0, col 1), 2->a21 (A row 1, col 0), 3->a22 (A row 1, col 1)
                    a_row = u_row // 2  # 0 or 1
                    a_col = u_row % 2   # 0 or 1
                    U[(1 + a_row) * m + a_col][col + s] = val
            
            for v_row in range(4):
                val = strassen_V[v_row][s]
                if val != 0:
                    # v_row 0->b11 (B row 0, col 0), 1->b12 (B row 0, col 1), 2->b21 (B row 1, col 0), 3->b22 (B row 1, col 1)
                    b_row = v_row // 2   # 0 or 1
                    b_col = v_row % 2    # 0 or 1
                    V[b_row * p + (2 * block_idx + b_col)][col + s] = val
            
            for w_row in range(4):
                val = strassen_W[w_row][s]
                if val != 0:
                    # w_row 0->c11, 1->c12, 2->c21, 3->c22
                    c_row = w_row // 2  # 0 or 1
                    c_col = w_row % 2   # 0 or 1
                    W[(2 * block_idx + c_col) * n + (1 + c_row)][col + s] = val
        
        # Second Strassen: A2_top_right (2x2) * B_block_top_right (2x2)
        # A2_top_right is rows 1-2 of A, cols 2-3
        # B_block_top_right is rows 2-3 of B, cols 2*block_idx to 2*block_idx+1
        for s in range(7):
            for u_row in range(4):
                val = strassen_U[u_row][s]
                if val != 0:
                    a_row = u_row // 2
                    a_col = u_row % 2
                    U[(1 + a_row) * m + (2 + a_col)][col + s] = val
            
            for v_row in range(4):
                val = strassen_V[v_row][s]
                if val != 0:
                    b_row = v_row // 2
                    b_col = v_row % 2
                    V[(2 + b_row) * p + (2 * block_idx + b_col)][col + s] = val
            
            for w_row in range(4):
                val = strassen_W[w_row][s]
                if val != 0:
                    c_row = w_row // 2
                    c_col = w_row % 2
                    W[(2 * block_idx + c_col) * n + (1 + c_row)][col + s] = val
        
        # Schoolbook part: A2_bot (2x1) * B_block_bot (1x2) -> 2x2
        # A2_bot is rows 1-2 of A, col 4
        # B_block_bot is row 4 of B, cols 2*block_idx to 2*block_idx+1
        # Rank 2: two rank-1 terms
        for k_offset in range(2):
            k = 2 * block_idx + k_offset
            U[1 * m + 4][col] = 1  # A row 1, col 4
            U[2 * m + 4][col] = 1  # A row 2, col 4
            V[4 * p + k][col] = 1  # B row 4, col k
            W[k * n + 1][col] = 1  # C row k, col 1
            W[k * n + 2][col] = 1  # C row k, col 2
            col += 1
        
        col += 0  # col already incremented
    
    # Verify col == 78
    assert col == 78, f"Expected col=78, got {col}"
    
    return {"U": U, "V": V, "W": W}
