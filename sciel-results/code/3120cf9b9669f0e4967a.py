import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    # Rank 77 decomposition:
    # 49 terms for 4x4x4 (Strassen)
    # 28 terms for 4x4x2 (Strassen-like)
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen coefficients for 2x2 x 2x2
    # A blocks: 0=00, 1=01, 2=10, 3=11
    # B blocks: 0=00, 1=01, 2=10, 3=11
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    # Part 1: 4x4x4 multiplication (cols 0-3 of output)
    # This is 4 blocks of 2x2x2, each using Strassen (7 terms)
    # Total 4 * 7 = 28? No.
    # 4x4x4 is 4x4 blocks? No, 2x2 blocks.
    # A is 2x2 blocks. B is 2x2 blocks.
    # There are 4 A-blocks and 4 B-blocks.
    # The multiplication is sum over 4 B-blocks.
    # C_00 = A_00 B_00 + A_01 B_10
    # C_01 = A_00 B_01 + A_01 B_11
    # C_10 = A_10 B_00 + A_11 B_10
    # C_11 = A_10 B_01 + A_11 B_11
    
    # We can use Strassen on the 2x2 block level?
    # No, standard Strassen for 4x4x4 uses 7 multiplications of 2x2x2.
    # Each 2x2x2 multiplication has rank 7.
    # Total rank 49.
    
    col = 0
    # Strassen for 4x4x4
    # The 7 terms of Strassen for 4x4x4 operate on 2x2 blocks.
    # Let's reuse the inner Strassen logic but applied to the 4x4 level.
    # Actually, the current code in the prompt does this correctly by iterating over top-level Strassen terms.
    
    # Top-level Strassen for 4x4x4 (which is 2x2 blocks of 2x2)
    # A blocks: 0,1,2,3 (2x2 each)
    # B blocks: 0,1,2,3 (2x2 each)
    # C blocks: 0,1,2,3 (2x2 each)
    
    # We need to map the 2x2 block indices to the full matrix indices.
    # A[i,j] where i=2*ai+ai2, j=2*bj+bj2.
    # Block (ai, bj) corresponds to rows 2*ai..2*ai+1, cols 2*bj..2*bj+1.
    
    for s in range(7):
        for si in range(7):
            # U part
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            # u_idx: row in 4x4 A
                            # ar, br are block indices. x_row, x_col are inner indices.
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            # V part
            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            # v_idx: row in 4x4 B (cols 0-3)
                            # ac, bc are block indices. y_row, y_inner_col are inner indices.
                            # B is 4x6. Cols 0-3 are the first 4 cols.
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col) 
                            # Wait, B is 4x6. Block structure for cols 0-3?
                            # B blocks for 4x4x4 part: 2x2 blocks.
                            # B col index k = 2*bc + y_inner_col?
                            # If we are doing 4x4x4, B is 4x4.
                            # So col index is 0-3.
                            # Block bc is 0 or 1. Inner col y_inner_col is 0 or 1.
                            # So col = 2*bc + y_inner_col.
                            v_idx = (ac * 2 + y_row) * 6 + (2 * bc + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            
            # W part
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    # w_idx: row in 4x6 C (cols 0-3)
                    # Output C is 4x6.
                    # Block (a, b) from recon[s] refers to C blocks?
                    # recon[s] gives contributions to C blocks.
                    # a is row block, b is col block of C.
                    # za, zb are inner indices.
                    # C col index = 2*b + zb.
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    # Wait, W index convention: W[k*n+i] with n=4?
                    # Task says: W[k*n+i, r]. n=4 (cols of A, rows of C).
                    # So W index = k*4 + i.
                    # k is col of C (0-5). i is row of C (0-3).
                    # In our loop, a is row block, b is col block.
                    # i = 2*a + za.
                    # k = 2*b + zb.
                    w_idx = (2 * b + zb) * 4 + (2 * a + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
    
    # Part 2: 4x4x2 multiplication (cols 4-5 of output)
    # B is 4x2.
    # This is 2 blocks of 2x2? No, B is 4x2.
    # Rows 0-3, Cols 4-5.
    # We can view this as two 4x4x1 multiplications?
    # Or one 4x4x2.
    # Rank 28.
    # Structure:
    # C_0:2, 4:6 = A_0:4, 0:4 * B_0:4, 4:6
    # C_2:4, 4:6 = A_2:4, 0:4 * B_0:4, 4:6
    
    # Using Strassen for 2x2x2 blocks?
    # A is 2x2 blocks. B is 2x1 blocks?
    # Let's use the same recursive structure.
    # Top-level Strassen for 4x4x2?
    # We can split A into 2x2 blocks.
    # B into 2x1 blocks?
    # Actually, we can use the 4x4x4 Strassen terms but with B having zeros in the last 2 cols?
    # No, that would be inefficient.
    
    # Let's use the 28 rank decomposition for 4x4x2.
    # It consists of 4 terms of 2x4x2?
    # Or 2 terms of 2x2x2 (for top rows) and 2 terms of 2x2x2 (for bottom rows)?
    # Top rows (0-1) of C:
    # C[0:2, 4:6] = A[0:2, 0:4] * B[0:4, 4:6]
    # A[0:2, 0:4] is 2x4. B[0:4, 4:6] is 4x2.
    # This is a 2x4x2 multiplication.
    # Rank of 2x4x2 is 14 (as derived earlier: sum of two 2x2x2).
    # So top part rank 14. Bottom part rank 14. Total 28.
    
    # Construct 2x4x2 decomposition.
    # A (2x4) = [A_0, A_1] where A_0, A_1 are 2x2.
    # B (4x2) = [B_0; B_1] where B_0, B_1 are 2x2.
    # C = A_0 B_0 + A_1 B_1.
    # Term 1: A_0 B_0. Rank 7 (Strassen).
    # Term 2: A_1 B_1. Rank 7 (Strassen).
    
    # So for the top part (rows 0-1 of C):
    # We use Strassen for A_0 B_0 (cols 4-5 of B, which are B_0)
    # and Strassen for A_1 B_1 (cols 4-5 of B, which are B_1? No.)
    # B[0:2, 4:6] is B_0. B[2:4, 4:6] is B_1.
    # A[0:2, 0:2] is A_0. A[0:2, 2:4] is A_1.
    # C_top = A_0 B_0 + A_1 B_1.
    
    # For each of these two 2x2x2 multiplications, we use 7 terms.
    # Total 14 terms for top part.
    
    # Similarly for bottom part (rows 2-3 of C):
    # A[2:4, 0:2] is A_2. A[2:4, 2:4] is A_3.
    # C_bottom = A_2 B_0 + A_3 B_1.
    # 14 terms.
    
    # Total 28 terms.
    
    # We will generate these 28 terms.
    
    # Helper to add a 2x2x2 Strassen term
    def add_strassen_2x2x2(U_mat, V_mat, W_mat, col_idx, a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        # a_row_offset, a_col_offset: offset in A for the 2x2 block
        # b_row_offset, b_col_offset: offset in B for the 2x2 block
        # c_row_offset, c_col_offset: offset in C for the 2x2 block
        
        for si in range(7):
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    u_idx = (a_row_offset + x_row) * 4 + (a_col_offset + x_col)
                    U_mat[u_idx, col_idx] += alpha[si][a]
            
            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    v_idx = (b_row_offset + y_row) * 6 + (b_col_offset + y_inner_col)
                    V_mat[v_idx, col_idx] += beta[si][b]
            
            for za, zb, zgamma in recon[si]:
                w_idx = (c_col_offset + zb) * 4 + (c_row_offset + za)
                W_mat[w_idx, col_idx] += zgamma
            
            col_idx += 1
        
        return col_idx

    # Top part: C[0:2, 4:6]
    # Term 1: A[0:2, 0:2] * B[0:2, 4:6]
    col = add_strassen_2x2x2(U, V, W, col, 0, 0, 0, 4, 0, 4)
    # Term 2: A[0:2, 2:4] * B[2:4, 4:6]
    col = add_strassen_2x2x2(U, V, W, col, 0, 2, 2, 4, 0, 4)
    
    # Bottom part: C[2:4, 4:6]
    # Term 3: A[2:4, 0:2] * B[0:2, 4:6]
    col = add_strassen_2x2x2(U, V, W, col, 2, 0, 0, 4, 2, 4)
    # Term 4: A[2:4, 2:4] * B[2:4, 4:6]
    col = add_strassen_2x2x2(U, V, W, col, 2, 2, 2, 4, 2, 4)
    
    # Check if we have used 77 columns
    # 49 + 28 = 77.
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
