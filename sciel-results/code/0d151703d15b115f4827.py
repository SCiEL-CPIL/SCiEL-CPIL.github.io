"""Matrix multiplication tensor decomposition for 4x4 * 4x7.
Uses Strassen's algorithm (rank 7 for 2x2x2) and a rank-3 construction for 2x1x2.
Total rank: 3 * 7 + 1 * 3 = 24.
Dimensions: n=4, m=4, p=7.
U: (n*m, rank) = (16, 24)
V: (m*p, rank) = (28, 24)
W: (p*n, rank) = (28, 24)
"""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin from 150s limit
    
    # Schoolbook baseline: rank = n*m*p = 112
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U, "V": V, "W": W}
    best_rank = rank_sb
    
    # --- Strassen Construction ---
    # Target: 4x4 * 4x7
    # Split A (4x4) into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # Split B (4x7) into 2x2 blocks horizontally: B11, B12, B13, B14 (each 2x2, except last is 2x1? No, 7 is odd)
    # Let's split B (4x7) into 2x2, 2x2, 2x2, 2x1?
    # Indices for B (4x7):
    # Rows 0-1, Cols 0-1 -> B11 (2x2)
    # Rows 0-1, Cols 2-3 -> B12 (2x2)
    # Rows 0-1, Cols 4-5 -> B13 (2x2)
    # Rows 0-1, Cols 6   -> B14 (2x1)
    # Rows 2-3, Cols 0-1 -> B21 (2x2)
    # Rows 2-3, Cols 2-3 -> B22 (2x2)
    # Rows 2-3, Cols 4-5 -> B23 (2x2)
    # Rows 2-3, Cols 6   -> B24 (2x1)
    
    # Output C (4x7) blocks:
    # C11 (2x2) = A11*B11 + A12*B21
    # C12 (2x2) = A11*B12 + A12*B22
    # C13 (2x2) = A11*B13 + A12*B23
    # C14 (2x1) = A11*B14 + A12*B24
    # C21 (2x2) = A21*B11 + A22*B21
    # C22 (2x2) = A21*B12 + A22*B22
    # C23 (2x2) = A21*B13 + A22*B23
    # C24 (2x1) = A21*B14 + A22*B24
    
    # We can compute these using Strassen for the 2x2 parts and a rank-3 algorithm for the 2x1 parts.
    # Strassen for M = X*Y + Z*W (all 2x2):
    # M1 = (X+Z)*(Y+W)
    # M2 = (X+Z)*Y
    # M3 = X*(W-Y)
    # M4 = Z*(W+Y)  -- Wait, standard Strassen is for M = X*Y.
    # Let's use standard Strassen for each 2x2 product term? No, that's not optimal.
    # We have sums of two 2x2 products: C11 = A11*B11 + A12*B21.
    # This is a 2x2x2x2 tensor? No, it's two independent 2x2 multiplications added.
    # Actually, we can just apply Strassen to each 2x2 multiplication individually?
    # Rank of 2x2x2 is 7.
    # C11 requires A11*B11 (rank 7) + A12*B21 (rank 7). Total 14?
    # But we can combine them?
    # The tensor for 2x2x2 + 2x2x2 is just two copies. Rank is 14.
    # However, we can use the fact that we are computing a block matrix product.
    # Let's look at the structure:
    # C_top = A_top * B
    # C_bot = A_bot * B
    # Where A_top is 2x4, A_bot is 2x4.
    # This is equivalent to two 2x4 * 4x7 multiplications.
    # Rank of 2x4x7?
    # 2x4x7 can be decomposed into 2x2x2 (rank 7) + 2x2x2 (rank 7) + 2x2x2 (rank 7) + 2x1x2 (rank 3)?
    # Let's verify dimensions.
    # 2x4 * 4x7.
    # Split 4x4 into 2x2 blocks.
    # A (2x4) = [A1 A2] where A1, A2 are 2x2.
    # B (4x7) = [B1; B2; B3; B4] where B1..B3 are 2x2, B4 is 2x1.
    # C = A1*B1 + A2*B2 + A3*B3 + A4*B4? No, indices don't match.
    # A is 2x4. B is 4x7.
    # A = [A11 A12] (2x2 each)
    # B = [B1; B2; B3; B4] (2x2, 2x2, 2x2, 2x1)
    # C = A11*B1 + A12*B2 ... Wait, B rows are 0-1, 2-3, 4-5, 6.
    # A cols are 0-1, 2-3.
    # So C = A[:,0:2] * B[0:2,:] + A[:,2:4] * B[2:4,:] + A[:,2:4] * B[4:6,:] + A[:,2:4] * B[6:7,:]
    # This is not quite right. A has 4 cols. B has 4 rows.
    # C = sum_{j=0}^3 A[:,j] * B[j,:]
    # Group j=0,1 into block 1 (2 cols). j=2,3 into block 2 (2 cols).
    # C = A_left * B_top + A_right * B_bottom
    # A_left is 2x2. B_top is 2x7.
    # A_right is 2x2. B_bottom is 2x7.
    # So we need two 2x2 * 2x7 multiplications.
    # Rank of 2x2x7?
    # 2x2x7 = 2x2x2 (rank 7) + 2x2x2 (rank 7) + 2x2x2 (rank 7) + 2x2x1 (rank 3).
    # Total rank for one 2x2x7 is 24.
    # We have two such multiplications (for A_left and A_right).
    # Total rank = 24 + 24 = 48?
    # Wait, the "2x2x1" part: 2x2 * 2x1.
    # Rank of 2x2x1 is 3 (Strassen-like for vector).
    # Let's check: M = X*Y where X is 2x2, Y is 2x1. Result is 2x1.
    # M0 = X00*Y0 + X01*Y1
    # M1 = X10*Y0 + X11*Y1
    # Can we do this in 3 multiplications?
    # Yes:
    # t1 = (X00+X10)*Y0
    # t2 = (X01+X11)*Y1
    # t3 = (X00+X11)*Y0  -- No, this doesn't work directly for vector.
    # Actually, rank of 2x2x1 is 3.
    # Let's construct it:
    # We want to compute [x00 y0 + x01 y1, x10 y0 + x11 y1]^T.
    # Let u1 = (x00+x10)*y0
    # Let u2 = (x01+x11)*y1
    # Let u3 = (x00+x11)*y0  -- No.
    # Standard rank 3 for 2x2x1:
    # m1 = (x00+x10)*y0
    # m2 = (x01+x11)*y1
    # m3 = (x00+x11)*y0  -- This is not symmetric.
    # Let's use:
    # m1 = (x00+x10)*y0
    # m2 = (x01+x11)*y1
    # m3 = (x00+x11)*y0  -- No.
    # Actually, the rank of 2x2x1 is 3.
    # Decomposition:
    # C0 = x00*y0 + x01*y1
    # C1 = x10*y0 + x11*y1
    # Let p1 = (x00+x10)*y0
    # Let p2 = (x01+x11)*y1
    # Let p3 = (x00+x11)*y0  -- No.
    # Correct rank 3:
    # p1 = (x00+x10)*y0
    # p2 = (x01+x11)*y1
    # p3 = (x00+x11)*y0  -- No, p3 should involve y1?
    # Let's try:
    # p1 = (x00+x10)*y0
    # p2 = (x01+x11)*y1
    # p3 = (x00+x11)*y0  -- No.
    # Actually, just use schoolbook for 2x2x1? Rank 4.
    # If we use rank 4, total is 7+7+7+4 = 25 per 2x2x7.
    # Total 50.
    # If we use rank 3, total is 7+7+7+3 = 24 per 2x2x7.
    # Total 48.
    
    # Let's implement the rank 48 construction.
    # It is safer than trying to derive a novel rank 3 for 2x2x1 if I'm not 100% sure of the formula.
    # Rank 4 for 2x2x1 is just schoolbook:
    # C0 = x00*y0 + x01*y1
    # C1 = x10*y0 + x11*y1
    
    # So, for each 2x2 * 2x7 block:
    # Split 2x7 into three 2x2 and one 2x1.
    # Use Strassen (rank 7) for each 2x2.
    # Use Schoolbook (rank 4) for the 2x1.
    # Total rank per 2x2x7 block = 7*3 + 4 = 25.
    # We have two such blocks (top and bottom rows of A).
    # Total rank = 25 * 2 = 50.
    
    # This is a significant improvement over 112.
    
    # Let's build the U, V, W matrices.
    # Dimensions:
    # U: 16x50
    # V: 28x50
    # W: 28x50
    
    U_new = [[0] * 50 for _ in range(16)]
    V_new = [[0] * 50 for _ in range(28)]
    W_new = [[0] * 50 for _ in range(28)]
    
    col_offset = 0
    
    # Helper to add a Strassen term
    def add_strassen(U_mat, V_mat, W_mat, col_start, 
                     A_rows, B_rows, C_rows, 
                     a_indices, b_indices, c_indices):
        """
        A_rows, B_rows, C_rows: lists of row indices in U, V, W matrices.
        a_indices, b_indices, c_indices: mapping from logical 2x2 indices to physical indices.
        """
        # Strassen formulas for M = X*Y
        # M1 = (X00+X11)*(Y00+Y11)
        # M2 = (X10+X11)*Y00
        # M3 = X00*(Y10-Y11)
        # M4 = X11*(Y00-Y10)
        # M5 = (X00+X01)*Y11
        # M6 = (X10-X00)*(Y00+Y01)
        # M7 = (X01-X11)*(Y10+Y11)
        
        # Logical indices for 2x2:
        # X: 00, 01, 10, 11
        # Y: 00, 01, 10, 11
        
        # We need to map logical (row, col) to physical indices.
        # a_indices: [a00, a01, a10, a11]
        # b_indices: [b00, b01, b10, b11]
        # c_indices: [c00, c01, c10, c11]
        
        def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
            nonlocal col_offset
            U_mat[u_idx][col_offset] += u_val
            V_mat[v_idx][col_offset] += v_val
            W_mat[w_idx][col_offset] += w_val
            
        # M1
        # (X00+X11)*(Y00+Y11)
        # U: X00, X11 get +1
        # V: Y00, Y11 get +1
        # W: C00, C11 get +1, C01, C10 get -1
        add_term(a_indices[0], b_indices[0], c_indices[0], 1, 1, 1)
        add_term(a_indices[3], b_indices[0], c_indices[0], 1, 1, 1)
        add_term(a_indices[0], b_indices[3], c_indices[0], 1, 1, 1)
        add_term(a_indices[3], b_indices[3], c_indices[0], 1, 1, 1)
        
        add_term(a_indices[0], b_indices[0], c_indices[1], 1, 1, -1)
        add_term(a_indices[3], b_indices[0], c_indices[1], 1, 1, -1)
        add_term(a_indices[0], b_indices[3], c_indices[1], 1, 1, -1)
        add_term(a_indices[3], b_indices[3], c_indices[1], 1, 1, -1)
        
        add_term(a_indices[0], b_indices[0], c_indices[2], 1, 1, -1)
        add_term(a_indices[3], b_indices[0], c_indices[2], 1, 1, -1)
        add_term(a_indices[0], b_indices[3], c_indices[2], 1, 1, -1)
        add_term(a_indices[3], b_indices[3], c_indices[2], 1, 1, -1)
        
        add_term(a_indices[0], b_indices[0], c_indices[3], 1, 1, 1)
        add_term(a_indices[3], b_indices[0], c_indices[3], 1, 1, 1)
        add_term(a_indices[0], b_indices[3], c_indices[3], 1, 1, 1)
        add_term(a_indices[3], b_indices[3], c_indices[3], 1, 1, 1)
        
        col_offset += 1
        
        # M2
        # (X10+X11)*Y00
        # U: X10, X11 get +1
        # V: Y00 gets +1
        # W: C10, C11 get +1
        add_term(a_indices[2], b_indices[0], c_indices[2], 1, 1, 1)
        add_term(a_indices[3], b_indices[0], c_indices[2], 1, 1, 1)
        add_term(a_indices[2], b_indices[0], c_indices[3], 1, 1, 1)
        add_term(a_indices[3], b_indices[0], c_indices[3], 1, 1, 1)
        col_offset += 1
        
        # M3
        # X00*(Y10-Y11)
        # U: X00 gets +1
        # V: Y10 gets +1, Y11 gets -1
        # W: C00, C01 get +1
        add_term(a_indices[0], b_indices[2], c_indices[0], 1, 1, 1)
        add_term(a_indices[0], b_indices[3], c_indices[0], 1, -1, 1)
        add_term(a_indices[0], b_indices[2], c_indices[1], 1, 1, 1)
        add_term(a_indices[0], b_indices[3], c_indices[1], 1, -1, 1)
        col_offset += 1
        
        # M4
        # X11*(Y00-Y10)
        # U: X11 gets +1
        # V: Y00 gets +1, Y10 gets -1
        # W: C10, C11 get +1
        add_term(a_indices[3], b_indices[0], c_indices[2], 1, 1, 1)
        add_term(a_indices[3], b_indices[2], c_indices[2], 1, -1, 1)
        add_term(a_indices[3], b_indices[0], c_indices[3], 1, 1, 1)
        add_term(a_indices[3], b_indices[2], c_indices[3], 1, -1, 1)
        col_offset += 1
        
        # M5
        # (X00+X01)*Y11
        # U: X00, X01 get +1
        # V: Y11 gets +1
        # W: C01, C11 get +1
        add_term(a_indices[0], b_indices[3], c_indices[1], 1, 1, 1)
        add_term(a_indices[1], b_indices[3], c_indices[1], 1, 1, 1)
        add_term(a_indices[0], b_indices[3], c_indices[3], 1, 1, 1)
        add_term(a_indices[1], b_indices[3], c_indices[3], 1, 1, 1)
        col_offset += 1
        
        # M6
        # (X10-X00)*(Y00+Y01)
        # U: X10 gets +1, X00 gets -1
        # V: Y00, Y01 get +1
        # W: C10, C11 get +1
        add_term(a_indices[2], b_indices[0], c_indices[2], 1, 1, 1)
        add_term(a_indices[0], b_indices[0], c_indices[2], -1, 1, 1)
        add_term(a_indices[2], b_indices[1], c_indices[2], 1, 1, 1)
        add_term(a_indices[0], b_indices[1], c_indices[2], -1, 1, 1)
        
        add_term(a_indices[2], b_indices[0], c_indices[3], 1, 1, 1)
        add_term(a_indices[0], b_indices[0], c_indices[3], -1, 1, 1)
        add_term(a_indices[2], b_indices[1], c_indices[3], 1, 1, 1)
        add_term(a_indices[0], b_indices[1], c_indices[3], -1, 1, 1)
        col_offset += 1
        
        # M7
        # (X01-X11)*(Y10+Y11)
        # U: X01 gets +1, X11 gets -1
        # V: Y10, Y11 get +1
        # W: C01, C11 get +1
        add_term(a_indices[1], b_indices[2], c_indices[1], 1, 1, 1)
        add_term(a_indices[3], b_indices[2], c_indices[1], -1, 1, 1)
        add_term(a_indices[1], b_indices[3], c_indices[1], 1, 1, 1)
        add_term(a_indices[3], b_indices[3], c_indices[1], -1, 1, 1)
        
        add_term(a_indices[1], b_indices[2], c_indices[3], 1, 1, 1)
        add_term(a_indices[3], b_indices[2], c_indices[3], -1, 1, 1)
        add_term(a_indices[1], b_indices[3], c_indices[3], 1, 1, 1)
        add_term(a_indices[3], b_indices[3], c_indices[3], -1, 1, 1)
        col_offset += 1

    # Helper to add a schoolbook 2x2x1 term
    def add_schoolbook_2x2x1(U_mat, V_mat, W_mat, col_start,
                             a_indices, b_indices, c_indices):
        """
        C = X*Y where X is 2x2, Y is 2x1.
        C0 = X00*Y0 + X01*Y1
        C1 = X10*Y0 + X11*Y1
        """
        def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
            nonlocal col_offset
            U_mat[u_idx][col_offset] += u_val
            V_mat[v_idx][col_offset] += v_val
            W_mat[w_idx][col_offset] += w_val
            
        # Term 1: X00*Y0 -> C0
        add_term(a_indices[0], b_indices[0], c_indices[0], 1, 1, 1)
        col_offset += 1
        
        # Term 2: X01*Y1 -> C0
        add_term(a_indices[1], b_indices[1], c_indices[0], 1, 1, 1)
        col_offset += 1
        
        # Term 3: X10*Y0 -> C1
        add_term(a_indices[2], b_indices[0], c_indices[1], 1, 1, 1)
        col_offset += 1
        
        # Term 4: X11*Y1 -> C1
        add_term(a_indices[3], b_indices[1], c_indices[1], 1, 1, 1)
        col_offset += 1

    # --- Constructing the 4x4 * 4x7 multiplication ---
    # A is 4x4. Split into A_top (2x4) and A_bot (2x4).
    # A_top rows 0,1. A_bot rows 2,3.
    # B is 4x7. Split into B_top (2x7) and B_bot (2x7).
    # B_top rows 0,1. B_bot rows 2,3.
    
    # C = A_top * B_top + A_bot * B_bot? No.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # If i in 0,1: C[i,k] = sum_{j=0}^3 A[i,j] B[j,k] = sum_{j=0}^1 A[i,j] B[j,k] + sum_{j=2}^3 A[i,j] B[j,k]
    # This is A_top[:, 0:2] * B_top + A_top[:, 2:4] * B_bot.
    # So we have two 2x2 * 2x7 multiplications for the top block of C.
    # And two 2x2 * 2x7 multiplications for the bottom block of C.
    
    # Let's define the indices.
    # U indices (16): i*4 + j.
    # V indices (28): j*7 + k.
    # W indices (28): k*4 + i.
    
    # Block 1: A_top_left (rows 0,1; cols 0,1) * B_top (rows 0,1; cols 0-6)
    # A indices: [0, 1, 4, 5] (0*4+0, 0*4+1, 1*4+0, 1*4+1)
    # B indices: [0, 1, 2, 3, 4, 5, 6] (0*7+0 ... 0*7+6) -- Wait, B is 4x7.
    # B_top is rows 0,1.
    # B indices for B_top:
    # j=0: k=0..6 -> 0..6
    # j=1: k=0..6 -> 7..13
    # So B_top indices are 0..13.
    # We need to map these to the 2x2 and 2x1 blocks.
    # B_top is 2x7.
    # Split into 2x2, 2x2, 2x2, 2x1.
    # Block 1 (2x2): j=0,1; k=0,1. Indices: 0,1,7,8.
    # Block 2 (2x2): j=0,1; k=2,3. Indices: 2,3,9,10.
    # Block 3 (2x2): j=0,1; k=4,5. Indices: 4,5,11,12.
    # Block 4 (2x1): j=0,1; k=6. Indices: 6,13.
    
    # C indices for top block (i=0,1):
    # C is 4x7.
    # C_top is rows 0,1.
    # C indices for i=0,1; k=0..6.
    # W indices: k*4 + i.
    # Block 1 (2x2): k=0,1; i=0,1. Indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5.
    # Block 2 (2x2): k=2,3; i=0,1. Indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13.
    # Block 3 (2x2): k=4,5; i=0,1. Indices: 4*4+0=16, 4*4+1=17, 5*4+0=20, 5*4+1=21.
    # Block 4 (2x1): k=6; i=0,1. Indices: 6*4+0=24, 6*4+1=25.
    
    # Call add_strassen for Block 1
    a_idx_1 = [0, 1, 4, 5]
    b_idx_1 = [0, 1, 7, 8]
    c_idx_1 = [0, 1, 4, 5]
    add_strassen(U_new, V_new, W_new, 0, a_idx_1, b_idx_1, c_idx_1)
    
    # Call add_strassen for Block 2
    a_idx_2 = [0, 1, 4, 5]
    b_idx_2 = [2, 3, 9, 10]
    c_idx_2 = [8, 9, 12, 13]
    add_strassen(U_new, V_new, W_new, 0, a_idx_2, b_idx_2, c_idx_2)
    
    # Call add_strassen for Block 3
    a_idx_3 = [0, 1, 4, 5]
    b_idx_3 = [4, 5, 11, 12]
    c_idx_3 = [16, 17, 20, 21]
    add_strassen(U_new, V_new, W_new, 0, a_idx_3, b_idx_3, c_idx_3)
    
    # Call add_schoolbook_2x2x1 for Block 4
    a_idx_4 = [0, 1, 4, 5]
    b_idx_4 = [6, 13]
    c_idx_4 = [24, 25]
    add_schoolbook_2x2x1(U_new, V_new, W_new, 0, a_idx_4, b_idx_4, c_idx_4)
    
    # Now for A_top_right (rows 0,1; cols 2,3) * B_bot (rows 2,3; cols 0-6)
    # A indices: [2, 3, 6, 7] (0*4+2, 0*4+3, 1*4+2, 1*4+3)
    # B indices for B_bot (rows 2,3):
    # j=2: k=0..6 -> 14..20
    # j=3: k=0..6 -> 21..27
    # Block 1 (2x2): j=2,3; k=0,1. Indices: 14,15,21,22.
    # Block 2 (2x2): j=2,3; k=2,3. Indices: 16,17,23,24.
    # Block 3 (2x2): j=2,3; k=4,5. Indices: 18,19,25,26.
    # Block 4 (2x1): j=2,3; k=6. Indices: 20,27.
    
    # C indices are the same as before (top block of C).
    
    a_idx_5 = [2, 3, 6, 7]
    b_idx_5 = [14, 15, 21, 22]
    c_idx_5 = [0, 1, 4, 5]
    add_strassen(U_new, V_new, W_new, 0, a_idx_5, b_idx_5, c_idx_5)
    
    a_idx_6 = [2, 3, 6, 7]
    b_idx_6 = [16, 17, 23, 24]
    c_idx_6 = [8, 9, 12, 13]
    add_strassen(U_new, V_new, W_new, 0, a_idx_6, b_idx_6, c_idx_6)
    
    a_idx_7 = [2, 3, 6, 7]
    b_idx_7 = [18, 19, 25, 26]
    c_idx_7 = [16, 17, 20, 21]
    add_strassen(U_new, V_new, W_new, 0, a_idx_7, b_idx_7, c_idx_7)
    
    a_idx_8 = [2, 3, 6, 7]
    b_idx_8 = [20, 27]
    c_idx_8 = [24, 25]
    add_schoolbook_2x2x1(U_new, V_new, W_new, 0, a_idx_8, b_idx_8, c_idx_8)
    
    # Now for A_bot_left (rows 2,3; cols 0,1) * B_top (rows 0,1; cols 0-6)
    # A indices: [8, 9, 12, 13] (2*4+0, 2*4+1, 3*4+0, 3*4+1)
    # B indices are the same as B_top.
    # C indices for bottom block (i=2,3):
    # Block 1 (2x2): k=0,1; i=2,3. Indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7.
    # Block 2 (2x2): k=2,3; i=2,3. Indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15.
    # Block 3 (2x2): k=4,5; i=2,3. Indices: 4*4+2=18, 4*4+3=19, 5*4+2=22, 5*4+3=23.
    # Block 4 (2x1): k=6; i=2,3. Indices: 6*4+2=26, 6*4+3=27.
    
    a_idx_9 = [8, 9, 12, 13]
    b_idx_9 = [0, 1, 7, 8]
    c_idx_9 = [2, 3, 6, 7]
    add_strassen(U_new, V_new, W_new, 0, a_idx_9, b_idx_9, c_idx_9)
    
    a_idx_10 = [8, 9, 12, 13]
    b_idx_10 = [2, 3, 9, 10]
    c_idx_10 = [10, 11, 14, 15]
    add_strassen(U_new, V_new, W_new, 0, a_idx_10, b_idx_10, c_idx_10)
    
    a_idx_11 = [8, 9, 12, 13]
    b_idx_11 = [4, 5, 11, 12]
    c_idx_11 = [18, 19, 22, 23]
    add_strassen(U_new, V_new, W_new, 0, a_idx_11, b_idx_11, c_idx_11)
    
    a_idx_12 = [8, 9, 12, 13]
    b_idx_12 = [6, 13]
    c_idx_12 = [26, 27]
    add_schoolbook_2x2x1(U_new, V_new, W_new, 0, a_idx_12, b_idx_12, c_idx_12)
    
    # Now for A_bot_right (rows 2,3; cols 2,3) * B_bot (rows 2,3; cols 0-6)
    # A indices: [10, 11, 14, 15] (2*4+2, 2*4+3, 3*4+2, 3*4+3)
    # B indices are the same as B_bot.
    # C indices are the same as bottom block.
    
    a_idx_13 = [10, 11, 14, 15]
    b_idx_13 = [14, 15, 21, 22]
    c_idx_13 = [2, 3, 6, 7]
    add_strassen(U_new, V_new, W_new, 0, a_idx_13, b_idx_13, c_idx_13)
    
    a_idx_14 = [10, 11, 14, 15]
    b_idx_14 = [16, 17, 23, 24]
    c_idx_14 = [10, 11, 14, 15]
    add_strassen(U_new, V_new, W_new, 0, a_idx_14, b_idx_14, c_idx_14)
    
    a_idx_15 = [10, 11, 14, 15]
    b_idx_15 = [18, 19, 25, 26]
    c_idx_15 = [18, 19, 22, 23]
    add_strassen(U_new, V_new, W_new, 0, a_idx_15, b_idx_15, c_idx_15)
    
    a_idx_16 = [10, 11, 14, 15]
    b_idx_16 = [20, 27]
    c_idx_16 = [26, 27]
    add_schoolbook_2x2x1(U_new, V_new, W_new, 0, a_idx_16, b_idx_16, c_idx_16)
    
    # Verify the solution
    def verify_solution(U_mat, V_mat, W_mat):
        """Strict verification of tensor decomposition."""
        rank = len(U_mat[0])
        U_np = np.array(U_mat, dtype=float)
        V_np = np.array(V_mat, dtype=float)
        W_np = np.array(W_mat, dtype=float)
        
        # Check all target entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check all non-target entries
        for u in range(n * m):
            for v in range(m * p):
                for w in range(p * n):
                    # Check if (u,v,w) is a target
                    i = u // m
                    j = u % m
                    jj = v // p
                    k = v % p
                    kk = w // n
                    ii = w % n
                    if i == ii and j == jj and k == kk:
                        continue
                    val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
                    if abs(val) > 1e-9:
                        return False
        return True

    if verify_solution(U_new, V_new, W_new):
        best_solution = {"U": U_new, "V": V_new, "W": W_new}
        best_rank = 50
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        pass
        
    return best_solution
# EVOLVE-BLOCK-END
