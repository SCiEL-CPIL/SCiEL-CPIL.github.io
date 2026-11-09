import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 32
    pn = p * n  # 16
    
    # Rank 56 construction
    # Split A into A_L (2x2) and A_R (2x2)
    # Split B into 4 blocks B_LL, B_LR, B_RL, B_RR (2x4 each)
    # C = [ A_L B_LL + A_R B_RL, A_L B_LR + A_R B_RR ]
    # Each term is 2x2 * 2x4, rank 14
    # Total rank 56
    
    R = 56
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    
    # Helper to add a 2x2x4 Strassen block
    # A_block is 2x2, B_block is 2x4
    # Indices:
    # A_i, j (i in 0,1; j in 0,1) -> U index i*4 + j (if A is 2x4)
    # Wait, A is 2x4.
    # A_L is columns 0,1. A_R is columns 2,3.
    # B_LL is rows 0,1 and cols 0,1? No.
    # B is 4x8.
    # Rows 0,1 are top half. Rows 2,3 are bottom half.
    # Cols 0,1,2,3 are left half. Cols 4,5,6,7 are right half.
    # B_LL: rows 0,1, cols 0,1? No, B is 4x8.
    # We split B by rows into 2x4 blocks.
    # B_top: rows 0,1 (2x8). B_bot: rows 2,3 (2x8).
    # A_L: cols 0,1 (2x2). A_R: cols 2,3 (2x2).
    # Term 1: A_L * B_top_left (2x4)
    # Term 2: A_R * B_bot_left (2x4)
    # Term 3: A_L * B_top_right (2x4)
    # Term 4: A_R * B_bot_right (2x4)
    
    # Strassen for 2x2 * 2x4
    # A is 2x2, B is 2x4.
    # B = [B1, B2] (2x2 each)
    # C = [A B1, A B2]
    # Rank 14.
    
    def add_strassen_2x2x4(offset, a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_col_offset):
        # a_row_offset: 0 or 1 (which 2 rows of A? No, A is 2 rows total)
        # a_col_offset: 0 or 2 (which 2 cols of A)
        # b_row_offset: 0 or 2 (which 2 rows of B)
        # b_col_offset: 0, 2, 4, 6 (which 4 cols of B)
        # c_col_offset: 0 or 4 (which 4 cols of C)
        
        # Strassen 7 multiplications for 2x2 * 2x2
        # M1 = (A22-A11)(B11-B22)
        # M2 = (A11+A21)B22
        # M3 = (A12+A22)B11
        # M4 = A11(B12-B22)
        # M5 = A22(B11-B21)
        # M6 = (A11+A12)B21
        # M7 = (A21+A22)B12
        
        # We need to map these to U, V, W.
        # U index: (i, j) where i is row of A, j is col of A.
        # V index: (j, k) where j is row of B, k is col of B.
        # W index: (k, i) where k is col of C, i is row of C.
        
        # The 7 multiplications correspond to 7 rank-1 terms?
        # No, each "multiplication" in Strassen is a product of linear forms.
        # In the tensor decomposition, a product of linear forms corresponds to a rank-1 term if the linear forms are monomials?
        # No, the tensor rank counts the number of multiplications in the circuit.
        # So we add 7 terms for each 2x2x2 block.
        
        # For 2x2 * 2x4, we have two 2x2x2 blocks.
        # Block 1: A * B1 (cols 0,1 of B_block)
        # Block 2: A * B2 (cols 2,3 of B_block)
        
        # We will add 7 terms for Block 1 and 7 for Block 2.
        
        # Strassen indices for 2x2:
        # A indices: 00, 01, 10, 11
        # B indices: 00, 01, 10, 11
        # C indices: 00, 01, 10, 11
        
        # The 7 products:
        # P1 = (A[1,1]-A[0,0]) * (B[0,0]-B[1,1])
        # P2 = (A[0,0]+A[1,0]) * B[1,1]
        # P3 = (A[0,1]+A[1,1]) * B[0,0]
        # P4 = A[0,0] * (B[0,1]-B[1,1])
        # P5 = A[1,1] * (B[0,0]-B[1,0])
        # P6 = (A[0,0]+A[0,1]) * B[1,0]
        # P7 = (A[1,0]+A[1,1]) * B[0,1]
        
        # C[0,0] = P1 + P4 - P5 + P7
        # C[0,1] = P3 + P5
        # C[1,0] = P2 + P4
        # C[1,1] = P1 - P2 + P3 + P6
        
        # We need to add these to U, V, W.
        # Each Pk is a product (linear_A) * (linear_B).
        # This is a rank-1 term in the tensor?
        # Yes, if we consider the linear forms as defining the entries.
        # U entry for (i, j) is the coefficient of A[i,j] in linear_A.
        # V entry for (j, k) is the coefficient of B[j,k] in linear_B.
        # W entry for (k, i) is the coefficient in the output.
        
        # Wait, the output C is a sum of Pk.
        # So for each Pk, we have a term U_k * V_k * W_k.
        # U_k is the vector of coefficients for A.
        # V_k is the vector of coefficients for B.
        # W_k is the vector of coefficients for C.
        
        # For P1:
        # linear_A = -A[0,0] + A[1,1]
        # linear_B = B[0,0] - B[1,1]
        # C contribution: +1 to C[0,0], +1 to C[1,1]
        
        # We add one column to U, V, W for each Pk.
        
        # There are 7 Pk for the first 2x2 block, and 7 for the second.
        # Total 14 columns per 2x2x4 block.
        
        # We have 4 such blocks. Total 56 columns.
        
        # Block 1: A_L (cols 0,1) * B_TopLeft (rows 0,1, cols 0,1)
        # A_L indices: (0,0), (0,1), (1,0), (1,1)
        # B_TopLeft indices: (0,0), (0,1), (1,0), (1,1) relative to block
        # Global B indices: (0,0), (0,1), (1,0), (1,1)
        # C output: C[0,0], C[0,1], C[1,0], C[1,1]
        
        # We will write a helper to add a 2x2x2 Strassen block.
        
        def add_2x2x2(base_offset, a_i_base, a_j_base, b_i_base, b_j_base, c_i_base, c_k_base):
            # a_i_base: row offset in A (0)
            # a_j_base: col offset in A (0 or 2)
            # b_i_base: row offset in B (0 or 2)
            # b_j_base: col offset in B (0, 2, 4, 6)
            # c_i_base: row offset in C (0)
            # c_k_base: col offset in C (0 or 4)
            
            # 7 terms
            # P1
            U[base_offset, a_i_base*4 + a_j_base + 0] -= 1 # A[0,0]
            U[base_offset, a_i_base*4 + a_j_base + 1] += 0 # A[0,1]
            U[base_offset, a_i_base*4 + a_j_base + 2] += 0 # A[1,0]
            U[base_offset, a_i_base*4 + a_j_base + 3] += 1 # A[1,1]
            
            V[base_offset, b_i_base*8 + b_j_base + 0] += 1 # B[0,0]
            V[base_offset, b_i_base*8 + b_j_base + 1] += 0 # B[0,1]
            V[base_offset, b_i_base*8 + b_j_base + 2] += 0 # B[1,0]
            V[base_offset, b_i_base*8 + b_j_base + 3] -= 1 # B[1,1]
            
            W[base_offset, c_k_base*2 + c_i_base + 0] += 1 # C[0,0]
            W[base_offset, c_k_base*2 + c_i_base + 1] += 0 # C[0,1]
            W[base_offset, c_k_base*2 + c_i_base + 2] += 0 # C[1,0]
            W[base_offset, c_k_base*2 + c_i_base + 3] += 1 # C[1,1]
            
            # P2
            U[base_offset+1, a_i_base*4 + a_j_base + 0] += 1
            U[base_offset+1, a_i_base*4 + a_j_base + 1] += 0
            U[base_offset+1, a_i_base*4 + a_j_base + 2] += 1
            U[base_offset+1, a_i_base*4 + a_j_base + 3] += 0
            
            V[base_offset+1, b_i_base*8 + b_j_base + 0] += 0
            V[base_offset+1, b_i_base*8 + b_j_base + 1] += 0
            V[base_offset+1, b_i_base*8 + b_j_base + 2] += 0
            V[base_offset+1, b_i_base*8 + b_j_base + 3] += 1
            
            W[base_offset+1, c_k_base*2 + c_i_base + 0] += 0
            W[base_offset+1, c_k_base*2 + c_i_base + 1] += 0
            W[base_offset+1, c_k_base*2 + c_i_base + 2] += 1
            W[base_offset+1, c_k_base*2 + c_i_base + 3] -= 1
            
            # P3
            U[base_offset+2, a_i_base*4 + a_j_base + 0] += 0
            U[base_offset+2, a_i_base*4 + a_j_base + 1] += 1
            U[base_offset+2, a_i_base*4 + a_j_base + 2] += 0
            U[base_offset+2, a_i_base*4 + a_j_base + 3] += 1
            
            V[base_offset+2, b_i_base*8 + b_j_base + 0] += 1
            V[base_offset+2, b_i_base*8 + b_j_base + 1] += 0
            V[base_offset+2, b_i_base*8 + b_j_base + 2] += 0
            V[base_offset+2, b_i_base*8 + b_j_base + 3] += 0
            
            W[base_offset+2, c_k_base*2 + c_i_base + 0] += 0
            W[base_offset+2, c_k_base*2 + c_i_base + 1] += 1
            W[base_offset+2, c_k_base*2 + c_i_base + 2] += 0
            W[base_offset+2, c_k_base*2 + c_i_base + 3] += 1
            
            # P4
            U[base_offset+3, a_i_base*4 + a_j_base + 0] += 1
            U[base_offset+3, a_i_base*4 + a_j_base + 1] += 0
            U[base_offset+3, a_i_base*4 + a_j_base + 2] += 0
            U[base_offset+3, a_i_base*4 + a_j_base + 3] += 0
            
            V[base_offset+3, b_i_base*8 + b_j_base + 0] += 0
            V[base_offset+3, b_i_base*8 + b_j_base + 1] += 1
            V[base_offset+3, b_i_base*8 + b_j_base + 2] += 0
            V[base_offset+3, b_i_base*8 + b_j_base + 3] -= 1
            
            W[base_offset+3, c_k_base*2 + c_i_base + 0] += 1
            W[base_offset+3, c_k_base*2 + c_i_base + 1] += 0
            W[base_offset+3, c_k_base*2 + c_i_base + 2] += 1
            W[base_offset+3, c_k_base*2 + c_i_base + 3] += 0
            
            # P5
            U[base_offset+4, a_i_base*4 + a_j_base + 0] += 0
            U[base_offset+4, a_i_base*4 + a_j_base + 1] += 0
            U[base_offset+4, a_i_base*4 + a_j_base + 2] += 0
            U[base_offset+4, a_i_base*4 + a_j_base + 3] += 1
            
            V[base_offset+4, b_i_base*8 + b_j_base + 0] += 1
            V[base_offset+4, b_i_base*8 + b_j_base + 1] += 0
            V[base_offset+4, b_i_base*8 + b_j_base + 2] -= 1
            V[base_offset+4, b_i_base*8 + b_j_base + 3] += 0
            
            W[base_offset+4, c_k_base*2 + c_i_base + 0] -= 1
            W[base_offset+4, c_k_base*2 + c_i_base + 1] += 1
            W[base_offset+4, c_k_base*2 + c_i_base + 2] += 0
            W[base_offset+4, c_k_base*2 + c_i_base + 3] += 0
            
            # P6
            U[base_offset+5, a_i_base*4 + a_j_base + 0] += 1
            U[base_offset+5, a_i_base*4 + a_j_base + 1] += 1
            U[base_offset+5, a_i_base*4 + a_j_base + 2] += 0
            U[base_offset+5, a_i_base*4 + a_j_base + 3] += 0
            
            V[base_offset+5, b_i_base*8 + b_j_base + 0] += 0
            V[base_offset+5, b_i_base*8 + b_j_base + 1] += 0
            V[base_offset+5, b_i_base*8 + b_j_base + 2] += 1
            V[base_offset+5, b_i_base*8 + b_j_base + 3] += 0
            
            W[base_offset+5, c_k_base*2 + c_i_base + 0] += 0
            W[base_offset+5, c_k_base*2 + c_i_base + 1] += 0
            W[base_offset+5, c_k_base*2 + c_i_base + 2] += 0
            W[base_offset+5, c_k_base*2 + c_i_base + 3] += 1
            
            # P7
            U[base_offset+6, a_i_base*4 + a_j_base + 0] += 0
            U[base_offset+6, a_i_base*4 + a_j_base + 1] += 0
            U[base_offset+6, a_i_base*4 + a_j_base + 2] += 1
            U[base_offset+6, a_i_base*4 + a_j_base + 3] += 1
            
            V[base_offset+6, b_i_base*8 + b_j_base + 0] += 0
            V[base_offset+6, b_i_base*8 + b_j_base + 1] += 1
            V[base_offset+6, b_i_base*8 + b_j_base + 2] += 0
            V[base_offset+6, b_i_base*8 + b_j_base + 3] += 0
            
            W[base_offset+6, c_k_base*2 + c_i_base + 0] += 1
            W[base_offset+6, c_k_base*2 + c_i_base + 1] += 0
            W[base_offset+6, c_k_base*2 + c_i_base + 2] += 0
            W[base_offset+6, c_k_base*2 + c_i_base + 3] += 0

        # Block 1: A_L (cols 0,1) * B_TopLeft (rows 0,1, cols 0,1)
        # a_i_base=0, a_j_base=0, b_i_base=0, b_j_base=0, c_i_base=0, c_k_base=0
        add_2x2x2(0, 0, 0, 0, 0, 0, 0)
        
        # Block 2: A_L (cols 0,1) * B_TopRight (rows 0,1, cols 2,3)
        # a_i_base=0, a_j_base=0, b_i_base=0, b_j_base=2, c_i_base=0, c_k_base=0
        # Wait, B_TopRight is cols 2,3 of B_Top.
        # B_Top is rows 0,1.
        # So b_j_base=2.
        # C output is C[0,0], C[0,1]? No, C cols 0,1.
        # So c_k_base=0.
        add_2x2x2(7, 0, 0, 0, 2, 0, 0)
        
        # Block 3: A_R (cols 2,3) * B_BotLeft (rows 2,3, cols 0,1)
        # a_i_base=0, a_j_base=2, b_i_base=2, b_j_base=0, c_i_base=0, c_k_base=0
        add_2x2x2(14, 0, 2, 2, 0, 0, 0)
        
        # Block 4: A_R (cols 2,3) * B_BotRight (rows 2,3, cols 2,3)
        # a_i_base=0, a_j_base=2, b_i_base=2, b_j_base=2, c_i_base=0, c_k_base=0
        add_2x2x2(21, 0, 2, 2, 2, 0, 0)
        
        # Block 5: A_L (cols 0,1) * B_TopLeft (rows 0,1, cols 4,5) -> C cols 4,5
        # a_i_base=0, a_j_base=0, b_i_base=0, b_j_base=4, c_i_base=0, c_k_base=4
        add_2x2x2(28, 0, 0, 0, 4, 0, 4)
        
        # Block 6: A_L (cols 0,1) * B_TopRight (rows 0,1, cols 6,7) -> C cols 6,7
        # a_i_base=0, a_j_base=0, b_i_base=0, b_j_base=6, c_i_base=0, c_k_base=4
        add_2x2x2(35, 0, 0, 0, 6, 0, 4)
        
        # Block 7: A_R (cols 2,3) * B_BotLeft (rows 2,3, cols 4,5) -> C cols 4,5
        # a_i_base=0, a_j_base=2, b_i_base=2, b_j_base=4, c_i_base=0, c_k_base=4
        add_2x2x2(42, 0, 2, 2, 4, 0, 4)
        
        # Block 8: A_R (cols 2,3) * B_BotRight (rows 2,3, cols 6,7) -> C cols 6,7
        # a_i_base=0, a_j_base=2, b_i_base=2, b_j_base=6, c_i_base=0, c_k_base=4
        add_2x2x2(49, 0, 2, 2, 6, 0, 4)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
