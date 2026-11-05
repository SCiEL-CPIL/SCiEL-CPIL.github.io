import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # --- Attempt: Rank 60 (Laderman rank-15 for 3x3 blocks x rank-4 schoolbook for 2x2 blocks) ---
    # Partition 6x6 into 3x3 grid of 2x2 blocks
    # Block-level: Laderman 15-product algorithm for 3x3 matrix mult
    # Inner-level: rank-4 schoolbook for 2x2 matrix mult
    # Total rank = 15 * 4 = 60
    
    # Laderman 15-product algorithm for 3x3 matrix multiplication
    # Each product is of the form: (linear combo of A entries) * (linear combo of B entries)
    # We need to express this as a tensor decomposition
    
    # The 15 Laderman products (standard formulation):
    # p1  = A[0,0] * (B[1,1] - B[1,2])
    # p2  = (A[0,0] + A[0,1]) * B[1,2]
    # p3  = (A[0,1] - A[0,0]) * B[0,0]
    # p4  = A[0,1] * (B[0,2] - B[0,0])
    # p5  = (A[1,1] + A[2,2]) * B[0,0]
    # p6  = (A[1,1] - A[2,2]) * (B[0,0] + B[0,1])
    # p7  = (A[1,1] + A[1,2]) * B[2,0]
    # p8  = (A[1,2] - A[1,1]) * (B[2,0] + B[2,1])
    # p9  = (A[2,0] + A[2,2]) * B[1,1]
    # p10 = (A[2,0] - A[2,2]) * (B[1,1] + B[1,2])
    # p11 = (A[2,0] + A[2,1]) * B[2,2]
    # p12 = (A[2,1] - A[2,0]) * (B[2,2] + B[2,1])
    # p13 = (A[0,0] + A[2,2]) * (B[0,0] + B[1,1] + B[2,2])
    # p14 = (A[0,0] + A[0,1] + A[2,1] + A[2,2]) * (B[0,0] + B[0,1] + B[1,0] + B[1,1])
    # p15 = (A[0,1] + A[1,0] + A[2,0] + A[2,1]) * (B[1,0] + B[1,1] + B[2,1] + B[2,2])
    
    # Define the 15 products as (coeff_A, coeff_B) pairs
    # coeff_A[r, c] means coefficient of A[r,c] in the first factor of product c
    # coeff_B[r, c] means coefficient of B[r,c] in the second factor of product c
    
    coeff_A = np.zeros((9, 15), dtype=int)
    coeff_B = np.zeros((9, 15), dtype=int)
    
    # p1: A[0,0] * (B[1,1] - B[1,2])
    coeff_A[0, 0] = 1  # A[0,0]
    coeff_B[4, 0] = 1  # B[1,1] (row 1, col 1 -> index 4)
    coeff_B[5, 0] = -1 # B[1,2] (row 1, col 2 -> index 5)
    
    # p2: (A[0,0] + A[0,1]) * B[1,2]
    coeff_A[0, 1] = 1  # A[0,0]
    coeff_A[1, 1] = 1  # A[0,1]
    coeff_B[5, 1] = 1  # B[1,2]
    
    # p3: (A[0,1] - A[0,0]) * B[0,0]
    coeff_A[1, 2] = 1  # A[0,1]
    coeff_A[0, 2] = -1 # A[0,0]
    coeff_B[0, 2] = 1  # B[0,0]
    
    # p4: A[0,1] * (B[0,2] - B[0,0])
    coeff_A[1, 3] = 1  # A[0,1]
    coeff_B[2, 3] = 1  # B[0,2]
    coeff_B[0, 3] = -1 # B[0,0]
    
    # p5: (A[1,1] + A[2,2]) * B[0,0]
    coeff_A[4, 4] = 1  # A[1,1]
    coeff_A[8, 4] = 1  # A[2,2]
    coeff_B[0, 4] = 1  # B[0,0]
    
    # p6: (A[1,1] - A[2,2]) * (B[0,0] + B[0,1])
    coeff_A[4, 5] = 1  # A[1,1]
    coeff_A[8, 5] = -1 # A[2,2]
    coeff_B[0, 5] = 1  # B[0,0]
    coeff_B[1, 5] = 1  # B[0,1]
    
    # p7: (A[1,1] + A[1,2]) * B[2,0]
    coeff_A[4, 6] = 1  # A[1,1]
    coeff_A[5, 6] = 1  # A[1,2]
    coeff_B[6, 6] = 1  # B[2,0]
    
    # p8: (A[1,2] - A[1,1]) * (B[2,0] + B[2,1])
    coeff_A[5, 7] = 1  # A[1,2]
    coeff_A[4, 7] = -1 # A[1,1]
    coeff_B[6, 7] = 1  # B[2,0]
    coeff_B[7, 7] = 1  # B[2,1]
    
    # p9: (A[2,0] + A[2,2]) * B[1,1]
    coeff_A[6, 8] = 1  # A[2,0]
    coeff_A[8, 8] = 1  # A[2,2]
    coeff_B[4, 8] = 1  # B[1,1]
    
    # p10: (A[2,0] - A[2,2]) * (B[1,1] + B[1,2])
    coeff_A[6, 9] = 1  # A[2,0]
    coeff_A[8, 9] = -1 # A[2,2]
    coeff_B[4, 9] = 1  # B[1,1]
    coeff_B[5, 9] = 1  # B[1,2]
    
    # p11: (A[2,0] + A[2,1]) * B[2,2]
    coeff_A[6, 10] = 1  # A[2,0]
    coeff_A[7, 10] = 1  # A[2,1]
    coeff_B[8, 10] = 1  # B[2,2]
    
    # p12: (A[2,1] - A[2,0]) * (B[2,2] + B[2,1])
    coeff_A[7, 11] = 1  # A[2,1]
    coeff_A[6, 11] = -1 # A[2,0]
    coeff_B[8, 11] = 1  # B[2,2]
    coeff_B[7, 11] = 1  # B[2,1]
    
    # p13: (A[0,0] + A[2,2]) * (B[0,0] + B[1,1] + B[2,2])
    coeff_A[0, 12] = 1  # A[0,0]
    coeff_A[8, 12] = 1  # A[2,2]
    coeff_B[0, 12] = 1  # B[0,0]
    coeff_B[4, 12] = 1  # B[1,1]
    coeff_B[8, 12] = 1  # B[2,2]
    
    # p14: (A[0,0] + A[0,1] + A[2,1] + A[2,2]) * (B[0,0] + B[0,1] + B[1,0] + B[1,1])
    coeff_A[0, 13] = 1  # A[0,0]
    coeff_A[1, 13] = 1  # A[0,1]
    coeff_A[7, 13] = 1  # A[2,1]
    coeff_A[8, 13] = 1  # A[2,2]
    coeff_B[0, 13] = 1  # B[0,0]
    coeff_B[1, 13] = 1  # B[0,1]
    coeff_B[3, 13] = 1  # B[1,0]
    coeff_B[4, 13] = 1  # B[1,1]
    
    # p15: (A[0,1] + A[1,0] + A[2,0] + A[2,1]) * (B[1,0] + B[1,1] + B[2,1] + B[2,2])
    coeff_A[1, 14] = 1  # A[0,1]
    coeff_A[3, 14] = 1  # A[1,0]
    coeff_A[6, 14] = 1  # A[2,0]
    coeff_A[7, 14] = 1  # A[2,1]
    coeff_B[3, 14] = 1  # B[1,0]
    coeff_B[4, 14] = 1  # B[1,1]
    coeff_B[7, 14] = 1  # B[2,1]
    coeff_B[8, 14] = 1  # B[2,2]
    
    # Now build the full 6x6x6 tensor decomposition
    # 6x6 matrix = 3x3 blocks of 2x2
    # For each Laderman product c (0..14), and each schoolbook product s (0..3):
    #   U[row, c*4+s] = coeff_A[block_row_idx, c] * U_2x2[row_in_2x2, s]
    #   V[col, c*4+s] = coeff_B[block_col_idx, c] * V_2x2[col_in_2x2, s]
    #   W[out_row, c*4+s] = W_2x2[out_row_2x2, s]
    
    # 2x2 schoolbook rank-4 decomposition:
    # U_2x2[2*i+j, s] = 1 if s == i*2+j
    # V_2x2[2*j+k, s] = 1 if s == i*2+j (same index pattern)
    # W_2x2[2*k+i, s] = 1 if s == i*2+j
    
    # Actually for 2x2 schoolbook:
    # Product s corresponds to (i,j) where s = i*2+j
    # U[row, s] = 1 if row = i*2+j (i.e., A[i,j])
    # V[col, s] = 1 if col = j*2+k (i.e., B[j,k])
    # W[out, s] = 1 if out = k*2+i (i.e., C[k,i] in W convention)
    
    # Wait, let me be more careful with the indexing.
    # For 2x2 matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
    # Schoolbook: 4 products, one for each (i,j,k) but with i,j,k each in {0,1}
    # Actually schoolbook for 2x2 is: C[i,k] = sum_j A[i,j]*B[j,k]
    # This has 4 terms total (2x2), each term is A[i,j]*B[j,k] contributing to C[i,k]
    # So rank 4: for each (i,j) pair, we compute A[i,j]*B[j,k] for all k
    # No wait, that's not right either. Let me think again.
    
    # 2x2 matrix multiplication: C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor has 2*2*2 = 8 entries.
    # Schoolbook decomposition: for each i,j,k in {0,1}:
    #   U[i*2+j, t] = 1  (where t = i*4+j*2+k)
    #   V[j*2+k, t] = 1
    #   W[k*2+i, t] = 1
    # This gives rank 8, not 4.
    
    # Actually, the rank of 2x2 matrix multiplication is 4 (it's known).
    # The schoolbook algorithm uses 4 multiplications:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Rank 4 decomposition:
    # Product 0: A[0,0] * B[0,0] -> contributes to C[0,0]
    # Product 1: A[0,1] * B[1,0] -> contributes to C[0,0]
    # Product 2: A[0,0] * B[0,1] -> contributes to C[0,1]
    # Product 3: A[0,1] * B[1,1] -> contributes to C[0,1]
    # ... wait, that's only for row 0 of C.
    
    # Hmm, I think I'm confusing myself. Let me use the standard Kronecker product approach.
    
    # The 2x2 matrix multiplication tensor can be decomposed with rank 4 as:
    # For (i,j) in {(0,0), (0,1), (1,0), (1,1)}:
    #   U[i*2+j, idx] = 1
    #   V[j*2+(j), idx] = 1  -- No, this isn't right.
    
    # Let me just use the explicit 4-product decomposition:
    # p0 = A[0,0] * B[0,0]
    # p1 = A[0,1] * B[1,0]  
    # p2 = A[1,0] * B[0,0]
    # p3 = A[1,1] * B[1,0]
    # ... no, that only covers B[:,0].
    
    # I think the correct rank-4 decomposition for 2x2 is:
    # p0 = A[0,0] * B[0,0]
    # p1 = A[0,1] * B[1,0]
    # p2 = A[0,0] * B[0,1]
    # p3 = A[0,1] * B[1,1]
    # p4 = A[1,0] * B[0,0]
    # p5 = A[1,1] * B[1,0]
    # p6 = A[1,0] * B[0,1]
    # p7 = A[1,1] * B[1,1]
    # That's rank 8, which is the schoolbook rank for 2x2.
    
    # The minimal rank for 2x2 matrix multiplication is actually 4? No, it's 7 (Strassen).
    # Wait, no. The rank of 2x2 matrix multiplication is 7 (Strassen's algorithm).
    # The schoolbook algorithm uses 4 multiplications for 2x2? No, schoolbook uses 4 for 2x2?
    # 2x2: C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k], so 2*2*2 = 8 scalar multiplications in schoolbook.
    # But Strassen reduces this to 7.
    
    # I made an error. For 2x2 matrix multiplication:
    # Schoolbook: 8 multiplications (rank 8)
    # Strassen: 7 multiplications (rank 7)
    
    # So if I use Strassen for the inner 2x2 blocks, the total rank would be 15 * 7 = 105.
    # If I use schoolbook, it's 15 * 8 = 120.
    
    # Let me use Strassen for the inner blocks to get rank 105.
    
    # Actually, wait. Let me reconsider. The 2x2 matrix multiplication tensor:
    # (A, B, C) where C[i,k] = sum_j A[i,j] B[j,k]
    # Indices: i,k in {0,1}, j in {0,1}
    # 8 nonzero entries in the tensor.
    # Schoolbook decomposition has rank 8.
    # Strassen has rank 7.
    
    # So total rank with Laderman + Strassen inner = 15 * 7 = 105.
    
    # Let me implement this: Laderman (rank 15) at block level, Strassen (rank 7) at inner level.
    # Total rank = 105.
    
    # But first, let me verify the Laderman coefficients are correct by testing.
    
    # For now, let me build the rank-105 solution and verify it.
    
    def build_laderman_strassen():
        rank = 15 * 7
        U = np.zeros((36, rank), dtype=int)
        V = np.zeros((36, rank), dtype=int)
        W = np.zeros((36, rank), dtype=int)
        
        # Strassen for 2x2:
        # 7 products:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B11)
        # M4 = A11*(B01-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B10)
        # M7 = (A01-A11)*(B01+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # U_strassen[2*i+j, m] for the A part
        U_s = np.zeros((4, 7), dtype=int)
        V_s = np.zeros((4, 7), dtype=int)
        W_s = np.zeros((4, 7), dtype=int)
        
        # M1: (A00+A11)*(B00+B11)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1
        
        # M2: (A10+A11)*B00
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[2, 1] = 1; W_s[3, 1] = -1
        
        # M3: A00*(B10-B11)
        U_s[0, 2] = 1
        V_s[2, 2] = 1; V_s[3, 2] = -1
        W_s[1, 2] = 1; W_s[3, 2] = 1
        
        # M4: A11*(B01-B00)
        U_s[3, 3] = 1
        V_s[1, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = 1; W_s[1, 3] = 1
        
        # M5: (A00+A01)*B11
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[0, 4] = -1; W_s[1, 4] = 1
        
        # M6: (A10-A00)*(B00+B10)
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[2, 5] = 1
        W_s[2, 5] = 1
        
        # M7: (A01-A11)*(B01+B11)
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[1, 6] = 1; V_s[3, 6] = 1
        W_s[1, 6] = 1
        
        # Now combine: for each Laderman product c (0..14) and Strassen product s (0..6):
        # The block-level product c involves:
        #   sum_{i,j} coeff_A[i*3+j, c] * A_block[i,j]
        #   sum_{j,k} coeff_B[j*3+k, c] * B_block[j,k]
        #   sum_{k,i} W_coeff[k*3+i, c] * C_block[k,i]
        
        # Wait, I need to also define the W coefficients for Laderman!
        # I forgot to define how the 15 products combine to give the output blocks.
        
        # Let me define the output combination matrix:
        # W_coeff[3*3, 15] where W_coeff[3*k+i, c] is the coefficient of product c in C_block[k,i]
        
        W_coeff = np.zeros((9, 15), dtype=int)
        
        # C[0,0] = p1 + p2 + p3 + p4 - p13 + p14
        W_coeff[0, 0] = 1  # C[0,0] from p1
        W_coeff[0, 1] = 1  # C[0,0] from p2
        W_coeff[0, 2] = 1  # C[0,0] from p3
        W_coeff[0, 3] = 1  # C[0,0] from p4
        W_coeff[0, 12] = -1 # C[0,0] from p13
        W_coeff[0, 13] = 1  # C[0,0] from p14
        
        # C[0,1] = p1 - p2 - p3 + p5 - p6 - p13 + p14
        W_coeff[1, 0] = 1  # C[0,1] from p1
        W_coeff[1, 1] = -1 # C[0,1] from p2
        W_coeff[1, 2] = -1 # C[0,1] from p3
        W_coeff[1, 4] = 1  # C[0,1] from p5
        W_coeff[1, 5] = -1 # C[0,1] from p6
        W_coeff[1, 12] = -1 # C[0,1] from p13
        W_coeff[1, 13] = 1  # C[0,1] from p14
        
        # C[0,2] = p2 + p4 - p7 - p8 - p14
        W_coeff[2, 1] = 1  # C[0,2] from p2
        W_coeff[2, 3] = 1  # C[0,2] from p4
        W_coeff[2, 6] = -1 # C[0,2] from p7
        W_coeff[2, 7] = -1 # C[0,2] from p8
        W_coeff[2, 13] = -1 # C[0,2] from p14
        
        # C[1,0] = p5 + p6 + p7 + p8 - p13
        W_coeff[3, 4] = 1  # C[1,0] from p5
        W_coeff[3, 5] = 1  # C[1,0] from p6
        W_coeff[3, 6] = 1  # C[1,0] from p7
        W_coeff[3, 7] = 1  # C[1,0] from p8
        W_coeff[3, 12] = -1 # C[1,0] from p13
        
        # C[1,1] = p5 - p6 - p7 + p9 - p10 + p13 - p15
        W_coeff[4, 4] = 1  # C[1,1] from p5
        W_coeff[4, 5] = -1 # C[1,1] from p6
        W_coeff[4, 6] = -1 # C[1,1] from p7
        W_coeff[4, 8] = 1  # C[1,1] from p9
        W_coeff[4, 9] = -1 # C[1,1] from p10
        W_coeff[4, 12] = 1  # C[1,1] from p13
        W_coeff[4, 14] = -1 # C[1,1] from p15
        
        # C[1,2] = p7 - p8 - p9 + p10 - p15
        W_coeff[5, 6] = 1  # C[1,2] from p7
        W_coeff[5, 7] = -1 # C[1,2] from p8
        W_coeff[5, 8] = -1 # C[1,2] from p9
        W_coeff[5, 9] = 1  # C[1,2] from p10
        W_coeff[5, 14] = -1 # C[1,2] from p15
        
        # C[2,0] = p9 + p10 + p11 + p12 - p13
        W_coeff[6, 8] = 1  # C[2,0] from p9
        W_coeff[6, 9] = 1  # C[2,0] from p10
        W_coeff[6, 10] = 1 # C[2,0] from p11
        W_coeff[6, 11] = 1 # C[2,0] from p12
        W_coeff[6, 12] = -1 # C[2,0] from p13
        
        # C[2,1] = p9 - p10 - p11 + p12 - p13 - p15
        W_coeff[7, 8] = 1  # C[2,1] from p9
        W_coeff[7, 9] = -1 # C[2,1] from p10
        W_coeff[7, 10] = -1 # C[2,1] from p11
        W_coeff[7, 11] = 1 # C[2,1] from p12
        W_coeff[7, 12] = -1 # C[2,1] from p13
        W_coeff[7, 14] = -1 # C[2,1] from p15
        
        # C[2,2] = p11 - p12 - p13 + p15
        W_coeff[8, 10] = 1 # C[2,2] from p11
        W_coeff[8, 11] = -1 # C[2,2] from p12
        W_coeff[8, 12] = -1 # C[2,2] from p13
        W_coeff[8, 14] = 1 # C[2,2] from p15
        
        # Now build the full decomposition:
        # For each Laderman product c (0..14) and Strassen product s (0..6):
        #   col = c * 7 + s
        #   For each block position (bi, bj) in A (3x3 grid):
        #     if coeff_A[bi*3+bj, c] != 0:
        #       For each inner position (ii, jj) in the 2x2 block:
        #         if U_s[ii*2+jj, s] != 0:
        #           row = (bi*2+ii)*6 + (bj*2+jj)
        #           U[row, col] += coeff_A[bi*3+bj, c] * U_s[ii*2+jj, s]
        #   Similarly for V and W.
        
        for c in range(15):
            for s in range(7):
                col = c * 7 + s
                # U part
                for bi in range(3):
                    for bj in range(3):
                        a_coeff = coeff_A[bi*3+bj, c]
                        if a_coeff != 0:
                            for ii in range(2):
                                for jj in range(2):
                                    u_val = U_s[ii*2+jj, s]
                                    if u_val != 0:
                                        row = (bi*2+ii)*6 + (bj*2+jj)
                                        U[row, col] += a_coeff * u_val
                # V part
                for bj in range(3):
                    for bk in range(3):
                        b_coeff = coeff_B[bj*3+bk, c]
                        if b_coeff != 0:
                            for jj in range(2):
                                for kk in range(2):
                                    v_val = V_s[jj*2+kk, s]
                                    if v_val != 0:
                                        row = (bj*2+jj)*6 + (bk*2+kk)
                                        V[row, col] += b_coeff * v_val
                # W part
                for bk in range(3):
                    for bi in range(3):
                        w_coeff = W_coeff[bk*3+bi, c]
                        if w_coeff != 0:
                            for kk in range(2):
                                for ii in range(2):
                                    w_val = W_s[kk*2+ii, s]
                                    if w_val != 0:
                                        row = (bk*2+kk)*6 + (bi*2+ii)
                                        W[row, col] += w_coeff * w_val
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify_decomposition(result, num_tests=5):
        """Verify that the decomposition correctly computes matrix multiplication."""
        U = np.array(result['U'], dtype=float)
        V = np.array(result['V'], dtype=float)
        W = np.array(result['W'], dtype=float)
        rank = U.shape[1]
        
        # Build the tensor: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
        # This should equal the matrix multiplication tensor:
        # T[i,j,k] = 1 if i = k*6 + j (i.e., C[i] = sum_k A[i,k]*B[k,j])
        # Wait, the convention is: W[k*n+i, r]
        # So for matrix mult C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor is: T[i, j, k] where i indexes A rows, j indexes A cols/B rows, k indexes B cols
        # T[i, j, k] = 1 (since C[i,k] = sum_j A[i,j]*B[j,k])
        
        # U has shape (n*m, rank) = (36, rank)
        # V has shape (m*p, rank) = (36, rank)  
        # W has shape (p*n, rank) = (36, rank)
        
        # U[i*m+j, c] corresponds to A[i,j]
        # V[j*p+k, c] corresponds to B[j,k]
        # W[k*n+i, c] corresponds to C[k,i] ... wait, that doesn't seem right.
        
        # The task says: W[k*n+i, r]
        # So W index = k*n + i, where k is the column of C and i is the row of C?
        # Or k is related to B's column and i to A's row?
        
        # For matrix multiplication C = A*B:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # A has shape (n, m) = (6, 6), indexed as A[i,j] -> flat index i*m+j
        # B has shape (m, p) = (6, 6), indexed as B[j,k] -> flat index j*p+k
        # C has shape (n, p) = (6, 6), indexed as C[i,k] -> flat index... 
        # The task says W[k*n+i, r], so C's flat index is k*n+i where k is column, i is row.
        # So C[i,k] is at position k*n+i in W.
        
        # The tensor decomposition: T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # where a indexes A (flat), b indexes B (flat), c indexes C (flat)
        # T[a, b, c] should be 1 if a, b, c correspond to a valid term in matrix mult.
        # a = i*m+j, b = j'*p+k, c = k'*n+i'
        # T[a,b,c] = 1 if i=i' and j=j' and k=k' (i.e., the term A[i,j]*B[j,k] contributes to C[i,k])
        
        # So T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k in {0..5}
        # And 0 otherwise.
        
        # Let's verify:
        for test in range(num_tests):
            np.random.seed(test)
            A = np.random.randint(-2, 3, size=(6, 6)).astype(float)
            B = np.random.randint(-2, 3, size=(6, 6)).astype(float)
            C_expected = A @ B
            
            # Compute using the decomposition
            A_flat = A.flatten()  # shape (36,)
            B_flat = B.flatten()  # shape (36,)
            
            # C_flat[c_index] = sum_r U[A_index, r] * V[B_index, r] * W[C_index, r]
            # But this is a tensor contraction, not a simple sum.
            # The correct way: C[i,k] = sum_j sum_r U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]
            
            C_computed = np.zeros((6, 6), dtype=float)
            for i in range(6):
                for k in range(6):
                    val = 0.0
                    for j in range(6):
                        for r in range(rank):
                            val += U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]
                    C_computed[i, k] = val
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        
        return True
    
    # Try the Laderman + Strassen decomposition
    try:
        if time.time() - start_time < 120:
            result_105 = build_laderman_strassen()
            if verify_decomposition(result_105, num_tests=3):
                return result_105
    except Exception:
        pass
    
    # Fallback: Rank 189 (2x2 Strassen x 3x3 Schoolbook)
    def build_rank_189():
        # 2x2 Strassen
        U_out = np.zeros((4, 7), dtype=int)
        V_out = np.zeros((4, 7), dtype=int)
        W_out = np.zeros((4, 7), dtype=int)
        
        U_out[0, 0] = 1; U_out[3, 0] = 1
        V_out[0, 0] = 1; V_out[3, 0] = 1
        W_out[0, 0] = 1; W_out[3, 0] = 1
        
        U_out[2, 1] = 1; U_out[3, 1] = 1
        V_out[0, 1] = 1
        W_out[1, 1] = 1; W_out[3, 1] = -1
        
        U_out[0, 2] = 1
        V_out[1, 2] = 1; V_out[3, 2] = -1
        W_out[2, 2] = 1; W_out[3, 2] = 1
        
        U_out[3, 3] = 1
        V_out[2, 3] = 1; V_out[0, 3] = -1
        W_out[0, 3] = 1; W_out[1, 3] = 1
        
        U_out[0, 4] = 1; U_out[1, 4] = 1
        V_out[3, 4] = 1
        W_out[0, 4] = -1; W_out[2, 4] = 1
        
        U_out[2, 5] = 1; U_out[0, 5] = -1
        V_out[0, 5] = 1; V_out[1, 5] = 1
        W_out[3, 5] = 1
        
        U_out[1, 6] = 1; U_out[3, 6] = -1
        V_out[2, 6] = 1; V_out[3, 6] = 1
        W_out[0, 6] = 1
        
        # 3x3 Schoolbook
        U_in = np.zeros((9, 27), dtype=int)
        V_in = np.zeros((9, 27), dtype=int)
        W_in = np.zeros((9, 27), dtype=int)
        
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_in[i * 3 + j, col] = 1
                    V_in[j * 3 + k, col] = 1
                    W_in[k * 3 + i, col] = 1
                    col += 1
                    
        rank = 7 * 27
        U = np.zeros((36, rank), dtype=int)
        V = np.zeros((36, rank), dtype=int)
        W = np.zeros((36, rank), dtype=int)
        
        for c_out in range(7):
            for c_in in range(27):
                c = c_out * 27 + c_in
                
                for r_out in range(4):
                    if U_out[r_out, c_out] != 0:
                        i_out = r_out // 2
                        j_out = r_out % 2
                        val_out = U_out[r_out, c_out]
                        for r_in in range(9):
                            if U_in[r_in, c_in] != 0:
                                i_in = r_in // 3
                                j_in = r_in % 3
                                val_in = U_in[r_in, c_in]
                                r = (i_out * 3 + i_in) * 6 + (j_out * 3 + j_in)
                                U[r, c] += val_out * val_in
                                
                for r_out in range(4):
                    if V_out[r_out, c_out] != 0:
                        j_out = r_out // 2
                        k_out = r_out % 2
                        val_out = V_out[r_out, c_out]
                        for r_in in range(9):
                            if V_in[r_in, c_in] != 0:
                                j_in = r_in // 3
                                k_in = r_in % 3
                                val_in = V_in[r_in, c_in]
                                r = (j_out * 3 + j_in) * 6 + (k_out * 3 + k_in)
                                V[r, c] += val_out * val_in
                                
                for r_out in range(4):
                    if W_out[r_out, c_out] != 0:
                        k_out = r_out // 2
                        i_out = r_out % 2
                        val_out = W_out[r_out, c_out]
                        for r_in in range(9):
                            if W_in[r_in, c_in] != 0:
                                k_in = r_in // 3
                                i_in = r_in % 3
                                val_in = W_in[r_in, c_in]
                                r = (k_out * 3 + k_in) * 6 + (i_out * 3 + i_in)
                                W[r, c] += val_out * val_in
                                
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    result = build_rank_189()
    return result
