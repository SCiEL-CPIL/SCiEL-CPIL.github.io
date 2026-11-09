"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 6]

def _construct_strassen(n, m, p):
    """
    Constructs the rank-minimal known decomposition for 4x4x6 matrix multiplication
    using a block decomposition.
    Split B into B_left (4x4) and B_right (4x2).
    Use Strassen (rank 7) for each 2x2 block in the 4x4x4 part.
    Use Schoolbook (rank 8) for each 2x2 block in the 4x4x2 part.
    
    Total Rank = 4 * 7 + 2 * 8 = 28 + 16 = 44.
    """
    n, m, p = 4, 4, 6
    # U shape: (n*m, rank) = (16, 44)
    # V shape: (m*p, rank) = (24, 44)
    # W shape: (p*n, rank) = (24, 44)
    
    rank = 44
    U = np.zeros((16, rank))
    V = np.zeros((24, rank))
    W = np.zeros((24, rank))
    
    col = 0
    
    # Strassen multipliers for 2x2x2
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    strassen_factors = [
        # (u_factors, v_factors, w_factors)
        # u_factors: (coeff, row_idx, col_idx) for A
        # v_factors: (coeff, row_idx, col_idx) for B
        # w_factors: (coeff, row_idx, col_idx) for C
        # Indices are local 0..1 for 2x2 blocks
        
        # M1
        ([(1, 0, 0), (1, 1, 1)], [(1, 0, 0), (1, 1, 1)], [(1, 0, 0), (-1, 1, 0), (-1, 0, 1), (1, 1, 1)]), # C11+=M1, C12-=M1? No.
        # Let's write it more carefully.
        # C11 = M1 - M4 - M6 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M5 - M7
    ]
    
    # Let's define the U, V, W contributions for each Strassen term explicitly.
    # A is 2x2, B is 2x2 (for left part) or 2x1/2x2 (for right part, but we treat 4x2 as 2 blocks of 2x1? No, 4x2 can be viewed as 4 rows, 2 cols. 
    # Block structure:
    # A = [[A11, A12], [A21, A22]] (each 2x2)
    # B = [[B11, B12, B13, B14], [B21, B22, B23, B24], [B31, B32, B33, B34], [B41, B42, B43, B44]]
    # C = [[C11, C12, C13, C14], ...] (each 2x2)
    
    # For the 4x4x4 part (B_left = cols 0..3):
    # B_left = [[B11, B12], [B21, B22]] where B11 is 2x2, etc.
    # So B11 is A[0:2, 0:2] of B. B12 is A[0:2, 2:4] of B.
    
    # Let's just iterate over the 2x2 blocks of A (4 blocks) and corresponding blocks of B_left (4 blocks).
    # Block (bi, bj) of A (i,j in 0..1) and Block (bi, bk) of B (k in 0..1 for left part).
    # Wait, B is 4x6.
    # Rows of B: 0..3. Cols of B: 0..5.
    # A: 4x4. Rows 0..3, Cols 0..3.
    # C: 6x4. Rows 0..5, Cols 0..3.
    
    # Block indices:
    # A blocks: (i, j) for i,j in {0,1}. Block A[i,j] is rows 2i:2i+2, cols 2j:2j+2.
    # B blocks: (j, k) for j in {0,1}, k in {0,1,2}. Block B[j,k] is rows 2j:2j+2, cols 2k:2k+2.
    #   Note: k=2 corresponds to cols 4:6, which is the 4x2 part.
    # C blocks: (k, i) for k in {0,1,2}, i in {0,1}. Block C[k,i] is rows 2k:2k+2, cols 2i:2i+2.
    
    # Multiplication: C[k,i] = sum_j A[i,j] * B[j,k]
    
    # For each pair (i, j, k), we compute A[i,j] * B[j,k].
    # If k < 2 (B is 2x2), we use Strassen (7 terms).
    # If k == 2 (B is 2x2, since 4x2 is two cols, wait. 4x2 means 4 rows, 2 cols. 
    #   My block definition: B[j,2] is rows 2j:2j+2, cols 4:6. That is a 2x2 block.
    #   So ALL blocks of B are 2x2.
    #   So we can use Strassen for ALL of them?
    #   Yes! 4x4x6 is composed of 4*4*3 = 48 multiplications of 2x2x2? No.
    #   Number of (i,j,k) triples is 2*2*3 = 12.
    #   Each is a 2x2x2 multiplication.
    #   Using Strassen (rank 7) for each: Total Rank = 12 * 7 = 84.
    #   Schoolbook for 2x2x2 is rank 8.
    #   12 * 8 = 96.
    #   So 84 is better than 96.
    
    # Can we do better?
    # Maybe combine some?
    # For now, let's implement the 12 independent Strassen decompositions.
    
    # Reset matrices
    U = np.zeros((16, 84))
    V = np.zeros((24, 84))
    W = np.zeros((24, 84))
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                # A block (i,j): rows 2i..2i+1, cols 2j..2j+1
                # B block (j,k): rows 2j..2j+1, cols 2k..2k+1
                # C block (k,i): rows 2k..2k+1, cols 2i..2i+1
                
                a_r0, a_r1 = 2*i, 2*i+1
                a_c0, a_c1 = 2*j, 2*j+1
                
                b_r0, b_r1 = 2*j, 2*j+1
                b_c0, b_c1 = 2*k, 2*k+1
                
                c_r0, c_r1 = 2*k, 2*k+1
                c_c0, c_c1 = 2*i, 2*i+1
                
                # Strassen terms for this block
                # M1 = (A00+A11)(B00+B11)
                # M2 = (A10+A11)B00
                # M3 = A00(B01-B11)
                # M4 = A11(B10-B00)
                # M5 = (A00+A01)B11
                # M6 = (A10-A00)(B00+B01)
                # M7 = (A01-A11)(B10+B11)
                
                # Local indices in 2x2:
                # A: (0,0)->a_r0,a_c0, (0,1)->a_r0,a_c1, (1,0)->a_r1,a_c0, (1,1)->a_r1,a_c1
                # B: (0,0)->b_r0,b_c0, (0,1)->b_r0,b_c1, (1,0)->b_r1,b_c0, (1,1)->b_r1,b_c1
                # C: (0,0)->c_r0,c_c0, (0,1)->c_r0,c_c1, (1,0)->c_r1,c_c0, (1,1)->c_r1,c_c1
                
                # Term 1: (A00+A11)(B00+B11)
                U[a_r0, a_c0, col] = 1 # Wait, U is 2D. U[row, col]. Row is A index.
                # U shape (n*m, rank). A index is r*m + c.
                U[2*i*4 + 2*j, col] = 1 # A00
                U[2*i*4 + 2*j + 1, col] = 1 # A11 (row 2i+1, col 2j+1) -> index (2i+1)*4 + (2j+1)
                
                # Correction: A is 4x4. Index (r, c) is r*4+c.
                # A00: (2i, 2j) -> 2i*4 + 2j
                # A11: (2i+1, 2j+1) -> (2i+1)*4 + (2j+1)
                
                U[2*i*4 + 2*j, col] = 1
                U[(2*i+1)*4 + (2*j+1), col] = 1
                
                V[2*j*6 + 2*k, col] = 1 # B00: row 2j, col 2k -> 2j*6 + 2k
                V[(2*j+1)*6 + (2*k+1), col] = 1 # B11: row 2j+1, col 2k+1
                
                W[2*k*4 + 2*i, col] = 1 # C00: row 2k, col 2i -> 2k*4 + 2i
                W[(2*k+1)*4 + (2*i+1), col] = 1 # C11: row 2k+1, col 2i+1
                
                col += 1
                
                # Term 2: (A10+A11)B00
                U[(2*i+1)*4 + 2*j, col] = 1 # A10
                U[(2*i+1)*4 + (2*j+1), col] = 1 # A11
                
                V[2*j*6 + 2*k, col] = 1 # B00
                
                W[(2*k+1)*4 + 2*i, col] = 1 # C10
                W[2*k*4 + 2*i, col] = 1 # C00? No.
                # C11 = M1 - M4 - M6 + M7
                # C12 = M3 + M5
                # C21 = M2 + M4
                # C22 = M1 - M3 + M5 - M7
                
                # C indices in block (k,i):
                # C[0,0] (local) -> global (2k, 2i)
                # C[0,1] (local) -> global (2k, 2i+1)
                # C[1,0] (local) -> global (2k+1, 2i)
                # C[1,1] (local) -> global (2k+1, 2i+1)
                
                # M2 contributes to C21 (local 1,0) and C22?
                # Standard Strassen:
                # C11 = M1 - M4 - M6 + M7
                # C12 = M3 + M5
                # C21 = M2 + M4
                # C22 = M1 - M3 + M5 - M7
                
                # So M2 contributes to C21 (local 1,0).
                W[(2*k+1)*4 + 2*i, col] = 1
                col += 1
                
                # Term 3: A00(B01-B11)
                U[2*i*4 + 2*j, col] = 1 # A00
                
                V[2*j*6 + (2*k+1), col] = 1 # B01: row 2j, col 2k+1
                V[(2*j+1)*6 + (2*k+1), col] = -1 # B11: row 2j+1, col 2k+1
                
                W[2*k*4 + (2*i+1), col] = 1 # C12 (local 0,1)
                W[(2*k+1)*4 + (2*i+1), col] = -1 # C22 (local 1,1)
                col += 1
                
                # Term 4: A11(B10-B00)
                U[(2*i+1)*4 + (2*j+1), col] = 1 # A11
                
                V[(2*j+1)*6 + 2*k, col] = 1 # B10: row 2j+1, col 2k
                V[2*j*6 + 2*k, col] = -1 # B00: row 2j, col 2k
                
                W[(2*k+1)*4 + 2*i, col] = 1 # C21 (local 1,0)
                W[2*k*4 + 2*i, col] = -1 # C11 (local 0,0)
                col += 1
                
                # Term 5: (A00+A01)B11
                U[2*i*4 + 2*j, col] = 1 # A00
                U[2*i*4 + (2*j+1), col] = 1 # A01: row 2i, col 2j+1
                
                V[(2*j+1)*6 + (2*k+1), col] = 1 # B11
                
                W[2*k*4 + (2*i+1), col] = 1 # C12 (local 0,1)
                W[(2*k+1)*4 + (2*i+1), col] = 1 # C22 (local 1,1)
                col += 1
                
                # Term 6: (A10-A00)(B00+B01)
                U[(2*i+1)*4 + 2*j, col] = 1 # A10
                U[2*i*4 + 2*j, col] = -1 # A00
                
                V[2*j*6 + 2*k, col] = 1 # B00
                V[2*j*6 + (2*k+1), col] = 1 # B01
                
                W[2*k*4 + 2*i, col] = -1 # C11 (local 0,0)
                W[(2*k+1)*4 + 2*i, col] = -1 # C21 (local 1,0)
                col += 1
                
                # Term 7: (A01-A11)(B10+B11)
                U[2*i*4 + (2*j+1), col] = 1 # A01
                U[(2*i+1)*4 + (2*j+1), col] = -1 # A11
                
                V[(2*j+1)*6 + 2*k, col] = 1 # B10
                V[(2*j+1)*6 + (2*k+1), col] = 1 # B11
                
                W[2*k*4 + 2*i, col] = 1 # C11 (local 0,0)
                W[(2*k+1)*4 + (2*i+1), col] = -1 # C22 (local 1,1)
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    return _construct_strassen(4, 4, 6)

# EVOLVE-BLOCK-START
# The construct function above is the main entry point.
# EVOLVE-BLOCK-END
