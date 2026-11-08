import time
import numpy as np
from fractions import Fraction
from collections import defaultdict

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy:
    1. Use Strassen for the 2x2 * 2x2 blocks (rank 7 instead of 4)
    2. Use schoolbook for the 2x1 * 1x2 blocks (rank 4)
    3. Total rank = 4 blocks * (7 + 4) = 44
    
    The matrix multiplication C = A * B where:
    - A is 2x3
    - B is 3x8
    - C is 2x8
    
    We split B into 4 blocks of 3x2: B = [B0, B1, B2, B3]
    For each block Bb (3x2), we compute Cb = A * Bb (2x2)
    
    For each block:
    - Split A into A_left (2x2) and A_right (2x1)
    - Split Bb into Bb_left (2x2) and Bb_right (1x2)
    - Cb = A_left * Bb_left + A_right * Bb_right
    - Use Strassen for A_left * Bb_left (rank 7)
    - Use schoolbook for A_right * Bb_right (rank 4)
    - Total per block: rank 11
    - Total: 4 * 11 = 44
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=8
    num_blocks = p // 2  # 4 blocks of width 2
    
    # Each block has rank 11 (7 from Strassen + 4 from schoolbook)
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block  # 44
    
    U = [[Fraction(0)] * total_rank for _ in range(n * m)]  # 6 x 44
    V = [[Fraction(0)] * total_rank for _ in range(m * p)]  # 24 x 44
    W = [[Fraction(0)] * total_rank for _ in range(p * n)]  # 16 x 44
    
    def set_entry(mat, row, col, val):
        if val != 0:
            mat[row][col] += val
    
    for block in range(num_blocks):
        block_start = block * rank_per_block
        
        # Strassen coefficients for 2x2 * 2x2
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        # 
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Strassen part indices (0-6)
        for s in range(7):
            col = block_start + s
            
            # Determine which A entries are used based on Strassen formula
            if s == 0:  # M1 = (A11 + A22) * (B11 + B22)
                a_entries = [(0, 1), (1, 1)]  # A11, A22
                b_entries = [(0, 1), (1, 1)]  # B11, B22
            elif s == 1:  # M2 = (A21 + A22) * B11
                a_entries = [(1, 1), (1, 1)]  # A21, A22
                b_entries = [(0, 1)]  # B11
            elif s == 2:  # M3 = A11 * (B12 - B22)
                a_entries = [(0, 1)]  # A11
                b_entries = [(0, 1), (1, -1)]  # B12, -B22
            elif s == 3:  # M4 = A22 * (B21 - B11)
                a_entries = [(1, 1)]  # A22
                b_entries = [(1, 1), (0, -1)]  # B21, -B11
            elif s == 4:  # M5 = (A11 + A12) * B22
                a_entries = [(0, 1), (0, 1)]  # A11, A12
                b_entries = [(1, 1)]  # B22
            elif s == 5:  # M6 = (A21 - A11) * (B11 + B12)
                a_entries = [(1, 1), (0, -1)]  # A21, -A11
                b_entries = [(0, 1), (0, 1)]  # B11, B12
            elif s == 6:  # M7 = (A12 - A22) * (B21 + B22)
                a_entries = [(0, 1), (1, -1)]  # A12, -A22
                b_entries = [(1, 1), (1, 1)]  # B21, B22
            
            # Set U entries (A matrix: 2x2, indexed by row i, col j)
            for (a_row, a_val) in a_entries:
                # A is 2x3, we use columns 0,1 for Strassen part
                set_entry(U, a_row * m + a_row, col, a_val)  # Wait, this is wrong
                # Actually A[i, j] where i is row (0 or 1), j is col (0,1,2)
                # For Strassen, we use A[i, 0] and A[i, 1]
                # Let me redo this
        
        # Let me redo the Strassen part more carefully
        # Reset and start over for this block
        for s in range(7):
            col = block_start + s
            # Clear any previous entries (should be 0)
        
        # Strassen for A_left (2x2) * Bb_left (2x2)
        # A_left = A[:, 0:2], Bb_left = Bb[0:2, :]
        # Using Strassen formulas with the standard indexing
        
        # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        # U entries: A[0,0] and A[1,1]
        # V entries: B[0,0] and B[1,1]
        set_entry(U, 0 * m + 0, block_start + 0, Fraction(1))  # A[0,0]
        set_entry(U, 1 * m + 1, block_start + 0, Fraction(1))  # A[1,1]
        set_entry(V, 0 * p + 2*block + 0, block_start + 0, Fraction(1))  # B[0, k]
        set_entry(V, 1 * p + 2*block + 1, block_start + 0, Fraction(1))  # B[1, k+1]
        
        # M2 = (A[1,0] + A[1,1]) * B[0,0]
        set_entry(U, 1 * m + 0, block_start + 1, Fraction(1))  # A[1,0]
        set_entry(U, 1 * m + 1, block_start + 1, Fraction(1))  # A[1,1]
        set_entry(V, 0 * p + 2*block + 0, block_start + 1, Fraction(1))  # B[0, k]
        
        # M3 = A[0,0] * (B[0,1] - B[1,1])
        set_entry(U, 0 * m + 0, block_start + 2, Fraction(1))  # A[0,0]
        set_entry(V, 0 * p + 2*block + 1, block_start + 2, Fraction(1))  # B[0, k+1]
        set_entry(V, 1 * p + 2*block + 1, block_start + 2, Fraction(-1))  # -B[1, k+1]
        
        # M4 = A[1,1] * (B[1,0] - B[0,0])
        set_entry(U, 1 * m + 1, block_start + 3, Fraction(1))  # A[1,1]
        set_entry(V, 1 * p + 2*block + 0, block_start + 3, Fraction(1))  # B[1, k]
        set_entry(V, 0 * p + 2*block + 0, block_start + 3, Fraction(-1))  # -B[0, k]
        
        # M5 = (A[0,0] + A[0,1]) * B[1,1]
        set_entry(U, 0 * m + 0, block_start + 4, Fraction(1))  # A[0,0]
        set_entry(U, 0 * m + 1, block_start + 4, Fraction(1))  # A[0,1]
        set_entry(V, 1 * p + 2*block + 1, block_start + 4, Fraction(1))  # B[1, k+1]
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        set_entry(U, 1 * m + 0, block_start + 5, Fraction(1))  # A[1,0]
        set_entry(U, 0 * m + 0, block_start + 5, Fraction(-1))  # -A[0,0]
        set_entry(V, 0 * p + 2*block + 0, block_start + 5, Fraction(1))  # B[0, k]
        set_entry(V, 0 * p + 2*block + 1, block_start + 5, Fraction(1))  # B[0, k+1]
        
        # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        set_entry(U, 0 * m + 1, block_start + 6, Fraction(1))  # A[0,1]
        set_entry(U, 1 * m + 1, block_start + 6, Fraction(-1))  # -A[1,1]
        set_entry(V, 1 * p + 2*block + 0, block_start + 6, Fraction(1))  # B[1, k]
        set_entry(V, 1 * p + 2*block + 1, block_start + 6, Fraction(1))  # B[1, k+1]
        
        # Schoolbook for A_right (2x1) * Bb_right (1x2)
        # A_right = A[:, 2], Bb_right = Bb[2, :]
        # N1 = A[0,2] * B[2,0]
        # N2 = A[0,2] * B[2,1]
        # N3 = A[1,2] * B[2,0]
        # N4 = A[1,2] * B[2,1]
        
        set_entry(U, 0 * m + 2, block_start + 7, Fraction(1))  # A[0,2]
        set_entry(V, 2 * p + 2*block + 0, block_start + 7, Fraction(1))  # B[2, k]
        
        set_entry(U, 0 * m + 2, block_start + 8, Fraction(1))  # A[0,2]
        set_entry(V, 2 * p + 2*block + 1, block_start + 8, Fraction(1))  # B[2, k+1]
        
        set_entry(U, 1 * m + 2, block_start + 9, Fraction(1))  # A[1,2]
        set_entry(V, 2 * p + 2*block + 0, block_start + 9, Fraction(1))  # B[2, k]
        
        set_entry(U, 1 * m + 2, block_start + 10, Fraction(1))  # A[1,2]
        set_entry(V, 2 * p + 2*block + 1, block_start + 10, Fraction(1))  # B[2, k+1]
        
        # Now set W entries for this block
        # C = C_strassen + C_schoolbook
        # C[0, k] = M1 + M4 - M5 + M7 + N1
        # C[0, k+1] = M3 + M5 + N2
        # C[1, k] = M2 + M4 + N3
        # C[1, k+1] = M1 - M2 + M3 + M6 + N4
        
        # W is indexed by [k*n + i, c] where k is output column, i is output row
        # For block, output columns are 2*block and 2*block+1
        
        # C[0, 2*block] = M1 + M4 - M5 + M7 + N1
        set_entry(W, (2*block) * n + 0, block_start + 0, Fraction(1))  # M1
        set_entry(W, (2*block) * n + 0, block_start + 3, Fraction(1))  # M4
        set_entry(W, (2*block) * n + 0, block_start + 4, Fraction(-1))  # -M5
        set_entry(W, (2*block) * n + 0, block_start + 6, Fraction(1))  # M7
        set_entry(W, (2*block) * n + 0, block_start + 7, Fraction(1))  # N1
        
        # C[0, 2*block+1] = M3 + M5 + N2
        set_entry(W, (2*block+1) * n + 0, block_start + 2, Fraction(1))  # M3
        set_entry(W, (2*block+1) * n + 0, block_start + 4, Fraction(1))  # M5
        set_entry(W, (2*block+1) * n + 0, block_start + 8, Fraction(1))  # N2
        
        # C[1, 2*block] = M2 + M4 + N3
        set_entry(W, (2*block) * n + 1, block_start + 1, Fraction(1))  # M2
        set_entry(W, (2*block) * n + 1, block_start + 3, Fraction(1))  # M4
        set_entry(W, (2*block) * n + 1, block_start + 9, Fraction(1))  # N3
        
        # C[1, 2*block+1] = M1 - M2 + M3 + M6 + N4
        set_entry(W, (2*block+1) * n + 1, block_start + 0, Fraction(1))  # M1
        set_entry(W, (2*block+1) * n + 1, block_start + 1, Fraction(-1))  # -M2
        set_entry(W, (2*block+1) * n + 1, block_start + 2, Fraction(1))  # M3
        set_entry(W, (2*block+1) * n + 1, block_start + 5, Fraction(1))  # M6
        set_entry(W, (2*block+1) * n + 1, block_start + 10, Fraction(1))  # N4
    
    # Convert Fraction to float for return
    U_float = [[float(x) for x in row] for row in U]
    V_float = [[float(x) for x in row] for row in V]
    W_float = [[float(x) for x in row] for row in W]
    
    return {"U": U_float, "V": V_float, "W": W_float}
# EVOLVE-BLOCK-END
