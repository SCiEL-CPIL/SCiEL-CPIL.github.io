"""Matrix multiplication 2x4x8 rank-minimal exact tensor decomposition."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 4, 8]

# Strassen's rank-7 decomposition for 2x2 matrix multiplication
# M1 = (A11 + A22)(B11 + B22)
# M2 = (A21 + A22)B11
# M3 = A11(B12 - B22)
# M4 = A22(B21 - B11)
# M5 = (A11 + A12)B22
# M6 = (A21 - A11)(B11 + B12)
# M7 = (A12 - A22)(B21 + B22)
# C11 = M1 + M4 - M5 + M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M2 + M3 + M6

def construct():
    """
    Construct a rank-56 decomposition for 2x4 x 4x8 matrix multiplication
    using Strassen's algorithm applied to 8 independent 2x2 sub-problems.
    
    The 2x4 matrix A is viewed as a 2x2 block matrix where each block is 1x2.
    The 4x8 matrix B is viewed as a 2x2 block matrix where each block is 2x4.
    Each of the 8 block products (1x2 * 2x4) is a 1x4 result.
    
    Wait, let me reconsider. The standard approach:
    - A is 2x4, B is 4x8, C is 2x8
    - Split A into 2x2 blocks of size 1x2: A11, A12, A21, A22 (each 1x2)
    - Split B into 2x2 blocks of size 2x4: B11, B12, B21, B22 (each 2x4)
    - C11 = A11*B11 + A12*B21 (1x4)
    - C12 = A11*B12 + A12*B22 (1x4)
    - C21 = A21*B11 + A22*B21 (1x4)
    - C22 = A21*B12 + A22*B22 (1x4)
    
    For each of the 8 products (like A11*B11), we can use rank-3 decomposition 
    (since it's 1x2 * 2x4, which is just a dot product with 2 terms, rank 2 actually).
    
    Actually, for 1x2 * 2x4, the rank is 2 (just two scalar multiplications).
    So naive would be 8 * 2 = 16... but that's not right either.
    
    Let me think about this differently. The total rank for 2x4*4x8:
    - Naive: 2*4*8 = 64
    - Using Strassen on 2x2 blocks: 7 * (rank of sub-problem)
    
    The sub-problem for each Strassen product involves multiplying matrices.
    Actually, the cleanest approach: view 2x4*4x8 as 8 independent 2x2*2x2 problems?
    No, that doesn't work directly.
    
    Let me use the approach from inspiration 2: Strassen on 2x2 block structure.
    Rank = 7 * 8 = 56.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # We'll build a rank-56 decomposition
    # The idea: decompose into 8 groups of 7 (Strassen products)
    # Each group handles one column of the output in a Strassen-like fashion
    
    # Actually, let me think more carefully.
    # A is 2x4, B is 4x8, C is 2x8
    # 
    # Split A into left/right halves: A_left (2x2), A_right (2x2)
    # Split B into top/bottom halves: B_top (4x8), B_bottom (4x8)
    # C = A_left * B_top + A_right * B_bottom
    #
    # Now A_left * B_top is 2x2 * 4x8. We can further decompose.
    # Split B_top into left/right: B_top_left (4x4), B_top_right (4x4)
    # A_left * B_top = A_left * B_top_left + A_left * B_top_right
    #
    # This is getting complicated. Let me just use the known rank-56 result.
    
    # The known approach: 2x4*4x8 can be done with rank 56.
    # This is 7 * 8, where 7 is Strassen's rank for 2x2, and 8 accounts for the extra dimensions.
    
    # Let me construct it explicitly.
    # View the problem as: for each of 8 "columns" of output, we do a 2x2 matrix mult.
    # But that's not quite right.
    
    # Alternative: Use the fact that 2x4*4x8 = sum over k=1..4 of A[:,k] * B[k,:]
    # where A[:,k] is 2x1 and B[k,:] is 1x8.
    # Each outer product has rank 1, so total rank 4*2*8... no.
    
    # Let me just implement the Strassen-based approach directly.
    # 
    # The 2x4*4x8 multiplication tensor can be decomposed using 56 rank-1 terms.
    # Here's how: partition the 4 intermediate dimension into two groups of 2.
    # A = [A1 A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x8
    # C = A1*B1 + A2*B2
    #
    # Each A_i*B_i is 2x2 * 2x8. We can further partition B_i into 4 columns of width 2.
    # So A1*B1 = [A1*B1_1, A1*B1_2, A1*B1_3, A1*B1_4] where each B1_j is 2x2
    #
    # For each 2x2*2x2 multiplication, use Strassen (rank 7).
    # Total: 2 (for A1,A2) * 4 (for the 4 column pairs) * 7 (Strassen) = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen's 7 products for 2x2 multiplication:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # For a 2x2 * 2x2 multiplication:
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # In our problem:
    # A is 2x4, B is 4x8
    # We partition A's columns into 2 groups of 2: A = [A_left, A_right] (each 2x2)
    # We partition B's rows into 2 groups of 2: B = [B_top; B_bottom] (each 2x8)
    # C = A_left * B_top + A_right * B_bottom
    #
    # B_top is 2x8, B_bottom is 2x8
    # Partition B_top's columns into 4 groups of 2: B_top = [Bt1, Bt2, Bt3, Bt4] (each 2x2)
    # Partition B_bottom's columns into 4 groups of 2: B_bottom = [Bb1, Bb2, Bb3, Bb4] (each 2x2)
    #
    # C = A_left * [Bt1, Bt2, Bt3, Bt4] + A_right * [Bb1, Bb2, Bb3, Bb4]
    # C = [A_left*Bt1 + A_right*Bb1, A_left*Bt2 + A_right*Bb2, ...]
    #
    # For each column group j (1..4):
    #   C_j = A_left * Bt_j + A_right * Bb_j
    #   Each is a 2x2 * 2x2 multiplication, use Strassen (rank 7)
    #   Total: 4 groups * 7 = 28 for A_left, 4 groups * 7 = 28 for A_right
    #   Total: 56
    
    col = 0
    for side in range(2):  # 0 for A_left, 1 for A_right
        for j in range(4):  # 4 column groups
            # A_side is 2x2, indices in A: row i (0,1), col side*2 + k (k=0,1)
            # Bt_j or Bb_j is 2x2
            # For A_left (side=0): B rows 0,1 (top)
            # For A_right (side=1): B rows 2,3 (bottom)
            b_row_offset = side * 2  # 0 for top, 2 for bottom
            
            for s in range(7):  # 7 Strassen products
                c = col
                col += 1
                
                if s == 0:  # M1 = (A11 + A22)(B11 + B22)
                    # A part: A[0, side*2+0] + A[1, side*2+1]
                    U[0 * m + side * 2 + 0][c] = 1
                    U[1 * m + side * 2 + 1][c] = 1
                    # B part: B[b_row_offset+0, j*2+0] + B[b_row_offset+1, j*2+1]
                    V[(b_row_offset + 0) * p + j * 2 + 0][c] = 1
                    V[(b_row_offset + 1) * p + j * 2 + 1][c] = 1
                    # C part: c11 += 1, c22 += 1
                    W[(j * 2 + 0) * n + 0][c] = 1  # C[0, j*2+0]
                    W[(j * 2 + 1) * n + 1][c] = 1  # C[1, j*2+1]
                    
                elif s == 1:  # M2 = (A21 + A22)B11
                    # A part: A[1, side*2+0] + A[1, side*2+1]
                    U[1 * m + side * 2 + 0][c] = 1
                    U[1 * m + side * 2 + 1][c] = 1
                    # B part: B[b_row_offset+0, j*2+0]
                    V[(b_row_offset + 0) * p + j * 2 + 0][c] = 1
                    # C part: c21 += 1
                    W[(j * 2 + 0) * n + 1][c] = 1  # C[1, j*2+0]
                    
                elif s == 2:  # M3 = A11(B12 - B22)
                    # A part: A[0, side*2+0]
                    U[0 * m + side * 2 + 0][c] = 1
                    # B part: B[b_row_offset+0, j*2+1] - B[b_row_offset+1, j*2+1]
                    V[(b_row_offset + 0) * p + j * 2 + 1][c] = 1
                    V[(b_row_offset + 1) * p + j * 2 + 1][c] = -1
                    # C part: c12 += 1, c22 += 1
                    W[(j * 2 + 1) * n + 0][c] = 1  # C[0, j*2+1]
                    W[(j * 2 + 1) * n + 1][c] = 1  # C[1, j*2+1]
                    
                elif s == 3:  # M4 = A22(B21 - B11)
                    # A part: A[1, side*2+1]
                    U[1 * m + side * 2 + 1][c] = 1
                    # B part: B[b_row_offset+1, j*2+0] - B[b_row_offset+0, j*2+0]
                    V[(b_row_offset + 1) * p + j * 2 + 0][c] = 1
                    V[(b_row_offset + 0) * p + j * 2 + 0][c] = -1
                    # C part: c11 += 1, c21 += 1
                    W[(j * 2 + 0) * n + 0][c] = 1  # C[0, j*2+0]
                    W[(j * 2 + 0) * n + 1][c] = 1  # C[1, j*2+0]
                    
                elif s == 4:  # M5 = (A11 + A12)B22
                    # A part: A[0, side*2+0] + A[0, side*2+1]
                    U[0 * m + side * 2 + 0][c] = 1
                    U[0 * m + side * 2 + 1][c] = 1
                    # B part: B[b_row_offset+1, j*2+1]
                    V[(b_row_offset + 1) * p + j * 2 + 1][c] = 1
                    # C part: c11 -= 1, c12 += 1
                    W[(j * 2 + 0) * n + 0][c] = -1  # C[0, j*2+0]
                    W[(j * 2 + 1) * n + 0][c] = 1  # C[0, j*2+1]
                    
                elif s == 5:  # M6 = (A21 - A11)(B11 + B12)
                    # A part: A[1, side*2+0] - A[0, side*2+0]
                    U[1 * m + side * 2 + 0][c] = 1
                    U[0 * m + side * 2 + 0][c] = -1
                    # B part: B[b_row_offset+0, j*2+0] + B[b_row_offset+0, j*2+1]
                    V[(b_row_offset + 0) * p + j * 2 + 0][c] = 1
                    V[(b_row_offset + 0) * p + j * 2 + 1][c] = 1
                    # C part: c22 += 1
                    W[(j * 2 + 1) * n + 1][c] = 1  # C[1, j*2+1]
                    
                elif s == 6:  # M7 = (A12 - A22)(B21 + B22)
                    # A part: A[0, side*2+1] - A[1, side*2+1]
                    U[0 * m + side * 2 + 1][c] = 1
                    U[1 * m + side * 2 + 1][c] = -1
                    # B part: B[b_row_offset+1, j*2+0] + B[b_row_offset+1, j*2+1]
                    V[(b_row_offset + 1) * p + j * 2 + 0][c] = 1
                    V[(b_row_offset + 1) * p + j * 2 + 1][c] = 1
                    # C part: c11 += 1
                    W[(j * 2 + 0) * n + 0][c] = 1  # C[0, j*2+0]
    
    return {"U": U, "V": V, "W": W}
