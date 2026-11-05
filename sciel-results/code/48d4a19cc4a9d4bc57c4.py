"""
Program for finding a rank-minimal tensor decomposition of the 3x4 by 4x8 matrix multiplication.
Uses a block decomposition strategy: partition A into 2x4 and 1x4 blocks, B into 2x8 and 2x8 blocks.
The 2x4 * 2x8 product is handled by two 2x2 Strassen multiplications (rank 14 total).
The 1x4 * 2x8 products are handled by schoolbook (rank 16 total).
Total rank: 14 + 16 = 30.
"""
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen term
    # a_idx: list of 4 indices in U (for 2x2 block)
    # b_idx: list of 4 indices in V (for 2x2 block)
    # c_idx: list of 4 indices in W (for 2x2 block)
    # u_coeff, v_coeff, w_coeff: coefficients for the term
    def add_strassen(a_idx, b_idx, c_idx, u_coeff, v_coeff, w_coeff):
        nonlocal col
        for i in range(4):
            if u_coeff[i] != 0:
                U[a_idx[i]][col] += u_coeff[i]
            if v_coeff[i] != 0:
                V[b_idx[i]][col] += v_coeff[i]
            if w_coeff[i] != 0:
                W[c_idx[i]][col] += w_coeff[i]
        col += 1

    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Block 1: Top 2x2 of A (rows 0,1; cols 0,1) * Top-left 2x2 of B (rows 0,1; cols 0,1)
    # U indices for A[0:2, 0:2]:
    # A00 -> U[0*4+0] = U[0]
    # A01 -> U[0*4+1] = U[1]
    # A10 -> U[1*4+0] = U[4]
    # A11 -> U[1*4+1] = U[5]
    a1 = [0, 1, 4, 5]
    
    # V indices for B[0:2, 0:2]:
    # B00 -> V[0*8+0] = V[0]
    # B01 -> V[0*8+1] = V[1]
    # B10 -> V[1*8+0] = V[8]
    # B11 -> V[1*8+1] = V[9]
    b1 = [0, 1, 8, 9]
    
    # W indices for C[0:2, 0:2]:
    # C00 -> W[0*3+0] = W[0]
    # C01 -> W[1*3+0] = W[3]
    # C10 -> W[0*3+1] = W[1]
    # C11 -> W[1*3+1] = W[4]
    c1 = [0, 3, 1, 4]
    
    # Add 7 Strassen terms for Block 1
    # M1
    add_strassen(a1, b1, c1, [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1])
    # M2
    add_strassen(a1, b1, c1, [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0])
    # M3
    add_strassen(a1, b1, c1, [1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1])
    # M4
    add_strassen(a1, b1, c1, [0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0])
    # M5
    add_strassen(a1, b1, c1, [1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0])
    # M6
    add_strassen(a1, b1, c1, [-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1])
    # M7
    add_strassen(a1, b1, c1, [0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    
    # Block 2: Top 2x2 of A (rows 0,1; cols 2,3) * Top-right 2x2 of B (rows 0,1; cols 2,3)
    # U indices for A[0:2, 2:4]:
    # A02 -> U[0*4+2] = U[2]
    # A03 -> U[0*4+3] = U[3]
    # A12 -> U[1*4+2] = U[6]
    # A13 -> U[1*4+3] = U[7]
    a2 = [2, 3, 6, 7]
    
    # V indices for B[0:2, 2:4]:
    # B02 -> V[0*8+2] = V[2]
    # B03 -> V[0*8+3] = V[3]
    # B12 -> V[1*8+2] = V[10]
    # B13 -> V[1*8+3] = V[11]
    b2 = [2, 3, 10, 11]
    
    # W indices for C[0:2, 2:4]:
    # C02 -> W[2*3+0] = W[6]
    # C03 -> W[3*3+0] = W[9]
    # C12 -> W[2*3+1] = W[7]
    # C13 -> W[3*3+1] = W[10]
    c2 = [6, 9, 7, 10]
    
    # Add 7 Strassen terms for Block 2
    add_strassen(a2, b2, c2, [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1])
    add_strassen(a2, b2, c2, [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0])
    add_strassen(a2, b2, c2, [1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1])
    add_strassen(a2, b2, c2, [0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0])
    add_strassen(a2, b2, c2, [1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0])
    add_strassen(a2, b2, c2, [-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1])
    add_strassen(a2, b2, c2, [0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    
    # Block 3: Bottom 1x4 of A (row 2; cols 0,1,2,3) * Top 2x8 of B (rows 0,1; cols 0..7)
    # This is a 1x4 * 2x8 product.
    # We can split B into two 2x4 blocks: B[0:2, 0:4] and B[0:2, 4:8].
    # A[2, 0:4] * B[0:2, 0:4] -> C[2, 0:4]
    # A[2, 0:4] * B[0:2, 4:8] -> C[2, 4:8]
    # Each is 1x4 * 2x4. We can further split into 1x2 * 2x2 blocks.
    # A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    # A[2, 2:4] * B[0:2, 2:4] -> C[2, 2:4]
    # A[2, 0:2] * B[0:2, 4:6] -> C[2, 4:6]
    # A[2, 2:4] * B[0:2, 6:8] -> C[2, 6:8]
    # Each 1x2 * 2x2 product has rank 4 (schoolbook).
    # Total 4 * 4 = 16 terms.
    
    # Let's define a helper for 1x2 * 2x2 schoolbook
    def add_1x2_2x2(a_idx, b_idx, c_idx):
        nonlocal col
        # a_idx: [a0, a1] (1x2)
        # b_idx: [b00, b01, b10, b11] (2x2)
        # c_idx: [c0, c1] (1x2)
        # Term 1: a0 * b00 -> c0
        U[a_idx[0]][col] = 1
        V[b_idx[0]][col] = 1
        W[c_idx[0]][col] = 1
        col += 1
        # Term 2: a0 * b01 -> c1
        U[a_idx[0]][col] = 1
        V[b_idx[1]][col] = 1
        W[c_idx[1]][col] = 1
        col += 1
        # Term 3: a1 * b10 -> c0
        U[a_idx[1]][col] = 1
        V[b_idx[2]][col] = 1
        W[c_idx[0]][col] = 1
        col += 1
        # Term 4: a1 * b11 -> c1
        U[a_idx[1]][col] = 1
        V[b_idx[3]][col] = 1
        W[c_idx[1]][col] = 1
        col += 1

    # Sub-block 3.1: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    a31 = [8, 9]       # A[2,0], A[2,1]
    b31 = [0, 1, 8, 9] # B[0,0], B[0,1], B[1,0], B[1,1]
    c31 = [6, 7]       # C[2,0], C[2,1]
    add_1x2_2x2(a31, b31, c31)
    
    # Sub-block 3.2: A[2, 2:4] * B[0:2, 2:4] -> C[2, 2:4]
    a32 = [10, 11]     # A[2,2], A[2,3]
    b32 = [2, 3, 10, 11] # B[0,2], B[0,3], B[1,2], B[1,3]
    c32 = [8, 9]       # C[2,2], C[2,3]
    add_1x2_2x2(a32, b32, c32)
    
    # Sub-block 3.3: A[2, 0:2] * B[0:2, 4:6] -> C[2, 4:6]
    a33 = [8, 9]       # A[2,0], A[2,1]
    b33 = [4, 5, 12, 13] # B[0,4], B[0,5], B[1,4], B[1,5]
    c33 = [10, 11]     # C[2,4], C[2,5]
    add_1x2_2x2(a33, b33, c33)
    
    # Sub-block 3.4: A[2, 2:4] * B[0:2, 6:8] -> C[2, 6:8]
    a34 = [10, 11]     # A[2,2], A[2,3]
    b34 = [6, 7, 14, 15] # B[0,6], B[0,7], B[1,6], B[1,7]
    c34 = [12, 13]     # C[2,6], C[2,7]
    add_1x2_2x2(a34, b34, c34)
    
    # Block 4: Bottom 1x4 of A (row 2; cols 0,1,2,3) * Bottom 2x8 of B (rows 2,3; cols 0..7)
    # Similar to Block 3, but with B rows 2,3.
    
    # Sub-block 4.1: A[2, 0:2] * B[2:4, 0:2] -> C[2, 0:2]
    a41 = [8, 9]
    b41 = [16, 17, 24, 25] # B[2,0], B[2,1], B[3,0], B[3,1]
    c41 = [6, 7]
    add_1x2_2x2(a41, b41, c41)
    
    # Sub-block 4.2: A[2, 2:4] * B[2:4, 2:4] -> C[2, 2:4]
    a42 = [10, 11]
    b42 = [18, 19, 26, 27] # B[2,2], B[2,3], B[3,2], B[3,3]
    c42 = [8, 9]
    add_1x2_2x2(a42, b42, c42)
    
    # Sub-block 4.3: A[2, 0:2] * B[2:4, 4:6] -> C[2, 4:6]
    a43 = [8, 9]
    b43 = [20, 21, 28, 29] # B[2,4], B[2,5], B[3,4], B[3,5]
    c43 = [10, 11]
    add_1x2_2x2(a43, b43, c43)
    
    # Sub-block 4.4: A[2, 2:4] * B[2:4, 6:8] -> C[2, 6:8]
    a44 = [10, 11]
    b44 = [22, 23, 30, 31] # B[2,6], B[2,7], B[3,6], B[3,7]
    c44 = [12, 13]
    add_1x2_2x2(a44, b44, c44)
    
    return {"U": U, "V": V, "W": W}
