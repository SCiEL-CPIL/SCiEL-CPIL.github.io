"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Rank 28 decomposition for 2x2x8 using Strassen-like formulas
    # For each pair of columns (2k, 2k+1) in B and corresponding 2 rows in C
    # We use 7 terms per 2x2x2 block, but here we have 2x2x8 which decomposes
    # as 4 blocks of 2x2x2, each using Strassen (7 terms) = 28 terms total.
    
    # Strassen formulas for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 2x2x8, we have 4 such blocks (for column pairs 0-1, 2-3, 4-5, 6-7)
    # Each block uses 7 terms, so 28 terms total.
    
    # Additionally, we need to handle the third column of A (column index 2)
    # which multiplies the third row of B to contribute to C.
    # A[0,2] * B[2,:] contributes to C[0,:]
    # A[1,2] * B[2,:] contributes to C[1,:]
    # This requires 2*8 = 16 additional terms (schoolbook for the 2x1x8 part)
    
    # Total rank = 28 + 16 = 44
    
    rank = 44
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    # Define Strassen linear forms
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    strassen_A = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1], # M7: A01 - A11
    ]
    
    strassen_B = [
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [-1, 0, 1, 0], # M4: B10 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1],  # M7: B10 + B11
    ]
    
    strassen_C = [
        [1, 0, 0, 0],   # C00 = M1 + M4 - M5 + M7
        [0, 1, 0, 0],   # C01 = M3 + M5
        [0, 0, 1, 0],   # C10 = M2 + M4
        [0, 0, 0, 1],   # C11 = M1 - M2 + M3 + M6
    ]
    
    # Coefficients for C in terms of M
    # C00 = 1*M1 + 0*M2 + 0*M3 + 1*M4 + (-1)*M5 + 0*M6 + 1*M7
    # C01 = 0*M1 + 0*M2 + 1*M3 + 0*M4 + 1*M5 + 0*M6 + 0*M7
    # C10 = 0*M1 + 1*M2 + 0*M3 + 1*M4 + 0*M5 + 0*M6 + 0*M7
    # C11 = 1*M1 + (-1)*M2 + 1*M3 + 0*M4 + 0*M5 + 1*M6 + 0*M7
    c_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ]
    
    # A global row indices for 2x2 block: A00->0, A01->1, A10->3, A11->4
    a_row_map = [0, 1, 3, 4]
    
    # Fill in the 28 Strassen terms for the 2x2x8 part
    for block in range(4):  # 4 blocks for column pairs (0,1), (2,3), (4,5), (6,7)
        b_col_start = block * 2  # starting column in B
        for r in range(7):  # 7 Strassen terms per block
            term_idx = block * 7 + r  # global term index
            
            # Fill U (A side)
            for local_a in range(4):
                coeff = strassen_A[r][local_a]
                if coeff != 0:
                    global_a_row = a_row_map[local_a]
                    U[global_a_row][term_idx] = coeff
            
            # Fill V (B side)
            # B local indices: 0=B0,col_start, 1=B0,col_start+1, 2=B1,col_start, 3=B1,col_start+1
            for local_b in range(4):
                coeff = strassen_B[r][local_b]
                if coeff != 0:
                    if local_b == 0:  # B0, col_start
                        global_b_row = 0 * p + b_col_start
                    elif local_b == 1:  # B0, col_start+1
                        global_b_row = 0 * p + b_col_start + 1
                    elif local_b == 2:  # B1, col_start
                        global_b_row = 1 * p + b_col_start
                    else:  # local_b == 3: B1, col_start+1
                        global_b_row = 1 * p + b_col_start + 1
                    V[global_b_row][term_idx] = coeff
            
            # Fill W (C side)
            # C local indices: 0=C0,col_start, 1=C0,col_start+1, 2=C1,col_start, 3=C1,col_start+1
            for local_c in range(4):
                coeff = c_coeffs[local_c][r]
                if coeff != 0:
                    if local_c == 0:  # C0, col_start
                        global_c_row = b_col_start * n + 0
                    elif local_c == 1:  # C0, col_start+1
                        global_c_row = (b_col_start + 1) * n + 0
                    elif local_c == 2:  # C1, col_start
                        global_c_row = b_col_start * n + 1
                    else:  # local_c == 3: C1, col_start+1
                        global_c_row = (b_col_start + 1) * n + 1
                    W[global_c_row][term_idx] = coeff
    
    # Fill in the 16 schoolbook terms for the 2x1x8 part (A[:,2] * B[2,:])
    for i in range(2):  # rows of A
        for k in range(8):  # columns of B
            term_idx = 28 + i * 8 + k  # global term index
            
            # U: A[i,2]
            U[i * m + 2][term_idx] = 1
            
            # V: B[2,k]
            V[2 * p + k][term_idx] = 1
            
            # W: C[i,k]
            W[k * n + i][term_idx] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
