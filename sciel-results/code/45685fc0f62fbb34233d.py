import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    # U shape: (6, R)
    # V shape: (24, R)
    # W shape: (16, R)
    
    rank = 44
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Part 1: A_1 B_1 (2x2 by 2x8)
    # A_1 is A[0:2, 0:2]
    # B_1 is B[0:2, 0:8]
    # Split B_1 into 4 blocks of 2x2
    
    for block_idx in range(4):
        # Block covers columns 2*block_idx and 2*block_idx + 1 of B
        b_col_0 = 2 * block_idx
        b_col_1 = 2 * block_idx + 1
        
        # Strassen terms
        # M1: (A00+A11)(B00+B11)
        U[0, col] = 1 # A00
        U[3, col] = 1 # A11 (index 1*2+1 = 3)
        V[0, col] = 1 # B00 (index 0*8+0 = 0)
        V[8, col] = 1 # B11 (index 1*8+1 = 9? No. B is 3x8. B_1 is top 2 rows.
                       # B_1[0, b_col_0] is B[0, b_col_0]. Index 0*8 + b_col_0.
                       # B_1[1, b_col_1] is B[1, b_col_1]. Index 1*8 + b_col_1.
                       # Wait, V index is j*p + k.
                       # j=0, k=b_col_0 -> 0*8 + b_col_0.
                       # j=1, k=b_col_1 -> 1*8 + b_col_1.
        V[0 * p + b_col_0, col] = 1
        V[1 * p + b_col_1, col] = 1
        # W: C00 (index 0*2+0=0), C01 (index 0*2+1=1), C10 (index 1*2+0=2), C11 (index 1*2+1=3)
        # For block 0, C cols are 0,1.
        # For block 1, C cols are 2,3.
        # W index is k*n + i.
        # k is the column index in C.
        # C_00 in block: i=0, k=b_col_0. Index b_col_0*2 + 0.
        # C_01 in block: i=0, k=b_col_1. Index b_col_1*2 + 0.
        # C_10 in block: i=1, k=b_col_0. Index b_col_0*2 + 1.
        # C_11 in block: i=1, k=b_col_1. Index b_col_1*2 + 1.
        
        W[b_col_0 * n + 0, col] = 1
        W[b_col_1 * n + 0, col] = 1
        W[b_col_0 * n + 1, col] = 1
        W[b_col_1 * n + 1, col] = 1
        col += 1
        
        # M2: (A10+A11)B00
        U[2, col] = 1 # A10 (index 1*2+0 = 2)
        U[3, col] = 1 # A11
        V[0 * p + b_col_0, col] = 1 # B00
        W[b_col_0 * n + 1, col] = 1 # C10
        col += 1
        
        # M3: A00(B01-B11)
        U[0, col] = 1 # A00
        V[0 * p + b_col_1, col] = 1 # B01 (j=0, k=b_col_1)
        V[1 * p + b_col_1, col] = -1 # B11 (j=1, k=b_col_1)
        W[b_col_1 * n + 0, col] = 1 # C01
        col += 1
        
        # M4: A11(B10-B00)
        U[3, col] = 1 # A11
        V[1 * p + b_col_0, col] = 1 # B10 (j=1, k=b_col_0)
        V[0 * p + b_col_0, col] = -1 # B00
        W[b_col_0 * n + 0, col] = 1 # C00
        W[b_col_0 * n + 1, col] = 1 # C10
        col += 1
        
        # M5: (A00+A01)B11
        U[0, col] = 1 # A00
        U[1, col] = 1 # A01 (index 0*2+1 = 1)
        V[1 * p + b_col_1, col] = 1 # B11
        W[b_col_0 * n + 0, col] = -1 # C00
        W[b_col_1 * n + 0, col] = 1 # C01
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[2, col] = 1 # A10
        U[0, col] = -1 # A00
        V[0 * p + b_col_0, col] = 1 # B00
        V[0 * p + b_col_1, col] = 1 # B01
        W[b_col_1 * n + 1, col] = 1 # C11
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[1, col] = 1 # A01
        U[3, col] = -1 # A11
        V[1 * p + b_col_0, col] = 1 # B10
        V[1 * p + b_col_1, col] = 1 # B11
        W[b_col_0 * n + 0, col] = 1 # C00
        W[b_col_1 * n + 1, col] = 1 # C11
        col += 1

    # Part 2: A_2 B_2 (2x1 by 1x8)
    # A_2 is A[:, 2] (column 2)
    # B_2 is B[2, :] (row 2)
    # C_ik += A_{i2} B_{2k}
    
    for i in range(n):
        for k in range(p):
            U[i * m + 2, col] = 1 # A_{i2}
            V[2 * p + k, col] = 1 # B_{2k}
            W[k * n + i, col] = 1 # C_{ik}
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
