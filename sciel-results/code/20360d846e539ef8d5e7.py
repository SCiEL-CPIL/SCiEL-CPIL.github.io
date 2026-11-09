# EVOLVE-BLOCK-START
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=5
    rank = 46
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # A_indices: [a, b, c, d] corresponding to U rows
    # B_indices: [e, f, g, h] corresponding to V rows
    # W_indices: [p, r, q, s] corresponding to W rows (Col 0 Row 0, Col 0 Row 1, Col 1 Row 0, Col 1 Row 1)
    # Note: W index mapping is k*n+i.
    # For 2x2 block at (row_offset, col_offset):
    # C_00 is at (row_offset, col_offset) -> W[col_offset*n + row_offset]
    # C_10 is at (row_offset+1, col_offset) -> W[col_offset*n + row_offset+1]
    # C_01 is at (row_offset, col_offset+1) -> W[(col_offset+1)*n + row_offset]
    # C_11 is at (row_offset+1, col_offset+1) -> W[(col_offset+1)*n + row_offset+1]
    
    def add_strassen(A_idx, B_idx, W_idx, col_offset, row_offset):
        nonlocal col
        # A_idx: [A00, A01, A10, A11]
        # B_idx: [B00, B01, B10, B11]
        # W_idx: [C00, C10, C01, C11]
        
        # M1 = (A00 + A11)(B00 + B11)
        U[A_idx[0], col] = 1; U[A_idx[3], col] = 1
        V[B_idx[0], col] = 1; V[B_idx[3], col] = 1
        W[W_idx[0], col] = 1; W[W_idx[3], col] = 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[A_idx[2], col] = 1; U[A_idx[3], col] = 1
        V[B_idx[0], col] = 1
        W[W_idx[1], col] = 1 # C10
        W[W_idx[3], col] = -1 # C11
        col += 1
        
        # M3 = A00(B01 - B11)
        U[A_idx[0], col] = 1
        V[B_idx[1], col] = 1; V[B_idx[3], col] = -1
        W[W_idx[2], col] = 1 # C01
        W[W_idx[3], col] = 1 # C11
        col += 1
        
        # M4 = A11(B10 - B00)
        U[A_idx[3], col] = 1
        V[B_idx[2], col] = 1; V[B_idx[0], col] = -1
        W[W_idx[0], col] = 1 # C00
        W[W_idx[1], col] = 1 # C10
        col += 1
        
        # M5 = (A00 + A01)B11
        U[A_idx[0], col] = 1; U[A_idx[1], col] = 1
        V[B_idx[3], col] = 1
        W[W_idx[0], col] = -1 # C00
        W[W_idx[2], col] = 1 # C01
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[A_idx[2], col] = 1; U[A_idx[0], col] = -1
        V[B_idx[0], col] = 1; V[B_idx[1], col] = 1
        W[W_idx[3], col] = 1 # C11
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[A_idx[1], col] = 1; U[A_idx[3], col] = -1
        V[B_idx[2], col] = 1; V[B_idx[3], col] = 1
        W[W_idx[0], col] = 1 # C00
        W[W_idx[2], col] = 1 # C01
        col += 1

    def add_schoolbook_2x2x1(A_idx, B_idx, W_idx, col_offset, row_offset):
        # A is 2x2, B is 2x1, C is 2x1
        # C0 = A00*B00 + A01*B10
        # C1 = A10*B00 + A11*B10
        nonlocal col
        # Term 1: A00 B00 -> C0
        U[A_idx[0], col] = 1
        V[B_idx[0], col] = 1
        W[W_idx[0], col] = 1 # C00
        col += 1
        # Term 2: A01 B10 -> C0
        U[A_idx[1], col] = 1
        V[B_idx[1], col] = 1
        W[W_idx[0], col] = 1 # C00
        col += 1
        # Term 3: A10 B00 -> C1
        U[A_idx[2], col] = 1
        V[B_idx[0], col] = 1
        W[W_idx[1], col] = 1 # C10
        col += 1
        # Term 4: A11 B10 -> C1
        U[A_idx[3], col] = 1
        V[B_idx[1], col] = 1
        W[W_idx[1], col] = 1 # C10
        col += 1

    def add_schoolbook_2x1x2(A_idx, B_idx, W_idx, col_offset, row_offset):
        # A is 2x1, B is 1x2, C is 2x2
        # C00 = A00*B00
        # C01 = A00*B01
        # C10 = A10*B00
        # C11 = A10*B01
        nonlocal col
        # Term 1: A00 B00 -> C00
        U[A_idx[0], col] = 1
        V[B_idx[0], col] = 1
        W[W_idx[0], col] = 1 # C00
        col += 1
        # Term 2: A00 B01 -> C01
        U[A_idx[0], col] = 1
        V[B_idx[1], col] = 1
        W[W_idx[2], col] = 1 # C01
        col += 1
        # Term 3: A10 B00 -> C10
        U[A_idx[1], col] = 1
        V[B_idx[0], col] = 1
        W[W_idx[1], col] = 1 # C10
        col += 1
        # Term 4: A10 B01 -> C11
        U[A_idx[1], col] = 1
        V[B_idx[1], col] = 1
        W[W_idx[3], col] = 1 # C11
        col += 1

    def add_schoolbook_2x1x1(A_idx, B_idx, W_idx, col_offset, row_offset):
        # A is 2x1, B is 1x1, C is 2x1
        nonlocal col
        # Term 1: A00 B00 -> C0
        U[A_idx[0], col] = 1
        V[B_idx[0], col] = 1
        W[W_idx[0], col] = 1
        col += 1
        # Term 2: A10 B00 -> C1
        U[A_idx[1], col] = 1
        V[B_idx[0], col] = 1
        W[W_idx[1], col] = 1
        col += 1

    # Define Indices
    # A: 2x5. U indices: i*5+j
    # A0 (cols 0,1): [0, 1, 5, 6] -> Wait, U is (n*m, rank). Row index i*m+j.
    # A is 2x5. Rows 0,1. Cols 0..4.
    # U[i*5+j].
    # A0 (cols 0,1):
    #   A00 (0,0): 0
    #   A01 (0,1): 1
    #   A10 (1,0): 5
    #   A11 (1,1): 6
    A0_idx = [0, 1, 5, 6]
    
    # A1 (cols 2,3):
    #   A00 (0,2): 2
    #   A01 (0,3): 3
    #   A10 (1,2): 7
    #   A11 (1,3): 8
    A1_idx = [2, 3, 7, 8]
    
    # A2 (col 4):
    #   A00 (0,4): 4
    #   A10 (1,4): 9
    A2_idx = [4, 9]

    # B: 5x5. V indices: j*5+k.
    # B0 (rows 0,1):
    #   B00 (0,0): 0, B01 (0,1): 1, B10 (1,0): 5, B11 (1,1): 6
    B00_idx = [0, 1, 5, 6]
    #   B00 (0,2): 2, B01 (0,3): 3, B10 (1,2): 7, B11 (1,3): 8
    B01_idx = [2, 3, 7, 8]
    #   B00 (0,4): 4, B10 (1,4): 9
    B02_idx = [4, 9]
    
    # B1 (rows 2,3):
    #   B00 (2,0): 10, B01 (2,1): 11, B10 (3,0): 15, B11 (3,1): 16
    B10_idx = [10, 11, 15, 16]
    #   B00 (2,2): 12, B01 (2,3): 13, B10 (3,2): 17, B11 (3,3): 18
    B11_idx = [12, 13, 17, 18]
    #   B00 (2,4): 14, B10 (3,4): 19
    B12_idx = [14, 19]
    
    # B2 (row 4):
    #   B00 (4,0): 20, B01 (4,1): 21
    B20_idx = [20, 21]
    #   B00 (4,2): 22, B01 (4,3): 23
    B21_idx = [22, 23]
    #   B00 (4,4): 24
    B22_idx = [24]

    # C: 2x5. W indices: k*2+i.
    # C0 (cols 0,1):
    #   C00 (0,0): 0, C10 (0,1): 1, C01 (1,0): 2, C11 (1,1): 3
    C00_idx = [0, 1, 2, 3]
    #   C00 (0,2): 4, C10 (0,3): 5, C01 (1,2): 6, C11 (1,3): 7
    C01_idx = [4, 5, 6, 7]
    #   C00 (0,4): 8, C10 (1,4): 9
    C02_idx = [8, 9]

    # Block 1: A0 B00
    add_strassen(A0_idx, B00_idx, C00_idx, 0, 0)
    # Block 2: A0 B01
    add_strassen(A0_idx, B01_idx, C01_idx, 2, 0)
    # Block 3: A0 B02
    add_schoolbook_2x2x1(A0_idx, B02_idx, C02_idx, 4, 0)
    
    # Block 4: A1 B10
    add_strassen(A1_idx, B10_idx, C00_idx, 0, 0)
    # Block 5: A1 B11
    add_strassen(A1_idx, B11_idx, C01_idx, 2, 0)
    # Block 6: A1 B12
    add_schoolbook_2x2x1(A1_idx, B12_idx, C02_idx, 4, 0)
    
    # Block 7: A2 B20
    add_schoolbook_2x1x2(A2_idx, B20_idx, C00_idx, 0, 0)
    # Block 8: A2 B21
    add_schoolbook_2x1x2(A2_idx, B21_idx, C01_idx, 2, 0)
    # Block 9: A2 B22
    add_schoolbook_2x1x1(A2_idx, B22_idx, C02_idx, 4, 0)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
