import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Total rank is 4 * 28 = 112
    total_rank = 112
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to add a 2x2x2 Strassen block
    # a_row, a_col, b_row, b_col, c_row, c_col are offsets in the global matrices
    # start_col is the starting column in U, V, W
    def add_strassen_2x2x2(a_row, a_col, b_row, b_col, c_row, c_col, start_col):
        # A is 2x2, B is 2x2, C is 2x2
        # Indices in A: (a_row + i, a_col + j)
        # Indices in B: (b_row + j, b_col + k)
        # Indices in C: (c_row + i, c_col + k)
        
        # M1 = (A22 + A11)(B11 + B22)
        # A22: (a_row+1, a_col+1), A11: (a_row, a_col)
        # B11: (b_row, b_col), B22: (b_row+1, b_col+1)
        U[a_row * m + a_col + 1][start_col] = 1
        U[a_row * m + a_col][start_col] = 1
        V[(b_row + 0) * p + b_col][start_col] = 1
        V[(b_row + 1) * p + b_col + 1][start_col] = 1
        # C11 += M1
        W[(c_col + 0) * n + c_row + 0][start_col] = 1
        # C22 += M1
        W[(c_col + 1) * n + c_row + 1][start_col] = 1
        
        # M2 = (A21 + A22)B11
        # A21: (a_row+1, a_col), A22: (a_row+1, a_col+1)
        # B11: (b_row, b_col)
        U[a_row * m + a_col][start_col + 1] = 1
        U[a_row * m + a_col + 1][start_col + 1] = 1
        V[b_row * p + b_col][start_col + 1] = 1
        # C21 += M2
        W[(c_col + 0) * n + c_row + 1][start_col + 1] = 1
        # C22 -= M2
        W[(c_col + 1) * n + c_row + 1][start_col + 1] = -1
        
        # M3 = A11(B21 - B22)
        # A11: (a_row, a_col)
        # B21: (b_row+1, b_col), B22: (b_row+1, b_col+1)
        U[a_row * m + a_col][start_col + 2] = 1
        V[(b_row + 1) * p + b_col][start_col + 2] = 1
        V[(b_row + 1) * p + b_col + 1][start_col + 2] = -1
        # C12 += M3
        W[(c_col + 1) * n + c_row + 0][start_col + 2] = 1
        # C22 += M3
        W[(c_col + 1) * n + c_row + 1][start_col + 2] = 1
        
        # M4 = A22(B12 - B11)
        # A22: (a_row+1, a_col+1)
        # B12: (b_row, b_col+1), B11: (b_row, b_col)
        U[a_row * m + a_col + 1][start_col + 3] = 1
        V[b_row * p + b_col + 1][start_col + 3] = 1
        V[b_row * p + b_col][start_col + 3] = -1
        # C11 += M4
        W[(c_col + 0) * n + c_row + 0][start_col + 3] = 1
        # C21 += M4
        W[(c_col + 0) * n + c_row + 1][start_col + 3] = 1
        
        # M5 = (A11 + A12)B22
        # A11: (a_row, a_col), A12: (a_row, a_col+1)
        # B22: (b_row+1, b_col+1)
        U[a_row * m + a_col][start_col + 4] = 1
        U[a_row * m + a_col + 1][start_col + 4] = 1
        V[(b_row + 1) * p + b_col + 1][start_col + 4] = 1
        # C11 -= M5
        W[(c_col + 0) * n + c_row + 0][start_col + 4] = -1
        # C12 += M5
        W[(c_col + 1) * n + c_row + 0][start_col + 4] = 1
        
        # M6 = (A21 - A11)(B11 + B12)
        # A21: (a_row+1, a_col), A11: (a_row, a_col)
        # B11: (b_row, b_col), B12: (b_row, b_col+1)
        U[a_row * m + a_col + 1][start_col + 5] = 1
        U[a_row * m + a_col][start_col + 5] = -1
        V[b_row * p + b_col][start_col + 5] = 1
        V[b_row * p + b_col + 1][start_col + 5] = 1
        # C22 += M6
        W[(c_col + 1) * n + c_row + 1][start_col + 5] = 1
        
        # M7 = (A12 - A22)(B21 + B22)
        # A12: (a_row, a_col+1), A22: (a_row+1, a_col+1)
        # B21: (b_row+1, b_col), B22: (b_row+1, b_col+1)
        U[a_row * m + a_col + 1][start_col + 6] = 1
        U[a_row * m + a_col + 1 + 0][start_col + 6] = 1 # Wait, A12 is (a_row, a_col+1)
        # Correction:
        U[a_row * m + a_col + 1][start_col + 6] = 1 # A12
        U[a_row * m + a_col + 1 + 1][start_col + 6] = -1 # A22 is (a_row+1, a_col+1) -> index a_row+1, a_col+1
        # Let's re-verify indices.
        # A12: row a_row, col a_col+1. Index: a_row*m + a_col+1.
        # A22: row a_row+1, col a_col+1. Index: (a_row+1)*m + a_col+1.
        U[a_row * m + a_col + 1][start_col + 6] = 1
        U[(a_row + 1) * m + a_col + 1][start_col + 6] = -1
        V[(b_row + 1) * p + b_col][start_col + 6] = 1
        V[(b_row + 1) * p + b_col + 1][start_col + 6] = 1
        # C11 += M7
        W[(c_col + 0) * n + c_row + 0][start_col + 6] = 1
        
        return start_col + 7

    # Helper to add a 2x2x1 block (rank 4)
    def add_2x2x1(a_row, a_col, b_row, b_col, c_row, c_col, start_col):
        # A 2x2, B 2x1, C 2x1
        # C1 = A11 B1 + A12 B2
        # C2 = A21 B1 + A22 B2
        
        # P1 = A11 B1
        U[a_row * m + a_col][start_col] = 1
        V[b_row * p + b_col][start_col] = 1
        W[(c_col + 0) * n + c_row + 0][start_col] = 1
        
        # P2 = A12 B2
        U[a_row * m + a_col + 1][start_col + 1] = 1
        V[(b_row + 1) * p + b_col][start_col + 1] = 1
        W[(c_col + 0) * n + c_row + 0][start_col + 1] = 1
        
        # P3 = A21 B1
        U[(a_row + 1) * m + a_col][start_col + 2] = 1
        V[b_row * p + b_col][start_col + 2] = 1
        W[(c_col + 0) * n + c_row + 1][start_col + 2] = 1
        
        # P4 = A22 B2
        U[(a_row + 1) * m + a_col + 1][start_col + 3] = 1
        V[(b_row + 1) * p + b_col][start_col + 3] = 1
        W[(c_col + 0) * n + c_row + 1][start_col + 3] = 1
        
        return start_col + 4

    # Helper to add a 2x1x3 block (rank 6)
    def add_2x1x3(a_row, a_col, b_row, b_col, c_row, c_col, start_col):
        # A 2x1, B 1x3, C 2x3
        # C1j = A1 Bj
        # C2j = A2 Bj
        
        for j in range(3):
            # P = A1 Bj
            U[a_row * m + a_col][start_col + j] = 1
            V[b_row * p + b_col + j][start_col + j] = 1
            W[(c_col + j) * n + c_row + 0][start_col + j] = 1
            
            # P = A2 Bj
            U[(a_row + 1) * m + a_col][start_col + 3 + j] = 1
            V[b_row * p + b_col + j][start_col + 3 + j] = 1
            W[(c_col + j) * n + c_row + 1][start_col + 3 + j] = 1
            
        return start_col + 6

    # Helper to build 2x5x3 block
    def build_2x5x3(a_row_offset, b_col_offset, c_row_offset, c_col_offset, start_col):
        # Local A: 2x5, Local B: 5x3, Local C: 2x3
        # A_L: 2x2 (cols 0-1 of local A)
        # A_R: 2x3 (cols 2-4 of local A)
        # B_L: 2x3 (rows 0-1 of local B)
        # B_R: 3x3 (rows 2-4 of local B)
        
        # Part 1: A_L B_L (2x2x3)
        # Split B_L into B_L1 (2x2) and B_L2 (2x1)
        # B_L1: rows 0-1, cols 0-1 of local B
        # B_L2: rows 0-1, col 2 of local B
        
        # A_L B_L1 (2x2x2)
        # a_row_offset, a_col_offset = 0 (local)
        # b_row_offset, b_col_offset = 0 (local)
        # c_row_offset, c_col_offset = 0 (local)
        start_col = add_strassen_2x2x2(
            a_row_offset, a_col_offset, # A_L is at local (0,0)
            b_row_offset + 0, b_col_offset + 0, # B_L1 is at local (0,0)
            c_row_offset, c_col_offset, # C is at local (0,0)
            start_col
        )
        
        # A_L B_L2 (2x2x1)
        # B_L2 is at local (0, 2)
        start_col = add_2x2x1(
            a_row_offset, a_col_offset, # A_L
            b_row_offset + 0, b_col_offset + 2, # B_L2
            c_row_offset, c_col_offset + 2, # C col 2
            start_col
        )
        
        # Part 2: A_R B_R (2x3x3)
        # A_R: 2x3 (cols 2-4 of local A)
        # B_R: 3x3 (rows 2-4 of local B)
        # Split A_R into A_R1 (2x2) and A_R2 (2x1)
        # A_R1: cols 2-3 of local A
        # A_R2: col 4 of local A
        
        # A_R1 B_R (2x2x3)
        # A_R1 is at local (0, 2)
        # B_R is at local (2, 0)
        # Split B_R into B_R1 (2x2) and B_R2 (2x1)
        # B_R1: rows 2-3, cols 0-1 of local B
        # B_R2: rows 2-3, col 2 of local B
        
        # A_R1 B_R1 (2x2x2)
        start_col = add_strassen_2x2x2(
            a_row_offset, a_col_offset + 2, # A_R1
            b_row_offset + 2, b_col_offset + 0, # B_R1
            c_row_offset, c_col_offset, # C
            start_col
        )
        
        # A_R1 B_R2 (2x2x1)
        start_col = add_2x2x1(
            a_row_offset, a_col_offset + 2, # A_R1
            b_row_offset + 2, b_col_offset + 2, # B_R2
            c_row_offset, c_col_offset + 2, # C col 2
            start_col
        )
        
        # A_R2 B_R (2x1x3)
        # A_R2 is at local (0, 4)
        # B_R is at local (2, 0)
        start_col = add_2x1x3(
            a_row_offset, a_col_offset + 4, # A_R2
            b_row_offset + 2, b_col_offset + 0, # B_R
            c_row_offset, c_col_offset, # C
            start_col
        )
        
        return start_col

    # Build 4 blocks
    current_col = 0
    # Block 1: A[0:2, :], B[:, 0:3] -> C[0:2, 0:3]
    current_col = build_2x5x3(0, 0, 0, 0, current_col)
    
    # Block 2: A[0:2, :], B[:, 3:6] -> C[0:2, 3:6]
    current_col = build_2x5x3(0, 3, 0, 3, current_col)
    
    # Block 3: A[2:4, :], B[:, 0:3] -> C[2:4, 0:3]
    current_col = build_2x5x3(2, 0, 2, 0, current_col)
    
    # Block 4: A[2:4, :], B[:, 3:6] -> C[2:4, 3:6]
    current_col = build_2x5x3(2, 3, 2, 3, current_col)
    
    return {"U": U, "V": V, "W": W}
