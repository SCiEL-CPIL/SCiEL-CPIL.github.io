import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 88
    # Top part: 2x4x8 using two 2x2x8 Strassen blocks (rank 28 each) -> 56
    # Bottom part: 1x4x8 using schoolbook (rank 32) -> 32
    # Total 88
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, current_col):
        # a_row_offset, a_col_offset: top-left of A block (2x2)
        # b_row_offset, b_col_offset: top-left of B block (2x2)
        # c_row_offset, c_col_offset: top-left of C block (2x2)
        
        # M1 = (A00+A11)(B00+B11)
        U[a_row_offset + a_col_offset + 0][current_col] = 1
        U[a_row_offset + 1 + a_col_offset + 1][current_col] = 1
        V[b_row_offset + b_col_offset + 0][current_col] = 1
        V[b_row_offset + 1 + b_col_offset + 1][current_col] = 1
        W[c_row_offset + c_col_offset + 0][current_col] = 1
        W[c_row_offset + 1 + c_col_offset + 0][current_col] = -1
        W[c_row_offset + c_col_offset + 1][current_col] = -1
        W[c_row_offset + 1 + c_col_offset + 1][current_col] = 1
        current_col += 1
        
        # M2 = (A10+A11)B00
        U[a_row_offset + 1 + a_col_offset + 0][current_col] = 1
        U[a_row_offset + 1 + a_col_offset + 1][current_col] = 1
        V[b_row_offset + b_col_offset + 0][current_col] = 1
        W[c_row_offset + c_col_offset + 1][current_col] = 1
        W[c_row_offset + 1 + c_col_offset + 0][current_col] = 1
        W[c_row_offset + c_col_offset + 0][current_col] = -1
        W[c_row_offset + 1 + c_col_offset + 1][current_col] = -1
        current_col += 1
        
        # M3 = A00(B10-B11)
        U[a_row_offset + a_col_offset + 0][current_col] = 1
        V[b_row_offset + 1 + b_col_offset + 0][current_col] = 1
        V[b_row_offset + 1 + b_col_offset + 1][current_col] = -1
        W[c_row_offset + c_col_offset + 0][current_col] = -1
        W[c_row_offset + 1 + c_col_offset + 0][current_col] = 1
        W[c_row_offset + c_col_offset + 1][current_col] = 1
        W[c_row_offset + 1 + c_col_offset + 1][current_col] = -1
        current_col += 1
        
        # M4 = A11(B01-B00)
        U[a_row_offset + 1 + a_col_offset + 1][current_col] = 1
        V[b_row_offset + b_col_offset + 1][current_col] = 1
        V[b_row_offset + b_col_offset + 0][current_col] = -1
        W[c_row_offset + c_col_offset + 0][current_col] = 1
        W[c_row_offset + 1 + c_col_offset + 0][current_col] = -1
        W[c_row_offset + c_col_offset + 1][current_col] = -1
        W[c_row_offset + 1 + c_col_offset + 1][current_col] = 1
        current_col += 1
        
        # M5 = (A00+A10)B11
        U[a_row_offset + a_col_offset + 0][current_col] = 1
        U[a_row_offset + 1 + a_col_offset + 0][current_col] = 1
        V[b_row_offset + 1 + b_col_offset + 1][current_col] = 1
        W[c_row_offset + c_col_offset + 1][current_col] = 1
        W[c_row_offset + 1 + c_col_offset + 1][current_col] = 1
        W[c_row_offset + c_col_offset + 0][current_col] = -1
        W[c_row_offset + 1 + c_col_offset + 0][current_col] = -1
        current_col += 1
        
        # M6 = (A10-A00)(B00+B10)
        U[a_row_offset + 1 + a_col_offset + 0][current_col] = 1
        U[a_row_offset + a_col_offset + 0][current_col] = -1
        V[b_row_offset + b_col_offset + 0][current_col] = 1
        V[b_row_offset + 1 + b_col_offset + 0][current_col] = 1
        W[c_row_offset + c_col_offset + 0][current_col] = -1
        W[c_row_offset + 1 + c_col_offset + 0][current_col] = 1
        W[c_row_offset + c_col_offset + 1][current_col] = 1
        W[c_row_offset + 1 + c_col_offset + 1][current_col] = -1
        current_col += 1
        
        # M7 = (A01-A11)(B01+B11)
        U[a_row_offset + a_col_offset + 1][current_col] = 1
        U[a_row_offset + 1 + a_col_offset + 1][current_col] = -1
        V[b_row_offset + b_col_offset + 1][current_col] = 1
        V[b_row_offset + 1 + b_col_offset + 1][current_col] = 1
        W[c_row_offset + c_col_offset + 1][current_col] = -1
        W[c_row_offset + 1 + c_col_offset + 1][current_col] = 1
        W[c_row_offset + c_col_offset + 0][current_col] = 1
        W[c_row_offset + 1 + c_col_offset + 0][current_col] = -1
        current_col += 1
        
        return current_col

    # Top Part: A[0:2, :] * B
    # A_top is 2x4. Split into A_TL (0,0)-(1,1) and A_TR (0,2)-(1,3)
    # B is 4x8. Split into B_T (0:2, :) and B_B (2:4, :)
    
    # Term 1: A_TL * B_T
    # A_TL is 2x2 at (0,0). B_T is 2x8 at (0,0).
    # B_T has 4 blocks of 2x2: cols 0-1, 2-3, 4-5, 6-7.
    for b_col in range(0, 8, 2):
        # A_TL (0,0) * B_T_block (0, b_col)
        # C block at (0, b_col)
        col = add_strassen_block(0, 0, 0, b_col, 0, b_col, col)
        
    # Term 2: A_TR * B_B
    # A_TR is 2x2 at (0,2). B_B is 2x8 at (2,0).
    # B_B has 4 blocks of 2x2: cols 0-1, 2-3, 4-5, 6-7.
    for b_col in range(0, 8, 2):
        # A_TR (0,2) * B_B_block (2, b_col)
        # C block at (0, b_col)
        col = add_strassen_block(0, 2, 2, b_col, 0, b_col, col)

    # Bottom Part: A[2, :] * B
    # A_bot is 1x4 at (2,0). B is 4x8 at (0,0).
    # C_bot is 1x8 at (2,0).
    for j in range(4):
        for k in range(8):
            U[2 * 4 + j][col] = 1
            V[j * 8 + k][col] = 1
            W[k * 3 + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
