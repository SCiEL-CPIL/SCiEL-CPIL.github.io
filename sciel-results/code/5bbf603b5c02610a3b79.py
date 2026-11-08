import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # a_i, a_j are global indices for A block (2x2)
    # b_i, b_j are global indices for B block (2x2)
    # k_offset is the column offset in B (and C)
    def add_strassen_block(a_row, a_col, b_row, k_offset):
        nonlocal col
        # A indices: (a_row, a_col), (a_row, a_col+1), (a_row+1, a_col), (a_row+1, a_col+1)
        # B indices: (b_row, k_offset), (b_row, k_offset+1), (b_row+1, k_offset), (b_row+1, k_offset+1)
        # C indices: (k_offset, a_row), (k_offset+1, a_row), (k_offset, a_row+1), (k_offset+1, a_row+1)
        # Note: W index is k*n + i
        
        # M1 = (A00 + A11)(B00 + B11)
        U[a_row * m + a_col][col] += 1
        U[(a_row + 1) * m + (a_col + 1)][col] += 1
        V[b_row * p + k_offset][col] += 1
        V[(b_row + 1) * p + (k_offset + 1)][col] += 1
        # C00 += M1, C11 += M1
        W[k_offset * n + a_row][col] += 1
        W[(k_offset + 1) * n + (a_row + 1)][col] += 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[(a_row + 1) * m + a_col][col] += 1
        U[(a_row + 1) * m + (a_col + 1)][col] += 1
        V[b_row * p + k_offset][col] += 1
        # C10 += M2
        W[k_offset * n + (a_row + 1)][col] += 1
        col += 1
        
        # M3 = A00(B10 - B11)
        U[a_row * m + a_col][col] += 1
        V[(b_row + 1) * p + k_offset][col] += 1
        V[(b_row + 1) * p + (k_offset + 1)][col] -= 1
        # C01 += M3
        W[(k_offset + 1) * n + a_row][col] += 1
        col += 1
        
        # M4 = A11(B01 - B00)
        U[(a_row + 1) * m + (a_col + 1)][col] += 1
        V[b_row * p + (k_offset + 1)][col] += 1
        V[b_row * p + k_offset][col] -= 1
        # C00 += M4, C10 += M4
        W[k_offset * n + a_row][col] += 1
        W[k_offset * n + (a_row + 1)][col] += 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[a_row * m + a_col][col] += 1
        U[a_row * m + (a_col + 1)][col] += 1
        V[(b_row + 1) * p + (k_offset + 1)][col] += 1
        # C00 -= M5, C01 += M5
        W[k_offset * n + a_row][col] -= 1
        W[(k_offset + 1) * n + a_row][col] += 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B10)
        U[(a_row + 1) * m + a_col][col] += 1
        U[a_row * m + a_col][col] -= 1
        V[b_row * p + k_offset][col] += 1
        V[(b_row + 1) * p + k_offset][col] += 1
        # C11 += M6
        W[(k_offset + 1) * n + (a_row + 1)][col] += 1
        col += 1
        
        # M7 = (A01 - A11)(B01 + B11)
        U[a_row * m + (a_col + 1)][col] += 1
        U[(a_row + 1) * m + (a_col + 1)][col] -= 1
        V[b_row * p + (k_offset + 1)][col] += 1
        V[(b_row + 1) * p + (k_offset + 1)][col] += 1
        # C00 += M7, C11 -= M7
        W[k_offset * n + a_row][col] += 1
        W[(k_offset + 1) * n + (a_row + 1)][col] -= 1
        col += 1

    # Helper to add a trivial 2x2x1 block
    # a_i, a_j are global indices for A block (2x2)
    # b_i is global row index for B block (2x1)
    # k_offset is the column offset in B (and C)
    def add_trivial_221_block(a_row, a_col, b_row, k_offset):
        nonlocal col
        # A00 B00
        U[a_row * m + a_col][col] += 1
        V[b_row * p + k_offset][col] += 1
        W[k_offset * n + a_row][col] += 1
        col += 1
        
        # A01 B10
        U[a_row * m + (a_col + 1)][col] += 1
        V[(b_row + 1) * p + k_offset][col] += 1
        W[k_offset * n + a_row][col] += 1
        col += 1
        
        # A10 B00
        U[(a_row + 1) * m + a_col][col] += 1
        V[b_row * p + k_offset][col] += 1
        W[k_offset * n + (a_row + 1)][col] += 1
        col += 1
        
        # A11 B10
        U[(a_row + 1) * m + (a_col + 1)][col] += 1
        V[(b_row + 1) * p + k_offset][col] += 1
        W[k_offset * n + (a_row + 1)][col] += 1
        col += 1

    # Block 1: A[:, 0:2] * B[0:2, :]
    # A_row = 0, A_col = 0
    # B_row = 0
    # k offsets: 0, 2, 4 for Strassen, 6 for Trivial
    add_strassen_block(0, 0, 0, 0)
    add_strassen_block(0, 0, 0, 2)
    add_strassen_block(0, 0, 0, 4)
    add_trivial_221_block(0, 0, 0, 6)
    
    # Block 2: A[:, 2:4] * B[2:4, :]
    # A_row = 0, A_col = 2
    # B_row = 2
    # k offsets: 0, 2, 4 for Strassen, 6 for Trivial
    add_strassen_block(0, 2, 2, 0)
    add_strassen_block(0, 2, 2, 2)
    add_strassen_block(0, 2, 2, 4)
    add_trivial_221_block(0, 2, 2, 6)
    
    return {"U": U, "V": V, "W": W}
