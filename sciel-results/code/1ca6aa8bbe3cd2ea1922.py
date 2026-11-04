import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = 3, 4, 5
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_222(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        nonlocal col
        # A block: 2x2 at (a_row_start, a_col_start)
        # B block: 2x2 at (b_row_start, b_col_start)
        # C block: 2x2 at (c_row_start, c_col_start)
        
        # Indices
        # A: (0,0) (0,1) (1,0) (1,1)
        # B: (0,0) (0,1) (1,0) (1,1)
        # C: (0,0) (0,1) (1,0) (1,1)
        
        # M1 = A00 * (B00 + B11)
        U[a_row_start + 0][col] = 1
        V[b_row_start + 0][col] = 1
        W[c_row_start + 0][col] = 1 # C00
        # C11 also gets M1? No, C11 = M1 - M2 + M3 + M5.
        # Wait, C11 is (1,1).
        W[c_row_start + 1 * 2 + 1][col] = 1 # This is wrong. W is (p*n, rank).
        # W index is k*n + i.
        # C is (i, k).
        # C[i, k] corresponds to W[k*n + i].
        
        # Let's clarify W indexing.
        # C is n x p.
        # C[i, k].
        # W[k*n + i, c].
        
        # For C block 2x2 at (c_row_start, c_col_start):
        # C[cr, cc] -> W[cc*n + cr]
        # C[cr, cc+1] -> W[(cc+1)*n + cr]
        # C[cr+1, cc] -> W[cc*n + (cr+1)]
        # C[cr+1, cc+1] -> W[(cc+1)*n + (cr+1)]
        
        # M1 contributes to C00 and C11.
        W[c_col_start * n + c_row_start][col] = 1
        W[(c_col_start + 1) * n + (c_row_start + 1)][col] = 1
        
        # B part: B00 + B11
        # We need to split this into two terms? No, U and V can have sums.
        # But U and V are linear maps.
        # U[..., col] is the coefficient for A.
        # V[..., col] is the coefficient for B.
        # If we want A00 * (B00 + B11), we can set:
        # U[A00] = 1
        # V[B00] = 1, V[B11] = 1
        # This works because multiplication is bilinear.
        
        V[b_row_start + 0][col] = 1
        V[b_row_start + 1][col + 1] = 0 # Wait, col is fixed for this term.
        # We are using one column 'col' for M1.
        # So V[B00] = 1 and V[B11] = 1 in the SAME column.
        V[b_row_start + 0][col] = 1
        V[b_row_start + 1][col] = 1 # B11 is at b_row_start+1, b_col_start+1.
        # B index: b_row * p + b_col.
        # B00: (b_row_start, b_col_start)
        # B11: (b_row_start+1, b_col_start+1)
        
        # Correct V indices:
        V[b_row_start * p + b_col_start][col] = 1
        V[(b_row_start + 1) * p + (b_col_start + 1)][col] = 1
        
        # U index: a_row * m + a_col.
        U[a_row_start * m + a_col_start][col] = 1
        
        col += 1
        
        # M2 = (A00 + A11) * B11
        U[a_row_start * m + a_col_start][col] = 1
        U[(a_row_start + 1) * m + (a_col_start + 1)][col] = 1
        V[(b_row_start + 1) * p + (b_col_start + 1)][col] = 1
        # C11 = ... - M2
        W[(c_col_start + 1) * n + (c_row_start + 1)][col] = -1
        # C21 = M5 + M2
        W[c_col_start * n + (c_row_start + 1)][col] = 1
        col += 1
        
        # M3 = (A00 + A01) * B10
        U[a_row_start * m + a_col_start][col] = 1
        U[a_row_start * m + (a_col_start + 1)][col] = 1
        V[(b_row_start + 1) * p + b_col_start][col] = 1
        # C12 = M3 + M6
        W[(c_col_start + 1) * n + c_row_start][col] = 1
        # C11 = ... + M3
        W[(c_col_start + 1) * n + (c_row_start + 1)][col] = 1
        col += 1
        
        # M4 = A11 * (B01 - B10)
        U[(a_row_start + 1) * m + (a_col_start + 1)][col] = 1
        V[b_row_start * p + (b_col_start + 1)][col] = 1
        V[(b_row_start + 1) * p + b_col_start][col] = -1
        # C11 = ... + M4
        W[(c_col_start + 1) * n + (c_row_start + 1)][col] = 1
        # C00 = M1 + M4 ...
        W[c_col_start * n + c_row_start][col] = 1
        col += 1
        
        # M5 = (A10 - A00) * (B00 + B01)
        U[(a_row_start + 1) * m + a_col_start][col] = 1
        U[a_row_start * m + a_col_start][col] = -1
        V[b_row_start * p + b_col_start][col] = 1
        V[b_row_start * p + (b_col_start + 1)][col] = 1
        # C21 = M5 + M2
        W[c_col_start * n + (c_row_start + 1)][col] = 1
        # C11 = ... + M5
        W[(c_col_start + 1) * n + (c_row_start + 1)][col] = 1
        col += 1
        
        # M6 = (A01 - A11) * (B10 + B11)
        U[a_row_start * m + (a_col_start + 1)][col] = 1
        U[(a_row_start + 1) * m + (a_col_start + 1)][col] = -1
        V[(b_row_start + 1) * p + b_col_start][col] = 1
        V[(b_row_start + 1) * p + (b_col_start + 1)][col] = 1
        # C12 = M3 + M6
        W[(c_col_start + 1) * n + c_row_start][col] = 1
        # C00 = ... - M6
        W[c_col_start * n + c_row_start][col] = -1
        col += 1
        
        # M7 = A00 * B00
        U[a_row_start * m + a_col_start][col] = 1
        V[b_row_start * p + b_col_start][col] = 1
        # C00 = ... + M7
        W[c_col_start * n + c_row_start][col] = 1
        # C11 = ... + M7
        W[(c_col_start + 1) * n + (c_row_start + 1)][col] = 1
        col += 1

    # A_top B_left
    # A_top: rows 0-1, cols 0-3
    # B_left: rows 0-3, cols 0-3
    # C_top: rows 0-1, cols 0-3
    
    # A11 B11 -> C00 (top-left 2x2)
    add_strassen_222(0, 0, 0, 0, 0, 0)
    # A12 B21 -> C00 (top-left 2x2)
    # A12 is A[0:2, 2:4]. B21 is B[2:4, 0:2].
    add_strassen_222(0, 2, 2, 0, 0, 0)
    
    # A11 B12 -> C02 (top-right 2x2)
    # A11 is A[0:2, 0:2]. B12 is B[0:2, 2:4].
    add_strassen_222(0, 0, 0, 2, 0, 2)
    # A12 B22 -> C02 (top-right 2x2)
    # A12 is A[0:2, 2:4]. B22 is B[2:4, 2:4].
    add_strassen_222(0, 2, 2, 2, 0, 2)

    # A_top B_right
    # A_top: rows 0-1, cols 0-3
    # B_right: rows 0-3, col 4
    # C_top: rows 0-1, col 4
    
    # A11 B_r1
    # A11: A[0:2, 0:2]. B_r1: B[0:2, 4].
    # C: C[0:2, 4].
    # Schoolbook 2x2x1
    for i in range(2):
        for j in range(2):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    # A12 B_r2
    # A12: A[0:2, 2:4]. B_r2: B[2:4, 4].
    for i in range(2):
        for j in range(2):
            U[i * m + (j + 2)][col] = 1
            V[(j + 2) * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

    # A_bot B_left
    # A_bot: row 2, cols 0-3
    # B_left: rows 0-3, cols 0-3
    # C_bot: row 2, cols 0-3
    
    # A21 B11
    # A21: A[2:3, 0:2]. B11: B[0:2, 0:2].
    # C: C[2, 0:2].
    for j in range(2):
        for k in range(2):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
            
    # A22 B21
    # A22: A[2:3, 2:4]. B21: B[2:4, 0:2].
    for j in range(2):
        for k in range(2):
            U[2 * m + (j + 2)][col] = 1
            V[(j + 2) * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
            
    # A21 B12
    # A21: A[2:3, 0:2]. B12: B[0:2, 2:4].
    for j in range(2):
        for k in range(2):
            U[2 * m + j][col] = 1
            V[j * p + (k + 2)][col] = 1
            W[(k + 2) * n + 2][col] = 1
            col += 1
            
    # A22 B22
    # A22: A[2:3, 2:4]. B22: B[2:4, 2:4].
    for j in range(2):
        for k in range(2):
            U[2 * m + (j + 2)][col] = 1
            V[(j + 2) * p + (k + 2)][col] = 1
            W[(k + 2) * n + 2][col] = 1
            col += 1

    # A_bot B_right
    # A_bot: row 2, cols 0-3
    # B_right: rows 0-3, col 4
    # C_bot: row 2, col 4
    
    # A21 B_r1
    for j in range(2):
        U[2 * m + j][col] = 1
        V[j * p + 4][col] = 1
        W[4 * n + 2][col] = 1
        col += 1
        
    # A22 B_r2
    for j in range(2):
        U[2 * m + (j + 2)][col] = 1
        V[(j + 2) * p + 4][col] = 1
        W[4 * n + 2][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
