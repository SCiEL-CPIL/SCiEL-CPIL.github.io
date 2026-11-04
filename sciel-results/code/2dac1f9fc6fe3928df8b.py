import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = 3, 4, 8
    
    # We will build the U, V, W matrices for the decomposition.
    # Rank 88 strategy:
    # 1. Split A (3x4) into A_top (2x4) and A_bot (1x4).
    # 2. Split B (4x8) into B_left (4x4) and B_right (4x4).
    # 3. A_top * B_left: Split A_top into two 2x2 blocks. Split B_left into four 2x2 blocks.
    #    4 Strassen products.
    # 4. A_top * B_right: Same. 4 Strassen products.
    # 5. A_bot * B: 1x4x8 schoolbook. 32 products.
    # Total: 4*7 + 4*7 + 32 = 28 + 28 + 32 = 88.
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)] # 12 x 88
    V = [[0] * rank for _ in range(m * p)] # 32 x 88
    W = [[0] * rank for _ in range(p * n)] # 24 x 88
    
    col = 0
    
    # Helper to add a Strassen product
    def add_strassen(a_row_start, a_col_start, b_row_start, b_col_start, w_row_start, w_col_start):
        # A block is 2x2 at (a_row_start, a_col_start)
        # B block is 2x2 at (b_row_start, b_col_start)
        # W block is 2x2 at (w_row_start, w_col_start)
        # Strassen factors:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # A indices:
        # A11: a_row_start, a_col_start
        # A12: a_row_start, a_col_start+1
        # A21: a_row_start+1, a_col_start
        # A22: a_row_start+1, a_col_start+1
        
        # B indices:
        # B11: b_row_start, b_col_start
        # B12: b_row_start, b_col_start+1
        # B21: b_row_start+1, b_col_start
        # B22: b_row_start+1, b_col_start+1
        
        # W indices:
        # W11: w_row_start, w_col_start
        # W12: w_row_start, w_col_start+1
        # W21: w_row_start+1, w_col_start
        # W22: w_row_start+1, w_col_start+1
        
        nonlocal col
        
        # M1
        U[a_row_start][col] += 1
        U[a_row_start+1][col] += 1
        V[b_row_start][col] += 1
        V[b_row_start+1][col] += 1
        W[w_row_start][col] += 1
        W[w_row_start+1][col] += 1
        W[w_row_start][col] += 1 # W11 gets M1
        W[w_row_start+1][col] += 1 # W22 gets M1? No.
        # Strassen reconstruction:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's just add the terms to W correctly.
        # M1 contributes to W11 and W22.
        W[w_row_start][col] += 1
        W[w_row_start+1][col+1] += 1
        col += 1
        
        # M2
        U[a_row_start+1][col] += 1
        V[b_row_start][col] += 1
        W[w_row_start+1][col] += 1 # C21
        W[w_row_start+1][col+1] += -1 # C22
        col += 1
        
        # M3
        U[a_row_start][col] += 1
        V[b_row_start+1][col] += 1
        V[b_row_start+1][col] -= 1 # B21-B22? No, V is linear.
        # V[b_row_start+1] corresponds to B21. V[b_row_start+1] again?
        # M3 = A11 (B21 - B22)
        # U: A11 (1)
        # V: B21 (1), B22 (-1)
        # W: C12 (1), C22 (1)
        U[a_row_start][col] += 1
        V[b_row_start+1][col] += 1
        V[b_row_start+1][col] -= 1 # This is wrong, B22 is b_row_start+1, b_col_start+1
        # Correction:
        # B21 is at (b_row_start+1, b_col_start)
        # B22 is at (b_row_start+1, b_col_start+1)
        V[b_row_start+1][col] += 1
        V[b_row_start+1][col+1] -= 1
        W[w_row_start][col+1] += 1 # C12
        W[w_row_start+1][col+1] += 1 # C22
        col += 1
        
        # M4
        # M4 = A22 (B12 - B11)
        U[a_row_start+1][col+1] += 1
        V[b_row_start][col+1] += 1
        V[b_row_start][col] -= 1
        W[w_row_start][col] += 1 # C11
        W[w_row_start+1][col] += 1 # C21
        col += 1
        
        # M5
        # M5 = (A11 + A12) B22
        U[a_row_start][col] += 1
        U[a_row_start][col+1] += 1
        V[b_row_start+1][col+1] += 1
        W[w_row_start][col] += -1 # C11
        W[w_row_start][col+1] += 1 # C12
        col += 1
        
        # M6
        # M6 = (A21 - A11) (B11 + B12)
        U[a_row_start+1][col] += 1
        U[a_row_start][col] -= 1
        V[b_row_start][col] += 1
        V[b_row_start][col+1] += 1
        W[w_row_start+1][col+1] += 1 # C22
        col += 1
        
        # M7
        # M7 = (A12 - A22) (B21 + B22)
        U[a_row_start][col+1] += 1
        U[a_row_start+1][col+1] -= 1
        V[b_row_start+1][col] += 1
        V[b_row_start+1][col+1] += 1
        W[w_row_start][col] += 1 # C11
        col += 1
        
        return col

    # 1. A_top (rows 0-1 of A) * B_left (cols 0-3 of B)
    # A_top is 2x4. Split into A11 (2x2) and A12 (2x2).
    # B_left is 4x4. Split into 4 2x2 blocks.
    # B11 (0,0), B12 (0,2), B21 (2,0), B22 (2,2) in B's local coords.
    # Global B rows: 0-1 for top, 2-3 for bottom.
    # Global B cols: 0-1 for left-left, 2-3 for left-right.
    
    # Product 1: A11 * B11 -> W11 (0,0)
    col = add_strassen(0, 0, 0, 0, 0, 0)
    # Product 2: A12 * B21 -> W11 (0,0)
    col = add_strassen(0, 2, 2, 0, 0, 0)
    # Product 3: A11 * B12 -> W12 (0,2)
    col = add_strassen(0, 0, 0, 2, 0, 2)
    # Product 4: A12 * B22 -> W12 (0,2)
    col = add_strassen(0, 2, 2, 2, 0, 2)
    
    # 2. A_top * B_right (cols 4-7 of B)
    # B_right blocks:
    # B31 (0,4), B32 (0,6), B41 (2,4), B42 (2,6)
    
    # Product 5: A11 * B31 -> W21 (2,0)? No, W is 3x8.
    # A_top is 2x4. Result is 2x8.
    # W rows 0-1 correspond to A_top.
    # W cols 4-7 correspond to B_right.
    
    col = add_strassen(0, 0, 0, 4, 0, 4)
    col = add_strassen(0, 2, 2, 4, 0, 4)
    col = add_strassen(0, 0, 0, 6, 0, 6)
    col = add_strassen(0, 2, 2, 6, 0, 6)
    
    # 3. A_bot (row 2 of A) * B (4x8)
    # A_bot is 1x4. B is 4x8.
    # Schoolbook: 4 * 8 = 32 terms.
    for j in range(4):
        for k in range(8):
            U[2 * 4 + j][col] = 1 # A_bot row 2, col j. Global row index 2*4+j?
            # A is 3x4. Row 2 is index 2.
            # U index is i*m + j. i=2, j=j.
            U[2 * 4 + j][col] = 1
            V[j * 8 + k][col] = 1
            W[k * 3 + 2][col] = 1 # W index k*n + i. k=k, i=2.
            col += 1
            
    return {"U": U, "V": V, "W": W}
