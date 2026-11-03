import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # The solution is built from 4 blocks of Strassen-like decompositions
    # Each block handles a 2x2 * 2x2 multiplication
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (Added to Block 1)
    # Wait, the standard decomposition splits into 4 independent 2x2 multiplications
    # A0 B0, A1 B2, A0 B1, A1 B3
    # Let's map the indices.
    # A rows 0,1 are A0 (cols 0,1) and A1 (cols 2,3)?
    # A is 2x4.
    # A = [A_left, A_right] where A_left is 2x2, A_right is 2x2.
    # B = [[B_tl, B_tr], [B_bl, B_br]] where each is 2x2.
    # C = [A_left B_tl + A_right B_bl, A_left B_tr + A_right B_br]
    
    # We will use 7 terms for each of the 4 products?
    # No, we can use the structure of the tensor.
    # The known rank 28 solution uses 7 terms for each of the 4 "quadrants" of the output?
    # Actually, the previous code used a specific pattern.
    # Let's replicate the logic that generated the valid rank 28 solution from the prompt.
    
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        # X_rows: indices in U (8)
        # Y_rows: indices in V (16)
        # Z_rows: indices in W (8)
        # col_offset: starting column in U, V, W
        
        # M1: (x0+x3)(y0+y3)
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        
        # M2: (x2+x3)y0
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        
        # M3: x0(y1-y3)
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        
        # M4: x3(y2-y0)
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        
        # M5: (x0+x1)y3
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        
        # M6: (x2-x0)(y0+y1)
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        
        # M7: (x1-x3)(y2+y3)
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    # Define the index mappings for the 4 blocks
    # The mapping in the prompt was:
    # Block 1: X=[0,1,4,5], Y=[0,1,4,5], Z=[0,2,1,3]
    # This corresponds to A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Let's verify indices.
    # A indices: 0,1 (row 0, col 0,1), 4,5 (row 1, col 0,1). So X=[0,1,4,5] is A_left.
    # B indices: 0,1 (row 0, col 0,1), 4,5 (row 1, col 0,1). So Y=[0,1,4,5] is B_tl.
    # C indices: 0,2 (col 0, row 0,1), 1,3 (col 1, row 0,1). So Z=[0,2,1,3] is C_tl.
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # A_right: rows 0,1, cols 2,3. Indices: 2,3 (row 0), 6,7 (row 1). X=[2,3,6,7].
    # B_bl: rows 2,3, cols 0,1. Indices: 8,9 (row 2), 12,13 (row 3). Y=[8,9,12,13].
    # C_tl: Same as before. Z=[0,2,1,3].
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # A_left: X=[0,1,4,5].
    # B_tr: rows 0,1, cols 2,3. Indices: 2,3 (row 0), 6,7 (row 1). Y=[2,3,6,7].
    # C_tr: cols 2,3, rows 0,1. Indices: 4,6 (col 2), 5,7 (col 3). Z=[4,6,5,7].
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # A_right: X=[2,3,6,7].
    # B_br: rows 2,3, cols 2,3. Indices: 10,11 (row 2), 14,15 (row 3). Y=[10,11,14,15].
    # C_tr: Z=[4,6,5,7].

    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    return {"U": U, "V": V, "W": W}
