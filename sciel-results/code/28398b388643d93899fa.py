import numpy as np

def construct():
    n, m, p = 3, 4, 4
    # Total rank will be 4 * 11 = 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 4 blocks
    # Block 0: A1 B11 -> C[:, 0:2]
    # Block 1: A2 B21 -> C[:, 0:2]
    # Block 2: A1 B12 -> C[:, 2:4]
    # Block 3: A2 B22 -> C[:, 2:4]
    
    # For each block, we have 11 terms.
    # Terms 0-6: Strassen for top 2x2 part of A (rows 0,1 of local A)
    # Terms 7-10: Trivial for bottom 1x2 part of A (row 2 of local A)
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B10)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # Local A indices for top 2x2: 0, 1
    # Local B indices: 0, 1, 2, 3 (0:00, 1:01, 2:10, 3:11) -- Wait, standard B is 2x2.
    # Let's use local indices 0..3 for B in the Strassen part.
    # B00 -> 0, B01 -> 1, B10 -> 2, B11 -> 3.
    
    # Strassen U coeffs (for local A rows 0, 1)
    # M1: A00=1, A11=1
    # M2: A10=1, A11=1
    # M3: A00=1
    # M4: A11=1
    # M5: A00=1, A10=1
    # M6: A11=1, A00=-1
    # M7: A01=1, A11=-1
    
    # Note: Local A for 3x2 block has rows 0,1,2.
    # Top part uses rows 0,1.
    # Strassen M1 uses A[0,0] and A[1,1].
    # Strassen M2 uses A[1,0] and A[1,1].
    # Strassen M3 uses A[0,0].
    # Strassen M4 uses A[1,1].
    # Strassen M5 uses A[0,0] and A[1,0].
    # Strassen M6 uses A[1,1] and A[0,0].
    # Strassen M7 uses A[0,1] and A[1,1].
    
    # Let's map local A (3x2) to global A.
    # Local A col 0,1 correspond to the 2 cols of the A block.
    # Local A row 0,1,2 correspond to the 3 rows of the A block.
    
    # We will define a helper to fill the 11 columns for a specific block.
    
    def fill_block(start_col, a_rows, b_rows, w_rows):
        # a_rows: list of 6 global row indices for U (3 rows * 2 cols)
        # b_rows: list of 4 global row indices for V (2 rows * 2 cols)
        # w_rows: list of 6 global row indices for W (3 rows * 2 cols)
        
        # Strassen Part (cols 0-6)
        # M1
        U[a_rows[0]][start_col + 0] += 1 # A[0,0]
        U[a_rows[2]][start_col + 0] += 1 # A[1,1] -- Wait, a_rows[2] is row 1, col 1?
        # a_rows is flattened: [r0c0, r0c1, r1c0, r1c1, r2c0, r2c1]
        # So A[0,0] is a_rows[0], A[0,1] is a_rows[1]
        # A[1,0] is a_rows[2], A[1,1] is a_rows[3]
        # A[2,0] is a_rows[4], A[2,1] is a_rows[5]
        
        # Correction:
        # M1: A[0,0] + A[1,1]
        U[a_rows[0]][start_col + 0] += 1
        U[a_rows[3]][start_col + 0] += 1
        V[b_rows[0]][start_col + 0] += 1 # B[0,0]
        V[b_rows[3]][start_col + 0] += 1 # B[1,1]
        
        # M2: A[1,0] + A[1,1] times B[0,0]
        U[a_rows[2]][start_col + 1] += 1
        U[a_rows[3]][start_col + 1] += 1
        V[b_rows[0]][start_col + 1] += 1
        
        # M3: A[0,0] times (B[1,0] - B[1,1])
        U[a_rows[0]][start_col + 2] += 1
        V[b_rows[2]][start_col + 2] += 1
        V[b_rows[3]][start_col + 2] -= 1
        
        # M4: A[1,1] times (B[0,0] - B[1,0])
        U[a_rows[3]][start_col + 3] += 1
        V[b_rows[0]][start_col + 3] += 1
        V[b_rows[2]][start_col + 3] -= 1
        
        # M5: (A[0,0] + A[1,0]) times B[1,1]
        U[a_rows[0]][start_col + 4] += 1
        U[a_rows[2]][start_col + 4] += 1
        V[b_rows[3]][start_col + 4] += 1
        
        # M6: (A[1,1] - A[0,0]) times (B[0,0] + B[1,0])
        U[a_rows[3]][start_col + 5] += 1
        U[a_rows[0]][start_col + 5] -= 1
        V[b_rows[0]][start_col + 5] += 1
        V[b_rows[2]][start_col + 5] += 1
        
        # M7: (A[0,1] - A[1,1]) times (B[1,1] + B[0,1])
        U[a_rows[1]][start_col + 6] += 1
        U[a_rows[3]][start_col + 6] -= 1
        V[b_rows[3]][start_col + 6] += 1
        V[b_rows[1]][start_col + 6] += 1
        
        # Now map these M's to C[0,0], C[0,1], C[1,0], C[1,1]
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # w_rows mapping:
        # w_rows[0] -> C[0,0]
        # w_rows[1] -> C[0,1]
        # w_rows[2] -> C[1,0]
        # w_rows[3] -> C[1,1]
        # w_rows[4] -> C[2,0]
        # w_rows[5] -> C[2,1]
        
        # Add contributions to W
        # M1
        W[w_rows[0]][start_col + 0] += 1
        W[w_rows[3]][start_col + 0] += 1
        # M4
        W[w_rows[0]][start_col + 3] += 1
        W[w_rows[2]][start_col + 3] += 1
        # M5
        W[w_rows[0]][start_col + 4] -= 1
        W[w_rows[1]][start_col + 4] += 1
        # M7
        W[w_rows[0]][start_col + 6] += 1
        # M3
        W[w_rows[1]][start_col + 2] += 1
        # M2
        W[w_rows[2]][start_col + 1] += 1
        # M6
        W[w_rows[3]][start_col + 5] += 1
        # M3 again for C[1,1]?
        # C[1,1] = M1 - M2 + M3 + M6
        # M1 already added to W[w_rows[3]]
        # -M2
        W[w_rows[3]][start_col + 1] -= 1
        # +M3
        W[w_rows[3]][start_col + 2] += 1
        # +M6 already added to W[w_rows[3]]
        
        # Trivial Part (cols 7-10) for A[2,:] * B
        # C[2,0] = A[2,0]B[0,0] + A[2,1]B[1,0]
        # C[2,1] = A[2,0]B[0,1] + A[2,1]B[1,1]
        
        # Term 7: A[2,0]B[0,0] -> C[2,0]
        U[a_rows[4]][start_col + 7] += 1
        V[b_rows[0]][start_col + 7] += 1
        W[w_rows[4]][start_col + 7] += 1
        
        # Term 8: A[2,1]B[1,0] -> C[2,0]
        U[a_rows[5]][start_col + 8] += 1
        V[b_rows[2]][start_col + 8] += 1
        W[w_rows[4]][start_col + 8] += 1
        
        # Term 9: A[2,0]B[0,1] -> C[2,1]
        U[a_rows[4]][start_col + 9] += 1
        V[b_rows[1]][start_col + 9] += 1
        W[w_rows[5]][start_col + 9] += 1
        
        # Term 10: A[2,1]B[1,1] -> C[2,1]
        U[a_rows[5]][start_col + 10] += 1
        V[b_rows[3]][start_col + 10] += 1
        W[w_rows[5]][start_col + 10] += 1

    # Block 0: A1 B11
    # A1 cols 0,1. Rows 0,1,2.
    a_rows_0 = [i*4 + 0 for i in range(3)] + [i*4 + 1 for i in range(3)]
    # Wait, a_rows needs to be [r0c0, r0c1, r1c0, r1c1, r2c0, r2c1]
    a_rows_0 = [0*4+0, 0*4+1, 1*4+0, 1*4+1, 2*4+0, 2*4+1]
    # B11 rows 0,1. Cols 0,1.
    b_rows_0 = [0*4+0, 0*4+1, 1*4+0, 1*4+1]
    # W rows for C[:, 0:2]
    # C[0,0]->0, C[0,1]->3, C[1,0]->1, C[1,1]->4, C[2,0]->2, C[2,1]->5
    w_rows_0 = [0, 3, 1, 4, 2, 5]
    fill_block(0, a_rows_0, b_rows_0, w_rows_0)

    # Block 1: A2 B21
    # A2 cols 2,3. Rows 0,1,2.
    a_rows_1 = [0*4+2, 0*4+3, 1*4+2, 1*4+3, 2*4+2, 2*4+3]
    # B21 rows 2,3. Cols 0,1.
    b_rows_1 = [2*4+0, 2*4+1, 3*4+0, 3*4+1]
    # W rows same as Block 0
    w_rows_1 = [0, 3, 1, 4, 2, 5]
    fill_block(11, a_rows_1, b_rows_1, w_rows_1)

    # Block 2: A1 B12
    # A1 cols 0,1.
    a_rows_2 = a_rows_0
    # B12 rows 0,1. Cols 2,3.
    b_rows_2 = [0*4+2, 0*4+3, 1*4+2, 1*4+3]
    # W rows for C[:, 2:4]
    # C[0,2]->6, C[0,3]->9, C[1,2]->7, C[1,3]->10, C[2,2]->8, C[2,3]->11
    w_rows_2 = [6, 9, 7, 10, 8, 11]
    fill_block(22, a_rows_2, b_rows_2, w_rows_2)

    # Block 3: A2 B22
    # A2 cols 2,3.
    a_rows_3 = a_rows_1
    # B22 rows 2,3. Cols 2,3.
    b_rows_3 = [2*4+2, 2*4+3, 3*4+2, 3*4+3]
    # W rows same as Block 2
    w_rows_3 = [6, 9, 7, 10, 8, 11]
    fill_block(33, a_rows_3, b_rows_3, w_rows_3)

    return {"U": U, "V": V, "W": W}
