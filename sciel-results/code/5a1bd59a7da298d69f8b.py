import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 44
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_1 (2x1) * B_1 (1x8)
    # A_1 is A[:, 0] (indices 0, 1)
    # B_1 is B[0, :] (indices 0..7)
    # C contribution is C[:, 0..7]
    # 16 terms
    for i in range(2): # A row
        for k in range(8): # B col
            U[i][col] = 1
            V[k][col] = 1 # B index: row 0, col k -> 0*8 + k = k
            W[k*n + i][col] = 1
            col += 1
            
    # Part 2: A_2 (2x2) * B_2 (2x8)
    # A_2 is A[:, 1:3] (indices 1, 2)
    # B_2 is B[1:3, :] (rows 1, 2)
    # Split B_2 into 4 blocks of 2 columns
    # Strassen for each 2x2x2 block
    
    # Strassen coefficients
    # M1..M7
    # A indices in local 2x2: 00, 01, 10, 11
    # Global A indices for A_2: 
    # Local 00 -> Global 1 (row 0, col 1)
    # Local 01 -> Global 2 (row 0, col 2)
    # Local 10 -> Global 3 (row 1, col 1)
    # Local 11 -> Global 4 (row 1, col 2)
    
    a_00 = 1
    a_01 = 2
    a_10 = 3
    a_11 = 4
    
    # B indices in local 2x2:
    # For block b (0..3), columns are 2b, 2b+1
    # Global B indices:
    # Local 00 -> Global row 1, col 2b -> 1*8 + 2b
    # Local 01 -> Global row 1, col 2b+1 -> 1*8 + 2b+1
    # Local 10 -> Global row 2, col 2b -> 2*8 + 2b
    # Local 11 -> Global row 2, col 2b+1 -> 2*8 + 2b+1
    
    # C indices in local 2x2:
    # Local 00 -> Global row 0, col 2b -> 2b*n + 0 = 2b*2
    # Local 01 -> Global row 0, col 2b+1 -> 2b*2 + 1
    # Local 10 -> Global row 1, col 2b -> 2b*2 + 2
    # Local 11 -> Global row 1, col 2b+1 -> 2b*2 + 3
    
    for b in range(4):
        # Define local B indices
        b_00 = 1*8 + 2*b
        b_01 = 1*8 + 2*b + 1
        b_10 = 2*8 + 2*b
        b_11 = 2*8 + 2*b + 1
        
        # Define local C indices
        c_00 = 2*b*2 + 0
        c_01 = 2*b*2 + 1
        c_10 = 2*b*2 + 2
        c_11 = 2*b*2 + 3
        
        # M1 = (A00+A11)(B00+B11)
        U[a_00][col] = 1
        U[a_11][col] = 1
        V[b_00][col] = 1
        V[b_11][col] = 1
        W[c_00][col] = 1
        W[c_11][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[a_10][col] = 1
        U[a_11][col] = 1
        V[b_00][col] = 1
        W[c_10][col] = 1
        W[c_11][col] = -1
        col += 1
        
        # M3 = A00(B01-B11)
        U[a_00][col] = 1
        V[b_01][col] = 1
        V[b_11][col] = -1
        W[c_01][col] = 1
        W[c_11][col] = 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[a_11][col] = 1
        V[b_10][col] = 1
        V[b_00][col] = -1
        W[c_00][col] = 1
        W[c_10][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[a_00][col] = 1
        U[a_01][col] = 1
        V[b_11][col] = 1
        W[c_00][col] = -1
        W[c_01][col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a_10][col] = 1
        U[a_00][col] = -1
        V[b_00][col] = 1
        V[b_01][col] = 1
        W[c_11][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a_01][col] = 1
        U[a_11][col] = -1
        V[b_10][col] = 1
        V[b_11][col] = 1
        W[c_00][col] = 1
        W[c_11][col] = -1
        col += 1
        
    return {"U": U, "V": V, "W": W}
