import time

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8
    
    # We use the known rank-28 decomposition based on 4 independent Strassen 2x2x2 multiplications.
    # Rank 28 is the current best known upper bound for M_{2,4,4}.
    # It is highly unlikely we can find a rank < 28 decomposition in the given time limit.
    
    rank = 28
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        # Strassen 2x2x2 decomposition
        # Inputs:
        # X_rows: indices in U corresponding to A (2x2 block)
        # Y_rows: indices in V corresponding to B (2x2 block)
        # Z_rows: indices in W corresponding to C (2x2 block)
        # The indices are ordered such that:
        # X[0]=A00, X[1]=A01, X[2]=A10, X[3]=A11
        # Y[0]=B00, Y[1]=B01, Y[2]=B10, Y[3]=B11
        # Z[0]=C00, Z[1]=C01, Z[2]=C10, Z[3]=C11
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        
        # M2 = A10 * (B00 + B01)
        U[X_rows[2]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        V[Y_rows[1]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        
        # M3 = (A00 + A01) * B10
        U[X_rows[0]][col_offset + 2] += 1
        U[X_rows[1]][col_offset + 2] += 1
        V[Y_rows[2]][col_offset + 2] += 1
        W[Z_rows[0]][col_offset + 2] += 1
        W[Z_rows[2]][col_offset + 2] += 1
        
        # M4 = A11 * (B11 - B00)
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[3]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[3]][col_offset + 3] += 1
        W[Z_rows[0]][col_offset + 3] -= 1
        
        # M5 = (A01 - A11) * (B10 + B11)
        U[X_rows[1]][col_offset + 4] += 1
        U[X_rows[3]][col_offset + 4] -= 1
        V[Y_rows[2]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[1]][col_offset + 4] += 1
        W[Z_rows[3]][col_offset + 4] += 1
        
        # M6 = (A00 - A10) * (B01 + B11)
        U[X_rows[0]][col_offset + 5] += 1
        U[X_rows[2]][col_offset + 5] -= 1
        V[Y_rows[1]][col_offset + 5] += 1
        V[Y_rows[3]][col_offset + 5] += 1
        W[Z_rows[1]][col_offset + 5] += 1
        W[Z_rows[2]][col_offset + 5] += 1
        
        # M7 = (A10 + A11) * (B00 - B10)
        U[X_rows[2]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] += 1
        V[Y_rows[0]][col_offset + 6] += 1
        V[Y_rows[2]][col_offset + 6] -= 1
        W[Z_rows[2]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] -= 1

    # Define the 4 blocks for 2x4 * 4x4
    # A is 2x4, split into A_L (2x2) and A_R (2x2)
    # B is 4x4, split into B_TL, B_TR, B_BL, B_BR (all 2x2)
    # C is 2x4, split into C_L (2x2) and C_R (2x2)
    
    # U indices for A:
    # A_L: rows 0,1 (cols 0,1 of A) -> indices 0,1 in U? 
    # U shape is (n*m, rank) = (8, 28).
    # U index = i*m + j.
    # A_L corresponds to i in {0,1}, j in {0,1}. Indices: 0,1,4,5.
    # A_R corresponds to i in {0,1}, j in {2,3}. Indices: 2,3,6,7.
    
    # V indices for B:
    # V shape is (m*p, rank) = (16, 28).
    # V index = j*p + k.
    # B_TL: j in {0,1}, k in {0,1}. Indices: 0,1,4,5.
    # B_TR: j in {0,1}, k in {2,3}. Indices: 2,3,6,7.
    # B_BL: j in {2,3}, k in {0,1}. Indices: 8,9,12,13.
    # B_BR: j in {2,3}, k in {2,3}. Indices: 10,11,14,15.
    
    # W indices for C:
    # W shape is (p*n, rank) = (8, 28).
    # W index = k*n + i.
    # C_L: k in {0,1}, i in {0,1}. Indices: 0,1,2,3.
    #   C[0,0] -> k=0, i=0 -> 0
    #   C[0,1] -> k=1, i=0 -> 1
    #   C[1,0] -> k=0, i=1 -> 2
    #   C[1,1] -> k=1, i=1 -> 3
    # C_R: k in {2,3}, i in {0,1}. Indices: 4,5,6,7.
    #   C[0,2] -> k=2, i=0 -> 4
    #   C[0,3] -> k=3, i=0 -> 5
    #   C[1,2] -> k=2, i=1 -> 6
    #   C[1,3] -> k=3, i=1 -> 7
    
    # Mapping for add_block expects Z_rows = [C00, C01, C10, C11] relative to the 2x2 block.
    # For C_L: [0, 1, 2, 3]
    # For C_R: [4, 5, 6, 7]
    
    # Block 1: A_L * B_TL -> C_L (part 1)
    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 1, 2, 3]
    add_block(0, X1, Y1, Z1)
    
    # Block 2: A_R * B_BL -> C_L (part 2)
    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 1, 2, 3]
    add_block(7, X2, Y2, Z2)
    
    # Block 3: A_L * B_TR -> C_R (part 1)
    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 5, 6, 7]
    add_block(14, X3, Y3, Z3)
    
    # Block 4: A_R * B_BR -> C_R (part 2)
    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 5, 6, 7]
    add_block(21, X4, Y4, Z4)
    
    return {"U": U, "V": V, "W": W}
