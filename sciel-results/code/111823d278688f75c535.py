import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # We decompose C = A B into 4 blocks of 2x2 multiplication
    # A is 2x4, split into A0 (cols 0,1) and A1 (cols 2,3)
    # B is 4x4, split into B00 (rows 0,1 cols 0,1), B10 (rows 2,3 cols 0,1), 
    #              B01 (rows 0,1 cols 2,3), B11 (rows 2,3 cols 2,3)
    # C is 2x4, split into C0 (cols 0,1) and C1 (cols 2,3)
    # C0 = A0 B00 + A1 B10
    # C1 = A0 B01 + A1 B11
    
    # Strassen 2x2 multiplication equations for X (2x2), Y (2x2) -> Z (2x2)
    # M1 = (X00 + X11)(Y00 + Y11)
    # M2 = (X10 + X11)Y00
    # M3 = X00(Y01 - Y11)
    # M4 = X11(Y10 - Y00)
    # M5 = (X00 + X01)Y11
    # M6 = (X10 - X00)(Y00 + Y01)
    # M7 = (X01 - X11)(Y10 + Y11)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    def apply_strassen(X_rows, X_cols, Y_rows, Y_cols, Z_rows, Z_cols, offset):
        # X_rows, X_cols are indices in U (flattened A)
        # Y_rows, Y_cols are indices in V (flattened B)
        # Z_rows, Z_cols are indices in W (flattened C)
        # offset is the starting rank index
        
        # M1: (X00+X11)(Y00+Y11)
        U[X_rows[0]*m + X_cols[0]][offset+0] += 1
        U[X_rows[1]*m + X_cols[1]][offset+0] += 1
        V[Y_rows[0]*p + Y_cols[0]][offset+0] += 1
        V[Y_rows[1]*p + Y_cols[1]][offset+0] += 1
        
        # M2: (X10+X11)Y00
        U[X_rows[1]*m + X_cols[0]][offset+1] += 1
        U[X_rows[1]*m + X_cols[1]][offset+1] += 1
        V[Y_rows[0]*p + Y_cols[0]][offset+1] += 1
        
        # M3: X00(Y01-Y11)
        U[X_rows[0]*m + X_cols[0]][offset+2] += 1
        V[Y_rows[0]*p + Y_cols[1]][offset+2] += 1
        V[Y_rows[1]*p + Y_cols[1]][offset+2] -= 1
        
        # M4: X11(Y10-Y00)
        U[X_rows[1]*m + X_cols[1]][offset+3] += 1
        V[Y_rows[1]*p + Y_cols[0]][offset+3] += 1
        V[Y_rows[0]*p + Y_cols[0]][offset+3] -= 1
        
        # M5: (X00+X01)Y11
        U[X_rows[0]*m + X_cols[0]][offset+4] += 1
        U[X_rows[0]*m + X_cols[1]][offset+4] += 1
        V[Y_rows[1]*p + Y_cols[1]][offset+4] += 1
        
        # M6: (X10-X00)(Y00+Y01)
        U[X_rows[1]*m + X_cols[0]][offset+5] += 1
        U[X_rows[0]*m + X_cols[0]][offset+5] -= 1
        V[Y_rows[0]*p + Y_cols[0]][offset+5] += 1
        V[Y_rows[0]*p + Y_cols[1]][offset+5] += 1
        
        # M7: (X01-X11)(Y10+Y11)
        U[X_rows[0]*m + X_cols[1]][offset+6] += 1
        U[X_rows[1]*m + X_cols[1]][offset+6] -= 1
        V[Y_rows[1]*p + Y_cols[0]][offset+6] += 1
        V[Y_rows[1]*p + Y_cols[1]][offset+6] += 1
        
        # W updates
        # Z00 = M1 + M4 - M5 + M7
        w_idx = Z_cols[0]*n + Z_rows[0]
        W[w_idx][offset+0] += 1
        W[w_idx][offset+3] += 1
        W[w_idx][offset+4] -= 1
        W[w_idx][offset+6] += 1
        
        # Z01 = M3 + M5
        w_idx = Z_cols[1]*n + Z_rows[0]
        W[w_idx][offset+2] += 1
        W[w_idx][offset+4] += 1
        
        # Z10 = M2 + M4
        w_idx = Z_cols[0]*n + Z_rows[1]
        W[w_idx][offset+1] += 1
        W[w_idx][offset+3] += 1
        
        # Z11 = M1 - M2 + M3 + M6
        w_idx = Z_cols[1]*n + Z_rows[1]
        W[w_idx][offset+0] += 1
        W[w_idx][offset+1] -= 1
        W[w_idx][offset+2] += 1
        W[w_idx][offset+5] += 1

    # Block 1: A_L * B_LL -> C_L_left
    # A_L: rows 0,1; cols 0,1 => U indices: rows 0,1, cols 0,1
    # B_LL: rows 0,1; cols 0,1 => V indices: rows 0,1, cols 0,1
    # C_L: rows 0,1; cols 0,1 => W indices: rows 0,1, cols 0,1
    apply_strassen([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0)
    
    # Block 2: A_R * B_RL -> C_L_right? No, C_L = A_L B_LL + A_R B_RL
    # So this contribution goes to C_L (cols 0,1)
    # A_R: rows 0,1; cols 2,3 => U indices: rows 0,1, cols 2,3
    # B_RL: rows 2,3; cols 0,1 => V indices: rows 2,3, cols 0,1
    # C_L: rows 0,1; cols 0,1
    apply_strassen([0,1], [2,3], [2,3], [0,1], [0,1], [0,1], 7)
    
    # Block 3: A_L * B_LR -> C_R_left
    # C_R = A_L B_LR + A_R B_RR
    # A_L: rows 0,1; cols 0,1
    # B_LR: rows 0,1; cols 2,3
    # C_R: rows 0,1; cols 2,3
    apply_strassen([0,1], [0,1], [0,1], [2,3], [0,1], [2,3], 14)
    
    # Block 4: A_R * B_RR -> C_R_right
    # A_R: rows 0,1; cols 2,3
    # B_RR: rows 2,3; cols 2,3
    # C_R: rows 0,1; cols 2,3
    apply_strassen([0,1], [2,3], [2,3], [2,3], [0,1], [2,3], 21)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
