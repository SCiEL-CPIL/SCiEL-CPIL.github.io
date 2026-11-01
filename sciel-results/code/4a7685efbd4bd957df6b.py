import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (X11+X22)(Y11+Y22)
    # M2 = (X21+X22)Y11
    # M3 = X11(Y12-Y22)
    # M4 = X22(Y21-Y11)
    # M5 = (X11+X12)Y22
    # M6 = (X21-X11)(Y11+Y12)
    # M7 = (X12-X22)(Y21+Y22)
    #
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # Define the 4 blocks
    # Block 1: A_L * B_LL -> C_L
    # Block 2: A_R * B_RL -> C_L
    # Block 3: A_L * B_LR -> C_R
    # Block 4: A_R * B_RR -> C_R
    
    # Helper to add a Strassen block
    def add_strassen_block(block_id, A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        # A_rows: list of 2 row indices for A (local 0,1 map to these)
        # A_cols: list of 2 col indices for A
        # B_rows: list of 2 row indices for B
        # B_cols: list of 2 col indices for B
        # C_rows: list of 2 row indices for C (output)
        # C_cols: list of 2 col indices for C (output)
        
        base_col = block_id * 7
        
        # Map local indices (0,1) to global indices
        # X is A, Y is B
        # X11 = A[A_rows[0], A_cols[0]]
        # X12 = A[A_rows[0], A_cols[1]]
        # X21 = A[A_rows[1], A_cols[0]]
        # X22 = A[A_rows[1], A_cols[1]]
        
        # Y11 = B[B_rows[0], B_cols[0]]
        # Y12 = B[B_rows[0], B_cols[1]]
        # Y21 = B[B_rows[1], B_cols[0]]
        # Y22 = B[B_rows[1], B_cols[1]]
        
        # M1: (X11+X22)(Y11+Y22)
        col = base_col + 0
        U[A_rows[0] * m + A_cols[0]][col] += 1
        U[A_rows[1] * m + A_cols[1]][col] += 1
        V[B_rows[0] * p + B_cols[0]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] += 1
        
        # M2: (X21+X22)Y11
        col = base_col + 1
        U[A_rows[1] * m + A_cols[0]][col] += 1
        U[A_rows[1] * m + A_cols[1]][col] += 1
        V[B_rows[0] * p + B_cols[0]][col] += 1
        
        # M3: X11(Y12-Y22)
        col = base_col + 2
        U[A_rows[0] * m + A_cols[0]][col] += 1
        V[B_rows[0] * p + B_cols[1]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] -= 1
        
        # M4: X22(Y21-Y11)
        col = base_col + 3
        U[A_rows[1] * m + A_cols[1]][col] += 1
        V[B_rows[1] * p + B_cols[0]][col] += 1
        V[B_rows[0] * p + B_cols[0]][col] -= 1
        
        # M5: (X11+X12)Y22
        col = base_col + 4
        U[A_rows[0] * m + A_cols[0]][col] += 1
        U[A_rows[0] * m + A_cols[1]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] += 1
        
        # M6: (X21-X11)(Y11+Y12)
        col = base_col + 5
        U[A_rows[1] * m + A_cols[0]][col] += 1
        U[A_rows[0] * m + A_cols[0]][col] -= 1
        V[B_rows[0] * p + B_cols[0]][col] += 1
        V[B_rows[0] * p + B_cols[1]][col] += 1
        
        # M7: (X12-X22)(Y21+Y22)
        col = base_col + 6
        U[A_rows[0] * m + A_cols[1]][col] += 1
        U[A_rows[1] * m + A_cols[1]][col] -= 1
        V[B_rows[1] * p + B_cols[0]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] += 1
        
        # Now update W based on Z = X Y
        # Z11 = M1 + M4 - M5 + M7
        # Z12 = M3 + M5
        # Z21 = M2 + M4
        # Z22 = M1 - M2 + M3 + M6
        
        # C is indexed by C_rows, C_cols
        # C[0,0] corresponds to C_rows[0], C_cols[0]
        # W index: k*n + i where k is col, i is row.
        # So for C[r, c], W index is c*n + r.
        
        # Z11 -> C[C_rows[0], C_cols[0]]
        w_idx = C_cols[0] * n + C_rows[0]
        W[w_idx][base_col + 0] += 1 # M1
        W[w_idx][base_col + 3] += 1 # M4
        W[w_idx][base_col + 4] -= 1 # M5
        W[w_idx][base_col + 6] += 1 # M7
        
        # Z12 -> C[C_rows[0], C_cols[1]]
        w_idx = C_cols[1] * n + C_rows[0]
        W[w_idx][base_col + 2] += 1 # M3
        W[w_idx][base_col + 4] += 1 # M5
        
        # Z21 -> C[C_rows[1], C_cols[0]]
        w_idx = C_cols[0] * n + C_rows[1]
        W[w_idx][base_col + 1] += 1 # M2
        W[w_idx][base_col + 3] += 1 # M4
        
        # Z22 -> C[C_rows[1], C_cols[1]]
        w_idx = C_cols[1] * n + C_rows[1]
        W[w_idx][base_col + 0] += 1 # M1
        W[w_idx][base_col + 1] -= 1 # M2
        W[w_idx][base_col + 2] += 1 # M3
        W[w_idx][base_col + 5] += 1 # M6

    # Block 1: A_L * B_LL -> C_L
    # A_L: rows 0,1; cols 0,1
    # B_LL: rows 0,1; cols 0,1
    # C_L: rows 0,1; cols 0,1
    add_strassen_block(0, [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Block 2: A_R * B_RL -> C_L
    # A_R: rows 0,1; cols 2,3
    # B_RL: rows 2,3; cols 0,1
    # C_L: rows 0,1; cols 0,1
    add_strassen_block(1, [0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # Block 3: A_L * B_LR -> C_R
    # A_L: rows 0,1; cols 0,1
    # B_LR: rows 0,1; cols 2,3
    # C_R: rows 0,1; cols 2,3
    add_strassen_block(2, [0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # Block 4: A_R * B_RR -> C_R
    # A_R: rows 0,1; cols 2,3
    # B_RR: rows 2,3; cols 2,3
    # C_R: rows 0,1; cols 2,3
    add_strassen_block(3, [0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])

    return {"U": U, "V": V, "W": W}
