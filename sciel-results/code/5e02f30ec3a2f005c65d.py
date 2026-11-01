import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We use 4 blocks of Strassen multiplication.
    # Block 0: A_L * B_LL -> C_L
    # Block 1: A_R * B_RL -> C_L
    # Block 2: A_L * B_LR -> C_R
    # Block 3: A_R * B_RR -> C_R
    
    # A is 2x4. A_L is cols 0,1. A_R is cols 2,3.
    # B is 4x4. B_LL is rows 0,1 cols 0,1. B_RL is rows 2,3 cols 0,1.
    #           B_LR is rows 0,1 cols 2,3. B_RR is rows 2,3 cols 2,3.
    # C is 2x4. C_L is cols 0,1. C_R is cols 2,3.
    
    # Strassen formulas for 2x2 matrices X, Y -> Z:
    # M1 = (X00 + X11) * (Y00 + Y11)
    # M2 = (X10 + X11) * Y00
    # M3 = X00 * (Y01 - Y11)
    # M4 = X11 * (Y10 - Y00)
    # M5 = (X00 + X01) * Y11
    # M6 = (X10 - X00) * (Y00 + Y01)
    # M7 = (X01 - X11) * (Y10 + Y11)
    #
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Helper to add a Strassen block
    def add_strassen(block_id, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        base = block_id * 7
        
        # Local indices for X (A part)
        # X00 -> A[a_rows[0], a_cols[0]]
        # X01 -> A[a_rows[0], a_cols[1]]
        # X10 -> A[a_rows[1], a_cols[0]]
        # X11 -> A[a_rows[1], a_cols[1]]
        
        # Local indices for Y (B part)
        # Y00 -> B[b_rows[0], b_cols[0]]
        # Y01 -> B[b_rows[0], b_cols[1]]
        # Y10 -> B[b_rows[1], b_cols[0]]
        # Y11 -> B[b_rows[1], b_cols[1]]
        
        # M1
        col = base + 0
        U[a_rows[0]*m + a_cols[0]][col] += 1
        U[a_rows[1]*m + a_cols[1]][col] += 1
        V[b_rows[0]*p + b_cols[0]][col] += 1
        V[b_rows[1]*p + b_cols[1]][col] += 1
        
        # M2
        col = base + 1
        U[a_rows[1]*m + a_cols[0]][col] += 1
        U[a_rows[1]*m + a_cols[1]][col] += 1
        V[b_rows[0]*p + b_cols[0]][col] += 1
        
        # M3
        col = base + 2
        U[a_rows[0]*m + a_cols[0]][col] += 1
        V[b_rows[0]*p + b_cols[1]][col] += 1
        V[b_rows[1]*p + b_cols[1]][col] -= 1
        
        # M4
        col = base + 3
        U[a_rows[1]*m + a_cols[1]][col] += 1
        V[b_rows[1]*p + b_cols[0]][col] += 1
        V[b_rows[0]*p + b_cols[0]][col] -= 1
        
        # M5
        col = base + 4
        U[a_rows[0]*m + a_cols[0]][col] += 1
        U[a_rows[0]*m + a_cols[1]][col] += 1
        V[b_rows[1]*p + b_cols[1]][col] += 1
        
        # M6
        col = base + 5
        U[a_rows[1]*m + a_cols[0]][col] += 1
        U[a_rows[0]*m + a_cols[0]][col] -= 1
        V[b_rows[0]*p + b_cols[0]][col] += 1
        V[b_rows[0]*p + b_cols[1]][col] += 1
        
        # M7
        col = base + 6
        U[a_rows[0]*m + a_cols[1]][col] += 1
        U[a_rows[1]*m + a_cols[1]][col] -= 1
        V[b_rows[1]*p + b_cols[0]][col] += 1
        V[b_rows[1]*p + b_cols[1]][col] += 1
        
        # Update W
        # Z00 -> C[c_rows[0], c_cols[0]]
        w_idx = c_cols[0]*n + c_rows[0]
        W[w_idx][base+0] += 1
        W[w_idx][base+3] += 1
        W[w_idx][base+4] -= 1
        W[w_idx][base+6] += 1
        
        # Z01 -> C[c_rows[0], c_cols[1]]
        w_idx = c_cols[1]*n + c_rows[0]
        W[w_idx][base+2] += 1
        W[w_idx][base+4] += 1
        
        # Z10 -> C[c_rows[1], c_cols[0]]
        w_idx = c_cols[0]*n + c_rows[1]
        W[w_idx][base+1] += 1
        W[w_idx][base+3] += 1
        
        # Z11 -> C[c_rows[1], c_cols[1]]
        w_idx = c_cols[1]*n + c_rows[1]
        W[w_idx][base+0] += 1
        W[w_idx][base+1] -= 1
        W[w_idx][base+2] += 1
        W[w_idx][base+5] += 1

    # Define blocks
    # A_L: rows [0,1], cols [0,1]
    # A_R: rows [0,1], cols [2,3]
    # B_LL: rows [0,1], cols [0,1]
    # B_RL: rows [2,3], cols [0,1]
    # B_LR: rows [0,1], cols [2,3]
    # B_RR: rows [2,3], cols [2,3]
    # C_L: rows [0,1], cols [0,1]
    # C_R: rows [0,1], cols [2,3]
    
    # Block 0: A_L * B_LL -> C_L
    add_strassen(0, [0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    
    # Block 1: A_R * B_RL -> C_L
    add_strassen(1, [0,1], [2,3], [2,3], [0,1], [0,1], [0,1])
    
    # Block 2: A_L * B_LR -> C_R
    add_strassen(2, [0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    
    # Block 3: A_R * B_RR -> C_R
    add_strassen(3, [0,1], [2,3], [2,3], [2,3], [0,1], [2,3])

    return {"U": U, "V": V, "W": W}
