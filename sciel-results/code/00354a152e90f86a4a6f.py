import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    # We will build a rank 90 decomposition
    # Structure:
    # A (4x5) split into 2x2, 2x3 (top) and 2x2, 2x3 (bottom)
    # B (5x6) split into 5x2 (B1), 5x2 (B2), 5x2 (B3)
    # Each Bk split into 2x2 (top) and 3x2 (bottom)
    # Each 3x2 split into 2x2 (top) and 1x2 (bottom)
    
    # Local dimensions
    # A top-left: 2x2
    # A top-right: 2x3 -> 2x2 + 2x1
    # A bot-left: 2x2
    # A bot-right: 2x3 -> 2x2 + 2x1
    
    # B top: 2x2
    # B bot: 3x2 -> 2x2 + 1x2
    
    # C blocks:
    # C_top = A_top_left * B_top + A_top_right * B_bot
    # C_bot = A_bot_left * B_top + A_bot_right * B_bot
    
    # Ranks:
    # A_top_left * B_top: 2x2x2 (Strassen, rank 7)
    # A_top_right * B_bot: (2x2+2x1) * (2x2+1x2)
    #   = 2x2x2 (Strassen, rank 7) + 2x1x2 (Rank 1, rank 1)
    #   Total 8
    # Similarly for bottom.
    # Total per Bk: 7 + 8 + 7 + 8 = 30
    # 3 Bks: 90
    
    R = 90
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)
    
    # Strassen coefficients
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def add_strassen(U_off, V_off, W_off, start_rank):
        # U_off: (i_offset, j_offset) in A
        # V_off: (j_offset, k_offset) in B
        # W_off: (k_offset, i_offset) in C (W index k*n+i)
        # We add 7 terms
        
        # M1
        # U: A11, A22
        # V: B11, B22
        # W: C11, C22
        # Coeffs for U: A11: 1, A22: 1
        # Coeffs for V: B11: 1, B22: 1
        # Coeffs for W: C11: 1, C22: 1
        
        # We need to map local 2x2 indices to global
        # Local A: 0,1 (rows), 0,1 (cols)
        # Local B: 0,1 (rows), 0,1 (cols)
        # Local C: 0,1 (cols), 0,1 (rows) -> W index k*2 + i? 
        # Wait, W is global (p*n).
        # Local C is 2x2.
        # W local indices: k in 0..1, i in 0..1.
        # Global W index: (k_offset + k) * n + (i_offset + i)
        
        r = start_rank
        
        # M1: (A11+A22)(B11+B22)
        # U: A11(1), A22(1)
        U[U_off[0] + 0, U_off[1] + 0, r] = 1.0
        U[U_off[0] + 1, U_off[1] + 1, r] = 1.0
        # V: B11(1), B22(1)
        V[V_off[0] + 0, V_off[1] + 0, r] = 1.0
        V[V_off[0] + 1, V_off[1] + 1, r] = 1.0
        # W: C11(1), C22(1)
        W[(W_off[0] + 0) * n + (W_off[1] + 0), r] = 1.0
        W[(W_off[0] + 1) * n + (W_off[1] + 1), r] = 1.0
        r += 1
        
        # M2: (A21+A22)B11
        U[U_off[0] + 1, U_off[1] + 0, r] = 1.0
        U[U_off[0] + 1, U_off[1] + 1, r] = 1.0
        V[V_off[0] + 0, V_off[1] + 0, r] = 1.0
        W[(W_off[0] + 0) * n + (W_off[1] + 1), r] = 1.0
        r += 1
        
        # M3: A11(B12-B22)
        U[U_off[0] + 0, U_off[1] + 0, r] = 1.0
        V[V_off[0] + 0, V_off[1] + 1, r] = 1.0
        V[V_off[0] + 1, V_off[1] + 1, r] = -1.0
        W[(W_off[0] + 1) * n + (W_off[1] + 0), r] = 1.0
        r += 1
        
        # M4: A22(B21-B11)
        U[U_off[0] + 1, U_off[1] + 1, r] = 1.0
        V[V_off[0] + 1, V_off[1] + 0, r] = 1.0
        V[V_off[0] + 0, V_off[1] + 0, r] = -1.0
        W[(W_off[0] + 0) * n + (W_off[1] + 1), r] = 1.0
        W[(W_off[0] + 0) * n + (W_off[1] + 0), r] = 1.0 # Wait, C11 gets M4
        # Correction: C11 = M1 + M4 ...
        # So W for M4 should affect C11 and C21?
        # C11 = M1 + M4 - M5 + M7
        # C21 = M2 + M4
        # So M4 contributes to C11 (1) and C21 (1).
        # My previous line for M4 W was:
        # W[C21] = 1.0
        # I missed C11.
        # Let's re-verify Strassen W contributions.
        
        # Let's list W contributions for each M:
        # M1: C11(1), C22(1)
        # M2: C21(1)
        # M3: C12(1)
        # M4: C11(1), C21(1)
        # M5: C11(-1), C12(1)
        # M6: C22(1)
        # M7: C11(1)
        
        # So for M4:
        W[(W_off[0] + 0) * n + (W_off[1] + 0), r] = 1.0 # C11
        W[(W_off[0] + 0) * n + (W_off[1] + 1), r] = 1.0 # C21
        r += 1
        
        # M5: (A11+A12)B22
        U[U_off[0] + 0, U_off[1] + 0, r] = 1.0
        U[U_off[0] + 0, U_off[1] + 1, r] = 1.0
        V[V_off[0] + 1, V_off[1] + 1, r] = 1.0
        W[(W_off[0] + 0) * n + (W_off[1] + 0), r] = -1.0 # C11
        W[(W_off[0] + 1) * n + (W_off[1] + 0), r] = 1.0 # C12
        r += 1
        
        # M6: (A21-A11)(B11+B12)
        U[U_off[0] + 1, U_off[1] + 0, r] = 1.0
        U[U_off[0] + 0, U_off[1] + 0, r] = -1.0
        V[V_off[0] + 0, V_off[1] + 0, r] = 1.0
        V[V_off[0] + 0, V_off[1] + 1, r] = 1.0
        W[(W_off[0] + 1) * n + (W_off[1] + 1), r] = 1.0 # C22
        r += 1
        
        # M7: (A12-A22)(B21+B22)
        U[U_off[0] + 0, U_off[1] + 1, r] = 1.0
        U[U_off[0] + 1, U_off[1] + 1, r] = -1.0
        V[V_off[0] + 1, V_off[1] + 0, r] = 1.0
        V[V_off[0] + 1, V_off[1] + 1, r] = 1.0
        W[(W_off[0] + 0) * n + (W_off[1] + 0), r] = 1.0 # C11
        r += 1
        
        return r

    def add_rank1_2x1x2(U_off, V_off, W_off, start_rank):
        # A is 2x1, B is 1x2
        # U: A00, A10
        # V: B00, B01
        # W: C00, C01, C10, C11 all 1?
        # T_i0k = A_i0 B_0k
        # W index: k*n + i.
        # k=0, i=0 -> W[0]
        # k=0, i=1 -> W[1]
        # k=1, i=0 -> W[2]
        # k=1, i=1 -> W[3]
        # All should be 1.
        
        r = start_rank
        U[U_off[0] + 0, U_off[1] + 0, r] = 1.0
        U[U_off[0] + 1, U_off[1] + 0, r] = 1.0
        V[V_off[0] + 0, V_off[1] + 0, r] = 1.0
        V[V_off[0] + 0, V_off[1] + 1, r] = 1.0
        W[(W_off[0] + 0) * n + (W_off[1] + 0), r] = 1.0
        W[(W_off[0] + 0) * n + (W_off[1] + 1), r] = 1.0
        W[(W_off[0] + 1) * n + (W_off[1] + 0), r] = 1.0
        W[(W_off[0] + 1) * n + (W_off[1] + 1), r] = 1.0
        r += 1
        return r

    # Global offsets
    # A: 4x5
    # Rows 0-1 (top), 2-3 (bot)
    # Cols 0-1 (left), 2-4 (right)
    
    # B: 5x6
    # Rows 0-1 (top), 2-4 (bot)
    # Cols 0-1 (B1), 2-3 (B2), 4-5 (B3)
    
    # C: 4x6
    # Rows 0-1 (top), 2-3 (bot)
    # Cols 0-1 (C1), 2-3 (C2), 4-5 (C3)
    
    r = 0
    for k_block in range(3):
        # B block k_block: cols 2*k_block to 2*k_block+1
        B_col_start = 2 * k_block
        C_col_start = 2 * k_block
        
        # Top part of C (rows 0-1)
        # C_top = A_top_left * B_top + A_top_right * B_bot
        
        # Term 1: A_top_left (2x2) * B_top (2x2)
        # A offset: row 0, col 0
        # B offset: row 0, col B_col_start
        # C offset: row 0, col C_col_start
        r = add_strassen((0, 0), (0, B_col_start), (C_col_start, 0), r)
        
        # Term 2: A_top_right (2x3) * B_bot (3x2)
        # Split A_top_right into 2x2 (cols 2-3) + 2x1 (col 4)
        # Split B_bot into 2x2 (rows 2-3) + 1x2 (row 4)
        
        # Sub-term 2a: A_top_right_2x2 (2x2) * B_bot_2x2 (2x2)
        # A offset: row 0, col 2
        # B offset: row 2, col B_col_start
        # C offset: row 0, col C_col_start
        r = add_strassen((0, 2), (2, B_col_start), (C_col_start, 0), r)
        
        # Sub-term 2b: A_top_right_2x1 (2x1) * B_bot_1x2 (1x2)
        # A offset: row 0, col 4
        # B offset: row 4, col B_col_start
        # C offset: row 0, col C_col_start
        r = add_rank1_2x1x2((0, 4), (4, B_col_start), (C_col_start, 0), r)
        
        # Bottom part of C (rows 2-3)
        # C_bot = A_bot_left * B_top + A_bot_right * B_bot
        
        # Term 3: A_bot_left (2x2) * B_top (2x2)
        # A offset: row 2, col 0
        # B offset: row 0, col B_col_start
        # C offset: row 2, col C_col_start
        r = add_strassen((2, 0), (0, B_col_start), (C_col_start, 2), r)
        
        # Term 4: A_bot_right (2x3) * B_bot (3x2)
        
        # Sub-term 4a: A_bot_right_2x2 (2x2) * B_bot_2x2 (2x2)
        # A offset: row 2, col 2
        # B offset: row 2, col B_col_start
        # C offset: row 2, col C_col_start
        r = add_strassen((2, 2), (2, B_col_start), (C_col_start, 2), r)
        
        # Sub-term 4b: A_bot_right_2x1 (2x1) * B_bot_1x2 (1x2)
        # A offset: row 2, col 4
        # B offset: row 4, col B_col_start
        # C offset: row 2, col C_col_start
        r = add_rank1_2x1x2((2, 4), (4, B_col_start), (C_col_start, 2), r)
        
        if time.time() > deadline:
            break

    if r != R:
        # Adjust R if needed, but should be 90
        R = r
        U = U[:, :R]
        V = V[:, :R]
        W = W[:, :R]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
