"""
Rewritten program for Matrix Multiplication 3,4,6.
Uses a hybrid decomposition:
- 2x4x6 part: Split into two 2x2x6 parts.
  - Each 2x2x6 part: Split into three 2x2x2 parts.
  - Each 2x2x2 part: Uses Strassen's algorithm (Rank 7).
  - Total rank for 2x4x6: 2 * 3 * 7 = 42.
- 1x4x6 part: Schoolbook decomposition (Rank 24).
- Total Rank: 42 + 24 = 66.
"""
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = 3, 4, 6
    
    # Total dimensions for matrices
    u_rows = n * m # 12
    v_rows = m * p # 24
    w_rows = p * n # 18
    
    # We will build the matrices block by block
    # Block 1: 2x4x6 (n=2, m=4, p=6)
    # Block 2: 1x4x6 (n=1, m=4, p=6)
    
    rank_246 = 42
    rank_146 = 24
    total_rank = rank_246 + rank_146
    
    U = np.zeros((u_rows, total_rank), dtype=int)
    V = np.zeros((v_rows, total_rank), dtype=int)
    W = np.zeros((w_rows, total_rank), dtype=int)
    
    # --- Helper: Strassen 2x2x2 Decomposition ---
    # Returns U, V, W for 2x2x2
    def strassen_222():
        # U: 4x7, V: 4x7, W: 4x7
        # U rows: i*2+j
        # V rows: j*2+k
        # W rows: k*2+i
        u = np.zeros((4, 7), dtype=int)
        v = np.zeros((4, 7), dtype=int)
        w = np.zeros((4, 7), dtype=int)
        
        # M1 = (A11+A22)(B11+B22)
        # A11 is i=0,j=0. A22 is i=1,j=1.
        # B11 is j=0,k=0. B22 is j=1,k=1.
        # W: C11 += 1, C22 += 1
        u[0, 0] = 1; u[3, 0] = 1
        v[0, 0] = 1; v[3, 0] = 1
        w[0, 0] = 1; w[3, 0] = 1
        
        # M2 = (A21+A22)B11
        # A21 is i=1,j=0. A22 is i=1,j=1.
        # B11 is j=0,k=0.
        # W: C21 += 1
        u[2, 1] = 1; u[3, 1] = 1
        v[0, 1] = 1
        w[2, 1] = 1 # k=1? No, k=0. W index is k*2+i. k=0, i=1 -> 1.
        # Correction: W[k*n+i]. n=2.
        # C21 is i=1, k=0. Index = 0*2 + 1 = 1.
        w[1, 1] = 1
        
        # M3 = A11(B12-B22)
        # A11 is i=0,j=0.
        # B12 is j=0,k=1. B22 is j=1,k=1.
        # W: C12 += 1
        u[0, 2] = 1
        v[1, 2] = 1; v[3, 2] = -1
        # C12 is i=0, k=1. Index = 1*2 + 0 = 2.
        w[2, 2] = 1
        
        # M4 = A22(B21-B11)
        # A22 is i=1,j=1.
        # B21 is j=1,k=0. B11 is j=0,k=0.
        # W: C11 += 1, C21 += 1
        u[3, 3] = 1
        v[2, 3] = 1; v[0, 3] = -1
        # C11 is i=0, k=0. Index = 0.
        w[0, 3] = 1
        # C21 is i=1, k=0. Index = 1.
        w[1, 3] = 1
        
        # M5 = (A11+A12)B22
        # A11 is i=0,j=0. A12 is i=0,j=1.
        # B22 is j=1,k=1.
        # W: C12 -= 1, C22 += 1
        u[0, 4] = 1; u[1, 4] = 1
        v[3, 4] = 1
        # C12 is i=0, k=1. Index = 2.
        w[2, 4] = -1
        # C22 is i=1, k=1. Index = 3.
        w[3, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        # A21 is i=1,j=0. A11 is i=0,j=0.
        # B11 is j=0,k=0. B12 is j=0,k=1.
        # W: C22 += 1
        u[2, 5] = 1; u[0, 5] = -1
        v[0, 5] = 1; v[1, 5] = 1
        # C22 is i=1, k=1. Index = 3.
        w[3, 5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        # A12 is i=0,j=1. A22 is i=1,j=1.
        # B21 is j=1,k=0. B22 is j=1,k=1.
        # W: C11 += 1, C22 -= 1
        u[1, 6] = 1; u[3, 6] = -1
        v[2, 6] = 1; v[3, 6] = 1
        # C11 is i=0, k=0. Index = 0.
        w[0, 6] = 1
        # C22 is i=1, k=1. Index = 3.
        w[3, 6] = -1
        
        return u, v, w

    # --- Build 2x2x6 Block ---
    # This block corresponds to n=2, m=2, p=6.
    # It consists of 3 Strassen 2x2x2 blocks.
    def build_226_block(offset_u_i, offset_v_j, offset_w_k):
        """
        Builds the U, V, W contributions for a 2x2x6 block.
        offset_u_i: offset for i index in global U (n dimension)
        offset_v_j: offset for j index in global V (m dimension)
        offset_w_k: offset for k index in global W (p dimension)
        Returns U_part (4x21), V_part (12x21), W_part (12x21) 
        Wait, U for 2x2 is 4 rows. V for 2x6 is 12 rows. W for 6x2 is 12 rows.
        """
        u_part = np.zeros((4, 21), dtype=int)
        v_part = np.zeros((12, 21), dtype=int)
        w_part = np.zeros((12, 21), dtype=int)
        
        u_str, v_str, w_str = strassen_222()
        
        for block_idx in range(3):
            # k_offset in the 2x2x2 block: 0, 2, 4
            k_off = block_idx * 2
            col_start = block_idx * 7
            
            # U: Same for all 3 blocks (shares A)
            # Global U row: (offset_u_i + i) * 2 + j
            # Local U row: i * 2 + j
            # So we just copy u_str
            u_part[:, col_start:col_start+7] = u_str
            
            # V: V rows are j * 6 + k (for m=2, p=6? No, m=2, p=6 means V is 2x6=12 rows)
            # Wait, in 2x2x6, m=2, p=6.
            # V index: j * p + k = j * 6 + k.
            # Local j in 0..1. Local k in 0..1 (within the 2x2x2 block).
            # Global k = offset_w_k + k_off + local_k.
            # Global j = offset_v_j + local_j.
            # Global V row = (offset_v_j + local_j) * 6 + (offset_w_k + k_off + local_k)
            
            for local_j in range(2):
                for local_k in range(2):
                    local_v_row = local_j * 2 + local_k
                    global_j = offset_v_j + local_j
                    global_k = offset_w_k + k_off + local_k
                    global_v_row = global_j * 6 + global_k
                    
                    # Copy values from v_str
                    # v_str row local_v_row, cols col_start..col_start+6
                    v_part[global_v_row, col_start:col_start+7] = v_str[local_v_row, :]
            
            # W: W rows are k * n + i (for p=6, n=2? No, p=6, n=2 means W is 6x2=12 rows)
            # W index: k * n + i = k * 2 + i.
            # Local k in 0..1. Local i in 0..1.
            # Global k = offset_w_k + k_off + local_k.
            # Global i = offset_u_i + local_i.
            # Global W row = (offset_w_k + k_off + local_k) * 2 + (offset_u_i + local_i)
            
            for local_k in range(2):
                for local_i in range(2):
                    local_w_row = local_k * 2 + local_i
                    global_k = offset_w_k + k_off + local_k
                    global_i = offset_u_i + local_i
                    global_w_row = global_k * 2 + global_i
                    
                    # Copy values from w_str
                    w_part[global_w_row, col_start:col_start+7] = w_str[local_w_row, :]
        
        return u_part, v_part, w_part

    # --- Build 2x4x6 Part ---
    # n=2, m=4, p=6.
    # Split m=4 into two m=2 parts.
    # Part A: i in 0..1, j in 0..1, k in 0..5.
    # Part B: i in 0..1, j in 2..3, k in 0..5.
    
    # Part A
    u_a, v_a, w_a = build_226_block(offset_u_i=0, offset_v_j=0, offset_w_k=0)
    # Part B
    u_b, v_b, w_b = build_226_block(offset_u_i=0, offset_v_j=2, offset_w_k=0)
    
    # Combine into 2x4x6 matrices
    # U: 2x4 = 8 rows.
    U_246 = np.zeros((8, 42), dtype=int)
    V_246 = np.zeros((24, 42), dtype=int) # 4*6
    W_246 = np.zeros((12, 42), dtype=int) # 6*2
    
    # U Part A: rows 0..3 (i=0,1; j=0,1)
    U_246[0:4, 0:21] = u_a
    # U Part B: rows 4..7 (i=0,1; j=2,3)
    # Local U rows for Part B are same 0..3, but global j is shifted by 2.
    # U global row = i*4 + j.
    # For Part B, j_local in 0..1 corresponds to j_global in 2..3.
    # So U_246 rows 4..7 get u_b.
    U_246[4:8, 21:42] = u_b
    
    # V Part A: rows 0..11 (j=0,1; k=0..5)
    V_246[0:12, 0:21] = v_a
    # V Part B: rows 12..23 (j=2,3; k=0..5)
    V_246[12:24, 21:42] = v_b
    
    # W Part A: rows 0..11 (k=0..5; i=0,1)
    W_246[0:12, 0:21] = w_a
    # W Part B: rows 0..11 (k=0..5; i=0,1) -- Overlap!
    # W is summed. So we add them.
    W_246[0:12, 21:42] = w_b
    
    # --- Build 1x4x6 Part ---
    # n=1, m=4, p=6. Rank 24.
    U_146 = np.zeros((4, 24), dtype=int)
    V_146 = np.zeros((24, 24), dtype=int)
    W_146 = np.zeros((6, 24), dtype=int)
    
    col = 0
    for j in range(4):
        for k in range(6):
            U_146[j, col] = 1
            V_146[j * 6 + k, col] = 1
            W_146[k, col] = 1 # k * 1 + 0
            col += 1
            
    # --- Assemble Final 3x4x6 ---
    # n=3, m=4, p=6.
    # U: 3x4 = 12 rows.
    # V: 4x6 = 24 rows.
    # W: 6x3 = 18 rows.
    
    # U:
    # Rows 0..7: i=0,1. (From 2x4x6 part)
    # Rows 8..11: i=2. (From 1x4x6 part)
    U_final[0:8, 0:42] = U_246
    U_final[8:12, 42:66] = U_146
    
    # V:
    # Rows 0..23: j=0..3.
    # 2x4x6 part uses all j=0..3.
    # 1x4x6 part uses all j=0..3.
    V_final[0:24, 0:42] = V_246
    V_final[0:24, 42:66] = V_146
    
    # W:
    # Rows 0..17: k=0..5, i=0..2.
    # 2x4x6 part contributes to i=0,1. (Rows 0..11)
    # 1x4x6 part contributes to i=2. (Rows 12..17)
    # W_246 is 12 rows (k=0..5, i=0,1).
    # W_146 is 6 rows (k=0..5, i=2? No, i=0 in local, but global i=2).
    # Global W row = k * 3 + i.
    # For 2x4x6 part: i_local in 0,1 -> i_global in 0,1.
    #   W_246 row r corresponds to k, i_local.
    #   Global row = k * 3 + i_local.
    # For 1x4x6 part: i_local in 0 -> i_global in 2.
    #   W_146 row r corresponds to k.
    #   Global row = k * 3 + 2.
    
    # Map W_246 to W_final
    for local_k in range(6):
        for local_i in range(2):
            local_w_row = local_k * 2 + local_i
            global_w_row = local_k * 3 + local_i
            W_final[global_w_row, 0:42] = W_246[local_w_row, 0:42]
            
    # Map W_146 to W_final
    for local_k in range(6):
        local_w_row = local_k
        global_w_row = local_k * 3 + 2
        W_final[global_w_row, 42:66] = W_146[local_w_row, :]
        
    return {"U": U_final, "V": V_final, "W": W_final}

# Initialize global arrays for construct
U_final = np.zeros((12, 66), dtype=int)
V_final = np.zeros((24, 66), dtype=int)
W_final = np.zeros((18, 66), dtype=int)
