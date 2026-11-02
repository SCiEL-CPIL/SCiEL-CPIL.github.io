"""Rewritten program using Strassen's algorithm extended to 2x2 x 2x16 matrix multiplication."""

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (32, rank)
    
    # We decompose the 16 columns of B into 8 blocks of 2 columns.
    # Each block corresponds to a 2x2x2 matrix multiplication.
    # Strassen rank for 2x2x2 is 7.
    # Total rank = 8 * 7 = 56.
    
    num_blocks = p // 2 # 16 // 2 = 8
    strassen_rank = 7
    total_rank = num_blocks * strassen_rank
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Define Strassen factors for a single 2x2x2 multiplication
    # Local indices for A (U_s): 0: a00, 1: a01, 2: a10, 3: a11
    # Local indices for B (V_s): 0: b00, 1: b01, 2: b10, 3: b11
    # Local indices for C (W_s): 0: c00, 1: c01, 2: c10, 3: c11
    # Columns 0..6 correspond to m1..m7
    
    # U_s entries: list of (local_row, value) for each column
    # V_s entries: list of (local_row, value) for each column
    # W_s entries: list of (local_row, value) for each column
    
    us_factors = [
        [(0, 1), (3, 1)],       # m1: (a00+a11)
        [(2, 1), (3, 1)],       # m2: (a10+a11)
        [(0, 1)],               # m3: a00
        [(3, 1)],               # m4: a11
        [(0, 1), (1, 1)],       # m5: (a00+a01)
        [(2, 1), (0, -1)],      # m6: (a10-a00)
        [(1, 1), (3, -1)],      # m7: (a01-a11)
    ]
    
    vs_factors = [
        [(0, 1), (3, 1)],       # m1: (b00+b11)
        [(0, 1)],               # m2: b00
        [(1, 1), (3, -1)],      # m3: (b01-b11)
        [(2, 1), (0, -1)],      # m4: (b10-b00)
        [(3, 1)],               # m5: b11
        [(0, 1), (1, 1)],       # m6: (b00+b01)
        [(2, 1), (3, 1)],       # m7: (b10+b11)
    ]
    
    ws_factors = [
        [(0, 1), (3, 1)],       # c00 = m1+m4-m5+m7 -> m1:+1, m4:+1, m5:-1, m7:+1 (Wait, check signs)
        # Let's re-verify Strassen signs for W
        # c00 = m1 + m4 - m5 + m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m6
        
        [(0, 1), (3, 1), (2, 1)],      # m1: c00:+1, c01:0, c10:0, c11:+1. m2: c10:+1. m3: c01:+1, c11:+1. m4: c00:+1, c10:+1. m5: c00:-1, c01:+1. m6: c11:+1. m7: c00:+1, c11:-1.
        # Let's map each column to the W rows properly.
    ]
    
    # Let's define W_s precisely column by column
    # Col 0 (m1): c00+=1, c11+=1
    w_col_0 = [(0, 1), (3, 1)]
    # Col 1 (m2): c10+=1
    w_col_1 = [(2, 1)]
    # Col 2 (m3): c01+=1, c11+=1
    w_col_2 = [(1, 1), (3, 1)]
    # Col 3 (m4): c00+=1, c10+=1
    w_col_3 = [(0, 1), (2, 1)]
    # Col 4 (m5): c00+=-1, c01+=1
    w_col_4 = [(0, -1), (1, 1)]
    # Col 5 (m6): c11+=1
    w_col_5 = [(3, 1)]
    # Col 6 (m7): c00+=1, c11+=-1
    w_col_6 = [(0, 1), (3, -1)]
    
    ws_factors = [w_col_0, w_col_1, w_col_2, w_col_3, w_col_4, w_col_5, w_col_6]
    
    for t in range(num_blocks):
        base_col = t * strassen_rank
        for s in range(strassen_rank):
            col_idx = base_col + s
            
            # Update U
            for local_row, val in us_factors[s]:
                U[local_row][col_idx] += val
                
            # Update V
            # Local 0 -> Global 2*t
            # Local 1 -> Global 2*t + 1
            # Local 2 -> Global 16 + 2*t
            # Local 3 -> Global 16 + 2*t + 1
            for local_row, val in vs_factors[s]:
                if local_row == 0:
                    global_row = 2 * t
                elif local_row == 1:
                    global_row = 2 * t + 1
                elif local_row == 2:
                    global_row = 16 + 2 * t
                else: # local_row == 3
                    global_row = 16 + 2 * t + 1
                V[global_row][col_idx] += val
                
            # Update W
            # Local 0 -> Global 2*t
            # Local 1 -> Global 2*t + 1
            # Local 2 -> Global 16 + 2*t
            # Local 3 -> Global 16 + 2*t + 1
            for local_row, val in ws_factors[s]:
                if local_row == 0:
                    global_row = 2 * t
                elif local_row == 1:
                    global_row = 2 * t + 1
                elif local_row == 2:
                    global_row = 16 + 2 * t
                else: # local_row == 3
                    global_row = 16 + 2 * t + 1
                W[global_row][col_idx] += val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
