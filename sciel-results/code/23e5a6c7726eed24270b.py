import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block definitions
    # (A_start_row, A_start_col, B_start_row, B_start_col)
    # A is 2x4. B is 4x6.
    # A_left: rows 0-1, cols 0-1
    # A_right: rows 0-1, cols 2-3
    # B_top: rows 0-1
    # B_bot: rows 2-3
    
    blocks = [
        # (a_rows, a_cols, b_rows, b_cols)
        # A indices in U: i*m + j. i in a_rows, j in a_cols.
        # B indices in V: j*p + k. j in b_rows, k in b_cols.
        # W indices: k*n + i. k in b_cols, i in a_rows.
        
        # Block 0: A_left * B_top_left (cols 0-1 of B)
        ( [0, 1], [0, 1], [0, 1], [0, 1] ),
        # Block 1: A_left * B_top_mid (cols 2-3 of B)
        ( [0, 1], [0, 1], [0, 1], [2, 3] ),
        # Block 2: A_left * B_top_right (cols 4-5 of B)
        ( [0, 1], [0, 1], [0, 1], [4, 5] ),
        # Block 3: A_right * B_bot_left (cols 0-1 of B)
        ( [0, 1], [2, 3], [2, 3], [0, 1] ),
        # Block 4: A_right * B_bot_mid (cols 2-3 of B)
        ( [0, 1], [2, 3], [2, 3], [2, 3] ),
        # Block 5: A_right * B_bot_right (cols 4-5 of B)
        ( [0, 1], [2, 3], [2, 3], [4, 5] ),
    ]
    
    # Strassen components
    # Each component: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (local_a_idx, coeff)
    # v_coeffs: list of (local_b_idx, coeff)
    # w_coeffs: list of (local_z_idx, coeff)
    # local_a_idx: 0->A11, 1->A12, 2->A21, 3->A22
    # local_b_idx: 0->B11, 1->B12, 2->B21, 3->B22
    # local_z_idx: 0->Z11, 1->Z12, 2->Z21, 3->Z22
    
    strassen = [
        # M1
        ( [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)] ),
        # M2
        ( [(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)] ),
        # M3
        ( [(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)] ),
        # M4
        ( [(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)] ),
        # M5
        ( [(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)] ),
        # M6
        ( [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)] ),
        # M7
        ( [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)] ),
    ]
    
    col = 0
    for block in blocks:
        a_rows, a_cols, b_rows, b_cols = block
        
        # Map local indices to global indices
        # Local A: 0->(a_rows[0], a_cols[0]), 1->(a_rows[0], a_cols[1]), 2->(a_rows[1], a_cols[0]), 3->(a_rows[1], a_cols[1])
        a_map = {
            0: a_rows[0] * m + a_cols[0],
            1: a_rows[0] * m + a_cols[1],
            2: a_rows[1] * m + a_cols[0],
            3: a_rows[1] * m + a_cols[1]
        }
        
        # Local B: 0->(b_rows[0], b_cols[0]), 1->(b_rows[0], b_cols[1]), 2->(b_rows[1], b_cols[0]), 3->(b_rows[1], b_cols[1])
        b_map = {
            0: b_rows[0] * p + b_cols[0],
            1: b_rows[0] * p + b_cols[1],
            2: b_rows[1] * p + b_cols[0],
            3: b_rows[1] * p + b_cols[1]
        }
        
        # Local Z: 0->(b_cols[0], a_rows[0]), 1->(b_cols[1], a_rows[0]), 2->(b_cols[0], a_rows[1]), 3->(b_cols[1], a_rows[1])
        # W index: k * n + i
        w_map = {
            0: b_cols[0] * n + a_rows[0],
            1: b_cols[1] * n + a_rows[0],
            2: b_cols[0] * n + a_rows[1],
            3: b_cols[1] * n + a_rows[1]
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen:
            for local_idx, coeff in u_coeffs:
                global_idx = a_map[local_idx]
                U[global_idx][col] += coeff
            
            for local_idx, coeff in v_coeffs:
                global_idx = b_map[local_idx]
                V[global_idx][col] += coeff
            
            for local_idx, coeff in w_coeffs:
                global_idx = w_map[local_idx]
                W[global_idx][col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
