import time

DIMENSIONS = [2, 4, 6]
_START_TIME = time.time()
_DEADLINE = 140.0

def _time_ok():
    return time.time() < _DEADLINE

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen components for 2x2x2
    # Indices: 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    strassen = [
        ( [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)] ),
        ( [(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)] ),
        ( [(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)] ),
        ( [(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)] ),
        ( [(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)] ),
        ( [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)] ),
        ( [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)] ),
    ]
    
    # We split the 2x4x6 multiplication into 6 blocks of 2x2x2
    # A is 2x4, B is 4x6.
    # A = [A11 A12; A21 A22] where each Ai is 1x2? No.
    # A rows 0,1. Cols 0,1,2,3.
    # B rows 0,1,2,3. Cols 0,1,2,3,4,5.
    
    # Block (a_row_block, a_col_block, b_row_block, b_col_block)
    # a_row_block: 0 or 1 (size 1)
    # a_col_block: 0 or 1 (size 2)
    # b_row_block: 0 or 1 (size 2)
    # b_col_block: 0, 1, or 2 (size 2)
    
    # Actually, the previous code used 2x2 blocks for A and B.
    # A 2x4 can be split into two 2x2 blocks horizontally.
    # B 4x6 can be split into two 2x6 blocks vertically?
    # Or 4x2 blocks?
    
    # Let's stick to the 6 blocks of 2x2x2.
    # A block: 2 rows, 2 cols.
    # B block: 2 rows, 2 cols.
    
    # A has 2 rows. So a_row_block is always [0,1].
    # A has 4 cols. Split into [0,1] and [2,3].
    # B has 4 rows. Split into [0,1] and [2,3].
    # B has 6 cols. Split into [0,1], [2,3], [4,5].
    
    blocks = [
        # (a_rows, a_cols, b_rows, b_cols)
        ([0, 1], [0, 1], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [0, 1], [2, 3]),
        ([0, 1], [0, 1], [0, 1], [4, 5]),
        ([0, 1], [2, 3], [2, 3], [0, 1]),
        ([0, 1], [2, 3], [2, 3], [2, 3]),
        ([0, 1], [2, 3], [2, 3], [4, 5]),
    ]
    
    col = 0
    for block in blocks:
        a_rows, a_cols, b_rows, b_cols = block
        a_map = {
            0: a_rows[0] * m + a_cols[0],
            1: a_rows[0] * m + a_cols[1],
            2: a_rows[1] * m + a_cols[0],
            3: a_rows[1] * m + a_cols[1]
        }
        b_map = {
            0: b_rows[0] * p + b_cols[0],
            1: b_rows[0] * p + b_cols[1],
            2: b_rows[1] * p + b_cols[0],
            3: b_rows[1] * p + b_cols[1]
        }
        w_map = {
            0: b_cols[0] * n + a_rows[0],
            1: b_cols[1] * n + a_rows[0],
            2: b_cols[0] * n + a_rows[1],
            3: b_cols[1] * n + a_rows[1]
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen:
            for local_idx, coeff in u_coeffs:
                U[a_map[local_idx]][col] += coeff
            for local_idx, coeff in v_coeffs:
                V[b_map[local_idx]][col] += coeff
            for local_idx, coeff in w_coeffs:
                W[w_map[local_idx]][col] += coeff
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
