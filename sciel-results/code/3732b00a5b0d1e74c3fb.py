import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-28 decomposition for 2x4 by 4x4 matrix multiplication.
    Uses Strassen's algorithm on 2x2 blocks.
    """
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen components for 2x2x2
    # Each component is (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    # Indices are (row, col) in the 2x2 block
    strassen_data = [
        # M1
        ([(0,0), (1,1)], [1, 1], 
         [(0,0), (1,1)], [1, 1], 
         [(0,0), (1,1)], [1, 1]),
        # M2
        ([(1,0), (1,1)], [1, 1], 
         [(0,0)], [1], 
         [(1,0), (1,1)], [1, -1]),
        # M3
        ([(0,0)], [1], 
         [(0,1), (1,1)], [1, -1], 
         [(0,1), (1,1)], [1, 1]),
        # M4
        ([(1,1)], [1], 
         [(1,0), (0,0)], [1, -1], 
         [(0,0), (1,0)], [1, 1]),
        # M5
        ([(0,0), (0,1)], [1, 1], 
         [(1,1)], [1], 
         [(0,0), (0,1)], [-1, 1]),
        # M6
        ([(1,0), (0,0)], [1, -1], 
         [(0,0), (0,1)], [1, 1], 
         [(1,1)], [1]),
        # M7
        ([(0,1), (1,1)], [1, -1], 
         [(1,0), (1,1)], [1, 1], 
         [(0,0)], [1]),
    ]

    # Terms: (a_col_off, b_row_off, b_col_off, c_col_off)
    # a_col_off: offset in A columns (0 or 2)
    # b_row_off: offset in B rows (0 or 2)
    # b_col_off: offset in B columns (0 or 2)
    # c_col_off: offset in C columns (0 or 2)
    terms = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 2, 2),
    ]

    for t_idx, (a_off, b_row, b_col, c_off) in enumerate(terms):
        for s_idx, (u_idx_list, u_vals, v_idx_list, v_vals, w_idx_list, w_vals) in enumerate(strassen_data):
            c = t_idx * 7 + s_idx
            
            for idx, val in zip(u_idx_list, u_vals):
                r, col = idx
                u_flat = r * m + (a_off + col)
                U[u_flat, c] += val
                
            for idx, val in zip(v_idx_list, v_vals):
                r, col = idx
                v_flat = (b_row + r) * p + (b_col + col)
                V[v_flat, c] += val
                
            for idx, val in zip(w_idx_list, w_vals):
                r, col = idx
                w_flat = (c_off + col) * n + r
                W[w_flat, c] += val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
