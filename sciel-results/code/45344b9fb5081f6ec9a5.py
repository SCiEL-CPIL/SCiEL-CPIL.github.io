"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49  # 7 blocks * 7 Strassen terms
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definition for a single 2x2 block
    # Each term is (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
    # Indices are relative to the block's start in V and W, and absolute in U.
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V relative indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> Absolute: 14*row + (2b+col)
    # W relative indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> Absolute: 2*col + row
    
    strassen_terms = [
        # M1: (A00+A11)(B00+B11)
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # M2: (A10+A11)B00
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        # M3: A00B10
        ([0], [1], [2], [1], [1, 3], [1, 1]),
        # M4: A11B11
        ([3], [1], [3], [1], [0, 2], [1, 1]),
        # M5: (A00+A01)B11
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),
        # M6: (A10-A00)(B01-B00)
        ([2, 0], [1, -1], [1, 0], [1, -1], [3], [1]),
        # M7: (A01-A11)(B00+B10)
        ([1, 3], [1, -1], [0, 2], [1, 1], [0], [1])
    ]
    
    col = 0
    for b in range(7):
        for term in strassen_terms:
            u_idx, u_val, v_idx_rel, v_val, w_idx_rel, w_val = term
            
            for u_i, u_v in zip(u_idx, u_val):
                U[u_i][col] += u_v
                
            for v_i, v_v in zip(v_idx_rel, v_val):
                # v_idx_rel: 0->B00, 1->B01, 2->B10, 3->B11
                # Row of B is 0 or 1. Col of B is 2b or 2b+1.
                # v_i=0: row 0, col 2b -> 14*0 + 2b
                # v_i=1: row 0, col 2b+1 -> 14*0 + 2b+1
                # v_i=2: row 1, col 2b -> 14*1 + 2b
                # v_i=3: row 1, col 2b+1 -> 14*1 + 2b+1
                row_b = v_i // 2
                col_b = v_i % 2
                abs_v = row_b * p + (2 * b + col_b)
                V[abs_v][col] += v_v
                
            for w_i, w_v in zip(w_idx_rel, w_val):
                # w_idx_rel: 0->C00, 1->C01, 2->C10, 3->C11
                # Col of C is 2b or 2b+1. Row of C is 0 or 1.
                # w_i=0: col 2b, row 0 -> 2*2b + 0
                # w_i=1: col 2b, row 1 -> 2*2b + 1
                # w_i=2: col 2b+1, row 0 -> 2*(2b+1) + 0
                # w_i=3: col 2b+1, row 1 -> 2*(2b+1) + 1
                col_c = w_i // 2
                row_c = w_i % 2
                abs_w = (2 * b + col_c) * n + row_c
                W[abs_w][col] += w_v
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
