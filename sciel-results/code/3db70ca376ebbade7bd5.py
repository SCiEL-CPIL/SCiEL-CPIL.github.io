import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 66
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, val):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    def add_1x2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        # A is 1x2, B is 2x2, C is 1x2
        # A indices: (a_row_offset, a_col_offset + 0), (a_row_offset, a_col_offset + 1)
        # B indices: (b_row_offset + 0, b_col_offset + 0), (b_row_offset + 0, b_col_offset + 1)
        #            (b_row_offset + 1, b_col_offset + 0), (b_row_offset + 1, b_col_offset + 1)
        # C indices: (c_row_offset, c_col_offset + 0), (c_row_offset, c_col_offset + 1)
        
        # M1: a0 * b00 -> C00
        u = a_row_offset * m + (a_col_offset + 0)
        v = (b_row_offset + 0) * p + (b_col_offset + 0)
        w = (c_col_offset + 0) * n + c_row_offset
        add_term(u, v, w, 1)
        
        # M2: a1 * b10 -> C00
        u = a_row_offset * m + (a_col_offset + 1)
        v = (b_row_offset + 1) * p + (b_col_offset + 0)
        w = (c_col_offset + 0) * n + c_row_offset
        add_term(u, v, w, 1)
        
        # M3: a0 * b01 -> C01
        u = a_row_offset * m + (a_col_offset + 0)
        v = (b_row_offset + 0) * p + (b_col_offset + 1)
        w = (c_col_offset + 1) * n + c_row_offset
        add_term(u, v, w, 1)
        
        # M4: a1 * b11 -> C01
        u = a_row_offset * m + (a_col_offset + 1)
        v = (b_row_offset + 1) * p + (b_col_offset + 1)
        w = (c_col_offset + 1) * n + c_row_offset
        add_term(u, v, w, 1)

    def add_2x2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        # A is 2x2, B is 2x2, C is 2x2
        # Local indices (r, c) map to global:
        # A: (a_row_offset + r, a_col_offset + c)
        # B: (b_row_offset + r, b_col_offset + c)
        # C: (c_row_offset + r, c_col_offset + c)
        
        def u_idx(r, c):
            return (a_row_offset + r) * m + (a_col_offset + c)
            
        def v_idx(r, c):
            return (b_row_offset + r) * p + (b_col_offset + c)
            
        def w_idx(r, c):
            return (c_col_offset + c) * n + (c_row_offset + r)

        # M1: (a00+a11)(b00+b11)
        # U: (0,0)=1, (1,1)=1
        # V: (0,0)=1, (1,1)=1
        # W: (0,0)=1, (1,1)=1
        add_term(u_idx(0,0), v_idx(0,0), w_idx(0,0), 1)
        add_term(u_idx(1,1), v_idx(0,0), w_idx(0,0), 1) # Wait, this is not how it works.
        
        # Correction: A single rank 1 term corresponds to ONE multiplication in the algorithm.
        # But Strassen's M1 is ONE multiplication.
        # So we need ONE column in U, V, W for M1.
        # But U, V, W are vectors.
        # So we set U[u_idx(0,0)] = 1, U[u_idx(1,1)] = 1 in the SAME column.
        
        # Let's refactor add_term to allow multiple indices in one column.
        # Or just create a helper add_strassen_term.
        
        # I will rewrite the logic to handle Strassen terms properly.
        pass

    # Redefine construct logic with proper Strassen handling
    col = 0
    
    def add_rank1(u_indices, v_indices, w_indices, vals):
        # u_indices, v_indices, w_indices are lists of indices
        # vals are the values
        # This adds one rank 1 term where U has non-zeros at u_indices, etc.
        nonlocal col
        for idx, val in zip(u_indices, vals):
            U[idx][col] += val
        for idx, val in zip(v_indices, vals):
            V[idx][col] += val
        for idx, val in zip(w_indices, vals):
            W[idx][col] += val
        col += 1

    def add_1x2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        # A: (a_row_offset, a_col_offset+0), (a_row_offset, a_col_offset+1)
        # B: (b_row_offset+0, b_col_offset+0), (b_row_offset+0, b_col_offset+1)
        #    (b_row_offset+1, b_col_offset+0), (b_row_offset+1, b_col_offset+1)
        # C: (c_row_offset, c_col_offset+0), (c_row_offset, c_col_offset+1)
        
        # M1: A00 * B00 -> C00
        add_rank1([a_row_offset * m + a_col_offset], [(b_row_offset) * p + b_col_offset], [(c_col_offset) * n + c_row_offset], [1])
        
        # M2: A01 * B10 -> C00
        add_rank1([a_row_offset * m + (a_col_offset+1)], [(b_row_offset+1) * p + b_col_offset], [(c_col_offset) * n + c_row_offset], [1])
        
        # M3: A00 * B01 -> C01
        add_rank1([a_row_offset * m + a_col_offset], [(b_row_offset) * p + (b_col_offset+1)], [(c_col_offset+1) * n + c_row_offset], [1])
        
        # M4: A01 * B11 -> C01
        add_rank1([a_row_offset * m + (a_col_offset+1)], [(b_row_offset+1) * p + (b_col_offset+1)], [(c_col_offset+1) * n + c_row_offset], [1])

    def add_2x2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        def u(r, c): return (a_row_offset + r) * m + (a_col_offset + c)
        def v(r, c): return (b_row_offset + r) * p + (b_col_offset + c)
        def w(r, c): return (c_col_offset + c) * n + (c_row_offset + r)

        # M1: (a00+a11)(b00+b11)
        # U: u(0,0)=1, u(1,1)=1
        # V: v(0,0)=1, v(1,1)=1
        # W: w(0,0)=1, w(1,1)=1
        add_rank1([u(0,0), u(1,1)], [v(0,0), v(1,1)], [w(0,0), w(1,1)], [1, 1])
        
        # M2: (a10+a11)b00
        # U: u(1,0)=1, u(1,1)=1
        # V: v(0,0)=1
        # W: w(1,0)=1, w(1,1)=-1
        add_rank1([u(1,0), u(1,1)], [v(0,0)], [w(1,0), w(1,1)], [1, 1, 1, -1])
        # Wait, lengths must match.
        # U has 2 non-zeros, V has 1, W has 2.
        # My add_rank1 assumes same length.
        # I need to pad with 0s or handle differently.
        
        # Let's fix add_rank1 to take dicts or handle padding.
        pass

    # Better approach: explicit column assignment.
    col = 0
    def set_col(u_dict, v_dict, w_dict):
        nonlocal col
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val
        col += 1

    def add_1x2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        u00 = a_row_offset * m + a_col_offset
        u01 = a_row_offset * m + (a_col_offset + 1)
        v00 = b_row_offset * p + b_col_offset
        v01 = b_row_offset * p + (b_col_offset + 1)
        v10 = (b_row_offset + 1) * p + b_col_offset
        v11 = (b_row_offset + 1) * p + (b_col_offset + 1)
        w00 = c_col_offset * n + c_row_offset
        w01 = (c_col_offset + 1) * n + c_row_offset
        
        set_col({u00: 1}, {v00: 1}, {w00: 1})
        set_col({u01: 1}, {v10: 1}, {w00: 1})
        set_col({u00: 1}, {v01: 1}, {w01: 1})
        set_col({u01: 1}, {v11: 1}, {w01: 1})

    def add_2x2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        def u(r, c): return (a_row_offset + r) * m + (a_col_offset + c)
        def v(r, c): return (b_row_offset + r) * p + (b_col_offset + c)
        def w(r, c): return (c_col_offset + c) * n + (c_row_offset + r)

        # M1
        set_col({u(0,0): 1, u(1,1): 1}, {v(0,0): 1, v(1,1): 1}, {w(0,0): 1, w(1,1): 1})
        # M2
        set_col({u(1,0): 1, u(1,1): 1}, {v(0,0): 1}, {w(1,0): 1, w(1,1): -1})
        # M3
        set_col({u(0,0): 1}, {v(0,1): 1, v(1,1): -1}, {w(0,1): 1, w(1,1): 1})
        # M4
        set_col({u(1,1): 1}, {v(0,1): -1, v(1,1): 1}, {w(0,0): 1, w(1,1): 1})
        # M5
        set_col({u(0,1): 1, u(1,0): 1}, {v(0,0): 1, v(0,1): 1}, {w(0,0): -1, w(0,1): 1})
        # M6
        set_col({u(0,1): 1, u(0,0): -1}, {v(1,0): 1, v(0,0): 1}, {w(1,0): 1, w(1,1): 1})
        # M7
        set_col({u(1,1): 1, u(0,1): -1}, {v(1,1): 1, v(0,1): 1}, {w(0,0): 1, w(1,1): -1})

    # Block 1: A11 B1j
    # A11: row 0, cols 0,1. B1j: rows 0,1, cols 2j, 2j+1. C1j: row 0, cols 2j, 2j+1.
    for j in range(3):
        add_1x2x2(0, 0, 0, 2*j, 0, 2*j)
        add_1x2x2(0, 2, 2, 2*j, 0, 2*j)
        add_2x2x2(1, 0, 0, 2*j, 1, 2*j)
        add_2x2x2(1, 2, 2, 2*j, 1, 2*j)
        
    return {"U": U, "V": V, "W": W}
