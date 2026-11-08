"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Rank 49 for 4x4x4 part, Rank 4 for 4x4x1 part. Total 53.
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 16 x 53
    V = [[0] * rank for _ in range(m * p)]  # 20 x 53
    W = [[0] * rank for _ in range(p * n)]  # 20 x 53
    
    # Helper to add a rank-1 term
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0
    
    # --- Part 1: 4x4 * 4x4 using Strassen (Rank 49) ---
    # Reusing the logic from the previous program but ensuring it only touches k=0..3
    
    def get_super_term_indices(super_idx, inner_idx):
        u_list = []
        v_list = []
        w_list = []
        
        # Block definitions for A and B (first 4 cols)
        # A00: i<2, j<2. A01: i<2, j>=2. A10: i>=2, j<2. A11: i>=2, j>=2.
        # B00: j<2, k<2. B01: j<2, k>=2. B10: j>=2, k<2. B11: j>=2, k>=2.
        
        if super_idx == 0: # M1
            p_blocks = [1, 0, 0, 1] # A00 + A11
            q_blocks = [1, 0, 0, 1] # B00 + B11
        elif super_idx == 1: # M2
            p_blocks = [0, 0, 1, 1] # A10 + A11
            q_blocks = [1, 0, 0, 0] # B00
        elif super_idx == 2: # M3
            p_blocks = [1, 0, 0, 0] # A00
            q_blocks = [0, 1, 0, -1] # B01 - B11
        elif super_idx == 3: # M4
            p_blocks = [0, 0, 0, 1] # A11
            q_blocks = [-1, 0, 1, 0] # B10 - B00
        elif super_idx == 4: # M5
            p_blocks = [1, 1, 0, 0] # A00 + A01
            q_blocks = [0, 0, 0, 1] # B11
        elif super_idx == 5: # M6
            p_blocks = [-1, 0, 1, 0] # A10 - A00
            q_blocks = [1, 1, 0, 0] # B00 + B01
        elif super_idx == 6: # M7
            p_blocks = [0, 1, 0, -1] # A01 - A11
            q_blocks = [0, 0, 1, 1] # B10 + B11
        else:
            return u_list, v_list, w_list
            
        def get_p_element(p_elem_idx):
            res = []
            r = p_elem_idx // 2
            c = p_elem_idx % 2
            for b_idx, coeff in enumerate(p_blocks):
                if coeff == 0: continue
                if b_idx == 0: # A00
                    gi = r; gj = c
                elif b_idx == 1: # A01
                    gi = r; gj = 2 + c
                elif b_idx == 2: # A10
                    gi = 2 + r; gj = c
                elif b_idx == 3: # A11
                    gi = 2 + r; gj = 2 + c
                res.append((gi, gj, coeff))
            return res
            
        def get_q_element(q_elem_idx):
            res = []
            r = q_elem_idx // 2
            c = q_elem_idx % 2
            for b_idx, coeff in enumerate(q_blocks):
                if coeff == 0: continue
                if b_idx == 0: # B00
                    gj = r; gk = c
                elif b_idx == 1: # B01
                    gj = r; gk = 2 + c
                elif b_idx == 2: # B10
                    gj = 2 + r; gk = c
                elif b_idx == 3: # B11
                    gj = 2 + r; gk = 2 + c
                res.append((gj, gk, coeff))
            return res
            
        if inner_idx == 0:
            p_elems = [(0, 1), (3, 1)]
            q_elems = [(0, 1), (3, 1)]
        elif inner_idx == 1:
            p_elems = [(2, 1), (3, 1)]
            q_elems = [(0, 1)]
        elif inner_idx == 2:
            p_elems = [(0, 1)]
            q_elems = [(1, 1), (3, -1)]
        elif inner_idx == 3:
            p_elems = [(3, 1)]
            q_elems = [(2, 1), (0, -1)]
        elif inner_idx == 4:
            p_elems = [(0, 1), (1, 1)]
            q_elems = [(3, 1)]
        elif inner_idx == 5:
            p_elems = [(2, 1), (0, -1)]
            q_elems = [(0, 1), (1, 1)]
        elif inner_idx == 6:
            p_elems = [(1, 1), (3, -1)]
            q_elems = [(2, 1), (3, 1)]
        else:
            return u_list, v_list, w_list
            
        for p_idx, p_coeff in p_elems:
            elems = get_p_element(p_idx)
            for i, j, c in elems:
                u_list.append((i, j, c * p_coeff))
                
        for q_idx, q_coeff in q_elems:
            elems = get_q_element(q_idx)
            for j, k, c in elems:
                v_list.append((j, k, c * q_coeff))
                
        if inner_idx == 0: s_contrib = [1, 0, 0, 1]
        elif inner_idx == 1: s_contrib = [0, 0, 1, -1]
        elif inner_idx == 2: s_contrib = [0, 1, 0, 1]
        elif inner_idx == 3: s_contrib = [1, 0, 1, 0]
        elif inner_idx == 4: s_contrib = [-1, 1, 0, 0]
        elif inner_idx == 5: s_contrib = [0, 0, 0, 1]
        elif inner_idx == 6: s_contrib = [1, 0, 0, 0]
        else: s_contrib = [0,0,0,0]
        
        c_block_contrib = [0, 0, 0, 0]
        if super_idx == 0: c_block_contrib = [1, 0, 0, 1]
        elif super_idx == 1: c_block_contrib = [0, 0, 1, -1]
        elif super_idx == 2: c_block_contrib = [0, 1, 0, 1]
        elif super_idx == 3: c_block_contrib = [1, 0, 1, 0]
        elif super_idx == 4: c_block_contrib = [-1, 1, 0, 0]
        elif super_idx == 5: c_block_contrib = [0, 0, 0, 1]
        elif super_idx == 6: c_block_contrib = [1, 0, 0, 0]
        
        for s_pos in range(4):
            s_coeff = s_contrib[s_pos]
            if s_coeff == 0: continue
            s_r = s_pos // 2
            s_c = s_pos % 2
            for c_b in range(4):
                c_coeff = c_block_contrib[c_b]
                if c_coeff == 0: continue
                total_coeff = s_coeff * c_coeff
                if total_coeff == 0: continue
                if c_b == 0:
                    gi = s_r; gk = s_c
                elif c_b == 1:
                    gi = s_r; gk = 2 + s_c
                elif c_b == 2:
                    gi = 2 + s_r; gk = s_c
                elif c_b == 3:
                    gi = 2 + s_r; gk = 2 + s_c
                w_list.append((gk, gi, total_coeff))
                
        return u_list, v_list, w_list

    for s in range(7):
        for t in range(7):
            u_list, v_list, w_list = get_super_term_indices(s, t)
            add_term(col, u_list, v_list, w_list)
            col += 1

    # --- Part 2: 4x4 * 4x1 using Optimized Rank 4 ---
    # r in 0..3
    # V[j*5+4, r] = 1 if j==r
    # U[i*4+j, r] = 1 if j==r
    # W[4*4+i, r] = 1 (for all i? No, W depends on i)
    # Wait, W[4*4+i, r] should be 1 for all i?
    # If W[4*4+i, r] = 1, then term r contributes to all i.
    # U[i*4+r, r] = 1.
    # V[r*5+4, r] = 1.
    # Product: 1 * 1 * 1 = 1 at (a=i*4+r, b=r*5+4, c=4*4+i).
    # This is correct for all i.
    
    for r in range(4):
        u_list = []
        v_list = []
        w_list = []
        
        # U: i in 0..3, j=r
        for i in range(4):
            u_list.append((i, r, 1))
            
        # V: j=r, k=4
        v_list.append((r, 4, 1))
        
        # W: k=4, i in 0..3
        for i in range(4):
            w_list.append((4, i, 1))
            
        add_term(col, u_list, v_list, w_list)
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
