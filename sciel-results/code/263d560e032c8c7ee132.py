"""Matrix multiplication 4x4x5 - rank 65 decomposition."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
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
    
    def get_indices(super_idx, inner_idx):
        u_list = []
        v_list = []
        w_list = []
        
        if super_idx == 0: p_blocks = [1, 0, 0, 1]
        elif super_idx == 1: p_blocks = [0, 0, 1, 1]
        elif super_idx == 2: p_blocks = [1, 0, 0, 0]
        elif super_idx == 3: p_blocks = [0, 0, 0, 1]
        elif super_idx == 4: p_blocks = [1, 1, 0, 0]
        elif super_idx == 5: p_blocks = [-1, 0, 1, 0]
        elif super_idx == 6: p_blocks = [0, 1, 0, -1]
        
        if super_idx == 0: q_blocks = [1, 0, 0, 1]
        elif super_idx == 1: q_blocks = [1, 0, 0, 0]
        elif super_idx == 2: q_blocks = [0, 1, 0, -1]
        elif super_idx == 3: q_blocks = [-1, 0, 1, 0]
        elif super_idx == 4: q_blocks = [0, 0, 0, 1]
        elif super_idx == 5: q_blocks = [1, 1, 0, 0]
        elif super_idx == 6: q_blocks = [0, 0, 1, 1]
        
        if inner_idx == 0:
            p_elems = [(0, 1), (3, 1)]
            q_elems = [(0, 1), (3, 1)]
            s_contrib = [1, 0, 0, 1]
        elif inner_idx == 1:
            p_elems = [(2, 1), (3, 1)]
            q_elems = [(0, 1)]
            s_contrib = [0, 0, 1, -1]
        elif inner_idx == 2:
            p_elems = [(0, 1)]
            q_elems = [(1, 1), (3, -1)]
            s_contrib = [0, 1, 0, 1]
        elif inner_idx == 3:
            p_elems = [(3, 1)]
            q_elems = [(2, 1), (0, -1)]
            s_contrib = [1, 0, 1, 0]
        elif inner_idx == 4:
            p_elems = [(0, 1), (1, 1)]
            q_elems = [(3, 1)]
            s_contrib = [-1, 1, 0, 0]
        elif inner_idx == 5:
            p_elems = [(2, 1), (0, -1)]
            q_elems = [(0, 1), (1, 1)]
            s_contrib = [0, 0, 0, 1]
        elif inner_idx == 6:
            p_elems = [(1, 1), (3, -1)]
            q_elems = [(2, 1), (3, 1)]
            s_contrib = [1, 0, 0, 0]
        
        for p_idx, p_coeff in p_elems:
            r = p_idx // 2
            c = p_idx % 2
            for b_idx, coeff in enumerate(p_blocks):
                if coeff == 0: continue
                if b_idx == 0: gi, gj = r, c
                elif b_idx == 1: gi, gj = r, 2+c
                elif b_idx == 2: gi, gj = 2+r, c
                elif b_idx == 3: gi, gj = 2+r, 2+c
                u_list.append((gi, gj, coeff * p_coeff))
        
        for q_idx, q_coeff in q_elems:
            r = q_idx // 2
            c = q_idx % 2
            for b_idx, coeff in enumerate(q_blocks):
                if coeff == 0: continue
                if b_idx == 0: gj, gk = r, c
                elif b_idx == 1: gj, gk = r, 2+c
                elif b_idx == 2: gj, gk = 2+r, c
                elif b_idx == 3: gj, gk = 2+r, 2+c
                v_list.append((gj, gk, coeff * q_coeff))
        
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
                
                if c_b == 0: gi, gk = s_r, s_c
                elif c_b == 1: gi, gk = s_r, 2+s_c
                elif c_b == 2: gi, gk = 2+s_r, s_c
                elif c_b == 3: gi, gk = 2+s_r, 2+s_c
                
                w_list.append((gk, gi, total_coeff))
                
        return u_list, v_list, w_list

    for s in range(7):
        for t in range(7):
            u_list, v_list, w_list = get_indices(s, t)
            add_term(col, u_list, v_list, w_list)
            col += 1

    for i in range(4):
        for j in range(4):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(col, u_list, v_list, w_list)
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
