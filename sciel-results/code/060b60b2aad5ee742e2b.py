"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
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
    
    # Strassen for 4x4 * 4x4
    inner_s_contrib = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    super_c_contrib = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    def get_p_elements(p_blocks, r, c):
        res = []
        for b_idx, coeff in enumerate(p_blocks):
            if coeff == 0: continue
            if b_idx == 0: gi, gj = r, c
            elif b_idx == 1: gi, gj = r, 2 + c
            elif b_idx == 2: gi, gj = 2 + r, c
            elif b_idx == 3: gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res

    def get_q_elements(q_blocks, r, c):
        res = []
        for b_idx, coeff in enumerate(q_blocks):
            if coeff == 0: continue
            if b_idx == 0: gj, gk = r, c
            elif b_idx == 1: gj, gk = r, 2 + c
            elif b_idx == 2: gj, gk = 2 + r, c
            elif b_idx == 3: gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res

    super_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1]),
        ([-1, 0, 1, 0], [1, 1, 0, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1])
    ]
    
    inner_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)]),
        ([(0, 1)], [(1, 1), (3, -1)]),
        ([(3, 1)], [(2, 1), (0, -1)]),
        ([(0, 1), (1, 1)], [(3, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)])
    ]

    for s_idx in range(7):
        if time.time() > deadline: break
        p_blocks, q_blocks = super_terms[s_idx]
        c_contrib = super_c_contrib[s_idx]
        
        for t_idx in range(7):
            if time.time() > deadline: break
            p_elems, q_elems = inner_terms[t_idx]
            s_contrib = inner_s_contrib[t_idx]
            
            u_list, v_list, w_list = [], [], []
            
            for p_idx, p_coeff in p_elems:
                r, c = p_idx // 2, p_idx % 2
                for i, j, c_val in get_p_elements(p_blocks, r, c):
                    u_list.append((i, j, c_val * p_coeff))
                    
            for q_idx, q_coeff in q_elems:
                r, c = q_idx // 2, q_idx % 2
                for j, k, c_val in get_q_elements(q_blocks, r, c):
                    v_list.append((j, k, c_val * q_coeff))
                    
            for s_pos in range(4):
                s_coeff = s_contrib[s_pos]
                if s_coeff == 0: continue
                s_r, s_c = s_pos // 2, s_pos % 2
                for c_b in range(4):
                    c_coeff = c_contrib[c_b]
                    if c_coeff == 0: continue
                    total_coeff = s_coeff * c_coeff
                    if total_coeff == 0: continue
                    if c_b == 0: gi, gk = s_r, s_c
                    elif c_b == 1: gi, gk = s_r, 2 + s_c
                    elif c_b == 2: gi, gk = 2 + s_r, s_c
                    else: gi, gk = 2 + s_r, 2 + s_c
                    w_list.append((gk, gi, total_coeff))
            
            add_term(col, u_list, v_list, w_list)
            col += 1

    # Schoolbook for 5th column
    for i in range(4):
        for j in range(4):
            if time.time() > deadline: break
            add_term(col, [(i, j, 1)], [(j, 4, 1)], [(4, i, 1)])
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
