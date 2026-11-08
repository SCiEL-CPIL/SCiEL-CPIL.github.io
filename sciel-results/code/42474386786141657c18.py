"""Direct rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication."""
import time

DIMENSIONS = [4, 4, 5]

def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Rank 65: 49 terms for 4x4*4x4 (Strassen x Strassen) + 16 terms for 4x4*4x1 (schoolbook)
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
    
    # Block coefficient definitions for P (A side) and Q (B side)
    p_blocks_all = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1],  # M7: A01-A11
    ]
    
    q_blocks_all = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [-1, 0, 1, 0],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1],   # M7: B10+B11
    ]
    
    # Inner Strassen term definitions
    inner_defs = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # t0: (P00+P11)(Q00+Q11)
        ([(2, 1), (3, 1)], [(0, 1)]),            # t1: (P10+P11)Q00
        ([(0, 1)], [(1, 1), (3, -1)]),           # t2: P00(Q01-Q11)
        ([(3, 1)], [(2, 1), (0, -1)]),           # t3: P11(Q10-Q00)
        ([(0, 1), (1, 1)], [(3, 1)]),            # t4: (P00+P01)Q11
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),   # t5: (P10-P00)(Q00+Q01)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),   # t6: (P01-P11)(Q10+Q11)
    ]
    
    # Contribution of inner term t to S (2x2 result of P*Q)
    s_contrib_all = [
        [1, 0, 0, 1],   # t0 -> S00, S11
        [0, 0, 1, -1],  # t1 -> S10, -S11
        [0, 1, 0, 1],   # t2 -> S01, S11
        [1, 0, 1, 0],   # t3 -> S00, S10
        [-1, 1, 0, 0],  # t4 -> -S00, S01
        [0, 0, 0, 1],   # t5 -> S11
        [1, 0, 0, 0],   # t6 -> S00
    ]
    
    # Contribution of super term s to C blocks (C00, C01, C10, C11)
    c_block_contrib_all = [
        [1, 0, 0, 1],   # M1 -> C00, C11
        [0, 0, 1, -1],  # M2 -> C10, -C11
        [0, 1, 0, 1],   # M3 -> C01, C11
        [1, 0, 1, 0],   # M4 -> C00, C10
        [-1, 1, 0, 0],  # M5 -> -C00, C01
        [0, 0, 0, 1],   # M6 -> C11
        [1, 0, 0, 0],   # M7 -> C00
    ]
    
    # Precompute global indices for P elements per block
    def get_p_global(p_idx, b_idx):
        r = p_idx // 2
        c = p_idx % 2
        if b_idx == 0:
            return (r, c)
        elif b_idx == 1:
            return (r, 2 + c)
        elif b_idx == 2:
            return (2 + r, c)
        else:
            return (2 + r, 2 + c)
    
    def get_q_global(q_idx, b_idx):
        r = q_idx // 2
        c = q_idx % 2
        if b_idx == 0:
            return (r, c)
        elif b_idx == 1:
            return (r, 2 + c)
        elif b_idx == 2:
            return (2 + r, c)
        else:
            return (2 + r, 2 + c)
    
    def get_c_global(cb, r, c):
        if cb == 0:
            return (r, c)
        elif cb == 1:
            return (r, 2 + c)
        elif cb == 2:
            return (2 + r, c)
        else:
            return (2 + r, 2 + c)
    
    # Precompute all term data for the 49 Strassen terms
    strassen_terms = []
    for s in range(7):
        for t in range(7):
            p_blocks = p_blocks_all[s]
            q_blocks = q_blocks_all[s]
            p_elems, q_elems = inner_defs[t]
            s_contrib = s_contrib_all[t]
            c_block_contrib = c_block_contrib_all[s]
            
            u_list = []
            v_list = []
            w_list = []
            
            for p_idx, p_coeff in p_elems:
                for b_idx, b_coeff in enumerate(p_blocks):
                    if b_coeff == 0:
                        continue
                    gi, gj = get_p_global(p_idx, b_idx)
                    u_list.append((gi, gj, b_coeff * p_coeff))
            
            for q_idx, q_coeff in q_elems:
                for b_idx, b_coeff in enumerate(q_blocks):
                    if b_coeff == 0:
                        continue
                    gj, gk = get_q_global(q_idx, b_idx)
                    v_list.append((gj, gk, b_coeff * q_coeff))
            
            for sp in range(4):
                sc = s_contrib[sp]
                if sc == 0:
                    continue
                s_r = sp // 2
                s_c = sp % 2
                for cb in range(4):
                    cc = c_block_contrib[cb]
                    if cc == 0:
                        continue
                    total_c = sc * cc
                    if total_c == 0:
                        continue
                    gi, gk = get_c_global(cb, s_r, s_c)
                    w_list.append((gk, gi, total_c))
            
            strassen_terms.append((u_list, v_list, w_list))
    
    # Apply the 49 Strassen terms
    for u_list, v_list, w_list in strassen_terms:
        add_term(col, u_list, v_list, w_list)
        col += 1
    
    # Part 2: 4x4 * 4x1 schoolbook for the 5th column (16 terms)
    for i in range(4):
        for j in range(4):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(col, u_list, v_list, w_list)
            col += 1
    
    elapsed = time.time() - t_start
    if elapsed > 120:
        pass
    
    return {"U": U, "V": V, "W": W}
