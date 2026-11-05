import time
import sys

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Nested Strassen: 7*7 = 49 for 4x4x4, 16 for 4x4x1 = 65 total
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_list, v_list, w_list):
        nonlocal col
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c
        col += 1
    
    # Define the 7 Strassen combinations for 2x2x2
    # For P = [[p00,p01],[p10,p11]], Q = [[q00,q01],[q10,q11]]
    # s1 = (p00+p11)(q00+q11)
    # s2 = (p10+p11)(q00)
    # s3 = p00(q01-q11)
    # s4 = p11(q10-q00)
    # s5 = (p00+p01)(q11)
    # s6 = (p10-p00)(q00+q01)
    # s7 = (p01-p11)(q10+q11)
    # Result: R00 = s1+s4-s5+s7, R01 = s3+s5, R10 = s2+s4, R11 = s1-s2+s3+s6
    
    # For each super-term (M1..M7), define the linear combinations of A blocks and B blocks
    # Then for each inner Strassen term, expand to scalar A and B entries
    
    # Block definitions:
    # A00: i in {0,1}, j in {0,1}
    # A01: i in {0,1}, j in {2,3}
    # A10: i in {2,3}, j in {0,1}
    # A11: i in {2,3}, j in {2,3}
    # B00: j in {0,1}, k in {0,1}
    # B01: j in {0,1}, k in {2,3}
    # B10: j in {2,3}, k in {0,1}
    # B11: j in {2,3}, k in {2,3}
    
    # Super-term definitions: (p_blocks, q_blocks, c_contrib)
    # p_blocks: coefficients for A00, A01, A10, A11
    # q_blocks: coefficients for B00, B01, B10, B11
    # c_contrib: which C blocks this contributes to, with signs
    # C00: i in {0,1}, k in {0,1}
    # C01: i in {0,1}, k in {2,3}
    # C10: i in {2,3}, k in {0,1}
    # C11: i in {2,3}, k in {2,3}
    
    super_terms = [
        # M1: (A00+A11)(B00+B11)
        {
            'p_blocks': [1, 0, 0, 1],   # A00 + A11
            'q_blocks': [1, 0, 0, 1],   # B00 + B11
            'c_contrib': [(0, 1), (3, 1)]  # C00 +1, C11 +1
        },
        # M2: (A10+A11)(B00)
        {
            'p_blocks': [0, 0, 1, 1],   # A10 + A11
            'q_blocks': [1, 0, 0, 0],   # B00
            'c_contrib': [(2, 1), (3, -1)]  # C10 +1, C11 -1
        },
        # M3: (A00)(B01-B11)
        {
            'p_blocks': [1, 0, 0, 0],   # A00
            'q_blocks': [0, 1, 0, -1],  # B01 - B11
            'c_contrib': [(1, 1), (3, 1)]  # C01 +1, C11 +1
        },
        # M4: (A11)(B10-B00)
        {
            'p_blocks': [0, 0, 0, 1],   # A11
            'q_blocks': [-1, 0, 1, 0],  # B10 - B00
            'c_contrib': [(0, 1), (2, 1)]  # C00 +1, C10 +1
        },
        # M5: (A00+A01)(B11)
        {
            'p_blocks': [1, 1, 0, 0],   # A00 + A01
            'q_blocks': [0, 0, 0, 1],   # B11
            'c_contrib': [(0, -1), (1, 1)]  # C00 -1, C01 +1
        },
        # M6: (A10-A00)(B00+B01)
        {
            'p_blocks': [-1, 0, 1, 0],  # A10 - A00
            'q_blocks': [1, 1, 0, 0],   # B00 + B01
            'c_contrib': [(3, 1)]  # C11 +1
        },
        # M7: (A01-A11)(B10+B11)
        {
            'p_blocks': [0, 1, 0, -1],  # A01 - A11
            'q_blocks': [0, 0, 1, 1],   # B10 + B11
            'c_contrib': [(0, 1)]  # C00 +1
        },
    ]
    
    # Inner Strassen terms for 2x2x2
    # For P = [[p00,p01],[p10,p11]], Q = [[q00,q01],[q10,q11]]
    # s1: p_elems=[(0,1),(3,1)], q_elems=[(0,1),(3,1)], r_elems=[(0,1),(3,1)]
    # s2: p_elems=[(2,1),(3,1)], q_elems=[(0,1)], r_elems=[(2,1),(3,-1)]
    # s3: p_elems=[(0,1)], q_elems=[(1,1),(3,-1)], r_elems=[(1,1),(3,1)]
    # s4: p_elems=[(3,1)], q_elems=[(2,1),(0,-1)], r_elems=[(0,1),(2,1)]
    # s5: p_elems=[(0,1),(1,1)], q_elems=[(3,1)], r_elems=[(0,-1),(1,1)]
    # s6: p_elems=[(2,1),(0,-1)], q_elems=[(0,1),(1,1)], r_elems=[(3,1)]
    # s7: p_elems=[(1,1),(3,-1)], q_elems=[(2,1),(3,1)], r_elems=[(0,1)]
    
    # p_elem_idx: 0=p00, 1=p01, 2=p10, 3=p11
    # q_elem_idx: 0=q00, 1=q01, 2=q10, 3=q11
    # r_elem_idx: 0=r00, 1=r01, 2=r10, 3=r11
    
    inner_strassen = [
        # s1
        {'p_elems': [(0, 1), (3, 1)], 'q_elems': [(0, 1), (3, 1)], 'r_elems': [(0, 1), (3, 1)]},
        # s2
        {'p_elems': [(2, 1), (3, 1)], 'q_elems': [(0, 1)], 'r_elems': [(2, 1), (3, -1)]},
        # s3
        {'p_elems': [(0, 1)], 'q_elems': [(1, 1), (3, -1)], 'r_elems': [(1, 1), (3, 1)]},
        # s4
        {'p_elems': [(3, 1)], 'q_elems': [(2, 1), (0, -1)], 'r_elems': [(0, 1), (2, 1)]},
        # s5
        {'p_elems': [(0, 1), (1, 1)], 'q_elems': [(3, 1)], 'r_elems': [(0, -1), (1, 1)]},
        # s6
        {'p_elems': [(2, 1), (0, -1)], 'q_elems': [(0, 1), (1, 1)], 'r_elems': [(3, 1)]},
        # s7
        {'p_elems': [(1, 1), (3, -1)], 'q_elems': [(2, 1), (3, 1)], 'r_elems': [(0, 1)]},
    ]
    
    # Helper: get global A indices for a P element
    # p_elem_idx: 0=p00, 1=p01, 2=p10, 3=p11
    # p_blocks: coefficients for A00, A01, A10, A11
    def get_p_indices(p_elem_idx, p_blocks):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_blocks):
            if coeff == 0:
                continue
            if b_idx == 0:  # A00
                gi, gj = r, c
            elif b_idx == 1:  # A01
                gi, gj = r, 2 + c
            elif b_idx == 2:  # A10
                gi, gj = 2 + r, c
            elif b_idx == 3:  # A11
                gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res
    
    # Helper: get global B indices for a Q element
    # q_elem_idx: 0=q00, 1=q01, 2=q10, 3=q11
    # q_blocks: coefficients for B00, B01, B10, B11
    def get_q_indices(q_elem_idx, q_blocks):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_blocks):
            if coeff == 0:
                continue
            if b_idx == 0:  # B00
                gj, gk = r, c
            elif b_idx == 1:  # B01
                gj, gk = r, 2 + c
            elif b_idx == 2:  # B10
                gj, gk = 2 + r, c
            elif b_idx == 3:  # B11
                gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res
    
    # Helper: get global C indices for an R element
    # r_elem_idx: 0=r00, 1=r01, 2=r10, 3=r11
    # c_contrib: list of (c_block_idx, sign)
    # c_block_idx: 0=C00, 1=C01, 2=C10, 3=C11
    def get_c_indices(r_elem_idx, c_contrib):
        r = r_elem_idx // 2
        c = r_elem_idx % 2
        res = []
        for c_block_idx, sign in c_contrib:
            if c_block_idx == 0:  # C00: i in {0,1}, k in {0,1}
                gi, gk = r, c
            elif c_block_idx == 1:  # C01: i in {0,1}, k in {2,3}
                gi, gk = r, 2 + c
            elif c_block_idx == 2:  # C10: i in {2,3}, k in {0,1}
                gi, gk = 2 + r, c
            elif c_block_idx == 3:  # C11: i in {2,3}, k in {2,3}
                gi, gk = 2 + r, 2 + c
            res.append((gk, gi, sign))
        return res
    
    # Process each super-term and each inner Strassen term
    for super_term in super_terms:
        p_blocks = super_term['p_blocks']
        q_blocks = super_term['q_blocks']
        c_contrib = super_term['c_contrib']
        
        for inner in inner_strassen:
            p_elems = inner['p_elems']
            q_elems = inner['q_elems']
            r_elems = inner['r_elems']
            
            # Build U list: combine all P elements
            u_list = []
            for p_elem_idx, coeff in p_elems:
                p_indices = get_p_indices(p_elem_idx, p_blocks)
                for gi, gj, c in p_indices:
                    u_list.append((gi, gj, coeff * c))
            
            # Build V list: combine all Q elements
            v_list = []
            for q_elem_idx, coeff in q_elems:
                q_indices = get_q_indices(q_elem_idx, q_blocks)
                for gj, gk, c in q_indices:
                    v_list.append((gj, gk, coeff * c))
            
            # Build W list: combine all R elements
            w_list = []
            for r_elem_idx, coeff in r_elems:
                c_indices = get_c_indices(r_elem_idx, c_contrib)
                for gk, gi, c in c_indices:
                    w_list.append((gk, gi, coeff * c))
            
            # Merge same indices
            u_dict = {}
            for gi, gj, c in u_list:
                key = (gi, gj)
                u_dict[key] = u_dict.get(key, 0) + c
            u_list = [(gi, gj, c) for (gi, gj), c in u_dict.items() if c != 0]
            
            v_dict = {}
            for gj, gk, c in v_list:
                key = (gj, gk)
                v_dict[key] = v_dict.get(key, 0) + c
            v_list = [(gj, gk, c) for (gj, gk), c in v_dict.items() if c != 0]
            
            w_dict = {}
            for gk, gi, c in w_list:
                key = (gk, gi)
                w_dict[key] = w_dict.get(key, 0) + c
            w_list = [(gk, gi, c) for (gk, gi), c in w_dict.items() if c != 0]
            
            if u_list and v_list and w_list:
                add_term(u_list, v_list, w_list)
    
    # Part 2: 4x4 * 4x1 (rank 16) - schoolbook
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1  # B index: row j, col 4
            W[4 * 4 + i][col] = 1  # C index: k=4, i. W[k*n+i] = W[4*4+i]
            col += 1
    
    return {"U": U, "V": V, "W": W}
