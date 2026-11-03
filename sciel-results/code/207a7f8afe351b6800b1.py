import time
from collections import defaultdict

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 4, 4, 6

    # Strassen 2x2x2 core
    # u, v, w are lists of (index, coeff)
    # Indices for 2x2: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_2x2 = [
        # (u, v, w)
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)], [((0,0),1),((0,1),1),((1,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)], [((1,0),1),((1,1),1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)], [((0,1),1),((1,1),1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)], [((0,0),1),((1,0),1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)], [((0,0),1),((1,0),1)]),
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)], [((1,0),1),((1,1),1)]),
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)], [((0,1),1),((1,1),1)]),
    ]
    
    # Mapping from result position (r,c) to list of (strassen_term_idx, coeff)
    result_map = {
        (0,0): [(0,1), (3,1), (4,-1), (6,1)],
        (0,1): [(2,1), (4,1)],
        (1,0): [(1,1), (3,1)],
        (1,1): [(0,1), (1,-1), (2,1), (5,1)],
    }

    terms = []

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        u = {i: c for i, c in u_coeffs if c != 0}
        v = {i: c for i, c in v_coeffs if c != 0}
        w = {i: c for i, c in w_coeffs if c != 0}
        if u and v and w:
            terms.append((u, v, w))

    # Helper to map block coordinates to global indices
    def a_idx(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (x*2+r)*4 + (y*2+c)

    def b_idx(block, r, c, col_offset=0):
        x, y = int(block[0]), int(block[1])
        return (x*2+r)*6 + (y*2+c+col_offset)

    def c_idx(block, r, c, col_offset=0):
        x, y = int(block[0]), int(block[1])
        return (y*2+c+col_offset)*4 + (x*2+r)

    # Part 1: 4x4x4 using Kronecker product of two 2x2x2 Strassen
    # Big blocks: 00, 01, 10, 11
    big_combos = [
        ([('00',1),('11',1)], [('00',1),('11',1)]),
        ([('10',1),('11',1)], [('00',1)]),
        ([('00',1)], [('01',1),('11',-1)]),
        ([('11',1)], [('10',1),('00',-1)]),
        ([('00',1),('01',1)], [('11',1)]),
        ([('10',1),('00',-1)], [('00',1),('01',1)]),
        ([('01',1),('11',-1)], [('10',1),('11',1)]),
    ]
    
    big_result_map = {
        '00': [(0,1), (3,1), (4,-1), (6,1)],
        '01': [(2,1), (4,1)],
        '10': [(1,1), (3,1)],
        '11': [(0,1), (1,-1), (2,1), (5,1)],
    }

    for big_idx, (a_parts, b_parts) in enumerate(big_combos):
        if time.time() > deadline:
            break
        for s_idx in range(7):
            if time.time() > deadline:
                break
            x_coeffs, y_coeffs, _ = strassen_2x2[s_idx]
            
            u_coeffs = []
            for (rx, cx), cx_c in x_coeffs:
                for a_block, a_sign in a_parts:
                    idx = a_idx(a_block, rx, cx)
                    u_coeffs.append((idx, a_sign * cx_c))
            
            v_coeffs = []
            for (ry, cy), cy_c in y_coeffs:
                for b_block, b_sign in b_parts:
                    idx = b_idx(b_block, ry, cy, 0)
                    v_coeffs.append((idx, b_sign * cy_c))
            
            w_coeffs = []
            for c_block, big_contribs in big_result_map.items():
                big_coeff = 0
                for m_idx, m_c in big_contribs:
                    if m_idx == big_idx:
                        big_coeff = m_c
                        break
                if big_coeff == 0:
                    continue
                for rc, s_contribs in result_map.items():
                    small_coeff = 0
                    for s_check, s_c in s_contribs:
                        if s_check == s_idx:
                            small_coeff = s_c
                            break
                    if small_coeff == 0:
                        continue
                    idx = c_idx(c_block, rc[0], rc[1], 0)
                    w_coeffs.append((idx, big_coeff * small_coeff))
            
            add_term(u_coeffs, v_coeffs, w_coeffs)

    # Part 2: 4x4x2 using Strassen for the 2x2 blocks
    # A blocks: 00, 01, 10, 11
    # B blocks: T (top 2 rows, cols 4-5), B (bottom 2 rows, cols 4-5)
    # C blocks: T (top 2 rows, cols 4-5), B (bottom 2 rows, cols 4-5)
    
    products = [
        ('00', 'T', 'top'),
        ('01', 'B', 'top'),
        ('10', 'T', 'bot'),
        ('11', 'B', 'bot'),
    ]
    
    for a_block, b_block, c_part in products:
        if time.time() > deadline:
            break
        for s_idx in range(7):
            if time.time() > deadline:
                break
            x_coeffs, y_coeffs, _ = strassen_2x2[s_idx]
            
            u_coeffs = []
            for (rx, cx), cx_c in x_coeffs:
                x, y = int(a_block[0]), int(a_block[1])
                idx = (x*2+rx)*4 + (y*2+cx)
                u_coeffs.append((idx, cx_c))
            
            v_coeffs = []
            for (ry, cy), cy_c in y_coeffs:
                if b_block == 'T':
                    idx = ry*6 + (4+cy)
                else:
                    idx = (2+ry)*6 + (4+cy)
                v_coeffs.append((idx, cy_c))
            
            w_coeffs = []
            for rc, s_contribs in result_map.items():
                small_coeff = 0
                for s_check, s_c in s_contribs:
                    if s_check == s_idx:
                        small_coeff = s_c
                        break
                if small_coeff == 0:
                    continue
                if c_part == 'top':
                    idx = (4+rc[1])*4 + rc[0]
                else:
                    idx = (4+rc[1])*4 + (2+rc[0])
                w_coeffs.append((idx, small_coeff))
            
            add_term(u_coeffs, v_coeffs, w_coeffs)

    # Merging
    def merge_by_uv(terms_in):
        groups = defaultdict(list)
        for u, v, w in terms_in:
            key = (tuple(sorted(u.items())), tuple(sorted(v.items())))
            groups[key].append(w)
        merged = []
        for key, w_list in groups.items():
            w_sum = defaultdict(float)
            for w in w_list:
                for i, c in w.items():
                    w_sum[i] += c
            w_sum = {i: c for i, c in w_sum.items() if abs(c) > 1e-9}
            if w_sum:
                u = dict(key[0])
                v = dict(key[1])
                merged.append((u, v, w_sum))
        return merged

    def merge_by_uw(terms_in):
        groups = defaultdict(list)
        for u, v, w in terms_in:
            key = (tuple(sorted(u.items())), tuple(sorted(w.items())))
            groups[key].append(v)
        merged = []
        for key, v_list in groups.items():
            v_sum = defaultdict(float)
            for v in v_list:
                for i, c in v.items():
                    v_sum[i] += c
            v_sum = {i: c for i, c in v_sum.items() if abs(c) > 1e-9}
            if v_sum:
                u = dict(key[0])
                w = dict(key[1])
                merged.append((u, v_sum, w))
        return merged

    def merge_by_vw(terms_in):
        groups = defaultdict(list)
        for u, v, w in terms_in:
            key = (tuple(sorted(v.items())), tuple(sorted(w.items())))
            groups[key].append(u)
        merged = []
        for key, u_list in groups.items():
            u_sum = defaultdict(float)
            for u in u_list:
                for i, c in u.items():
                    u_sum[i] += c
            u_sum = {i: c for i, c in u_sum.items() if abs(c) > 1e-9}
            if u_sum:
                v = dict(key[0])
                w = dict(key[1])
                merged.append((u_sum, v, w))
        return merged

    current = terms
    for _ in range(10):
        if time.time() > deadline:
            break
        merged_uv = merge_by_uv(current)
        merged_uw = merge_by_uw(current)
        merged_vw = merge_by_vw(current)
        best = min([merged_uv, merged_uw, merged_vw], key=len)
        if len(best) >= len(current):
            break
        current = best

    rank = len(current)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u, v, w) in enumerate(current):
        for i, c in u.items():
            U[i][col] = c
        for i, c in v.items():
            V[i][col] = c
        for i, c in w.items():
            W[i][col] = c

    return {"U": U, "V": V, "W": W}
