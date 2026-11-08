import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = 4, 4, 7
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))

    def zero_u(): return [0] * (n * m)
    def zero_v(): return [0] * (m * p)
    def zero_w(): return [0] * (p * n)

    # Helper to add a scalar to a vector at an index
    def add_to_vec(vec, idx, val):
        vec[idx] += val

    # --- Part 1: 4x4x4 Strassen (Cols 0-3) ---
    # Outer Strassen terms for 4x4 * 4x4
    # Blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    outer_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]), # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]), # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]), # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]), # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]), # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]), # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]), # M7
    ]
    
    # Inner Strassen terms for 2x2 * 2x2
    # Local blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    inner_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]), # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]), # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]), # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]), # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]), # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]), # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]), # M7
    ]
    
    def get_idx_A(block, local):
        br, bc = block // 2, block % 2
        lr, lc = local // 2, local % 2
        r, c = br*2+lr, bc*2+lc
        return r*m + c
        
    def get_idx_B(block, local):
        br, bc = block // 2, block % 2
        lr, lc = local // 2, local % 2
        r, c = br*2+lr, bc*2+lc
        return r*p + c
        
    def get_idx_C(block, local):
        br, bc = block // 2, block % 2
        lr, lc = local // 2, local % 2
        r, c = br*2+lr, bc*2+lc
        return c*n + r # W[k*n+i] -> col*n + row

    for oa, ob, oc in outer_terms:
        for ia, ib, ic in inner_terms:
            u = zero_u()
            v = zero_v()
            w = zero_w()
            
            for (la, da) in ia:
                for (ba, ca) in oa:
                    idx = get_idx_A(ba, la)
                    add_to_vec(u, idx, ca * da)
            
            for (lb, db) in ib:
                for (bb, cb) in ob:
                    idx = get_idx_B(bb, lb)
                    add_to_vec(v, idx, cb * db)
            
            for (lc, dc) in ic:
                for (bc, cc) in oc:
                    idx = get_idx_C(bc, lc)
                    add_to_vec(w, idx, cc * dc)
            
            add_term(u, v, w)
            
    # --- Part 2: 4x4x2 Strassen (Cols 4-5) ---
    # We use 2x2 Strassen for each 2x2 block of A and corresponding B/C blocks.
    # B cols 4-5 are shared.
    def strassen_2x2_block(i_off, j_off, k_off):
        # i_off, j_off, k_off are top-left corners
        # A block: rows i_off..i_off+1, cols j_off..j_off+1
        # B block: rows j_off..j_off+1, cols k_off..k_off+1
        # C block: rows i_off..i_off+1, cols k_off..k_off+1
        
        # 7 terms
        # M1
        u = zero_u(); v = zero_v(); w = zero_w()
        u[i_off*m + j_off] = 1; u[(i_off+1)*m + (j_off+1)] = 1
        v[j_off*p + k_off] = 1; v[(j_off+1)*p + (k_off+1)] = 1
        w[k_off*n + i_off] = 1; w[(k_off+1)*n + (i_off+1)] = 1
        add_term(u, v, w)
        
        # M2
        u = zero_u(); v = zero_v(); w = zero_w()
        u[(i_off+1)*m + j_off] = 1; u[(i_off+1)*m + (j_off+1)] = 1
        v[j_off*p + k_off] = 1
        w[k_off*n + (i_off+1)] = 1; w[(k_off+1)*n + (i_off+1)] = -1
        add_term(u, v, w)
        
        # M3
        u = zero_u(); v = zero_v(); w = zero_w()
        u[i_off*m + j_off] = 1
        v[j_off*p + (k_off+1)] = 1; v[(j_off+1)*p + (k_off+1)] = -1
        w[(k_off+1)*n + i_off] = 1; w[(k_off+1)*n + (i_off+1)] = 1
        add_term(u, v, w)
        
        # M4
        u = zero_u(); v = zero_v(); w = zero_w()
        u[(i_off+1)*m + (j_off+1)] = 1
        v[(j_off+1)*p + k_off] = 1; v[j_off*p + k_off] = -1
        w[k_off*n + i_off] = 1; w[k_off*n + (i_off+1)] = 1
        add_term(u, v, w)
        
        # M5
        u = zero_u(); v = zero_v(); w = zero_w()
        u[i_off*m + j_off] = 1; u[i_off*m + (j_off+1)] = 1
        v[(j_off+1)*p + (k_off+1)] = 1
        w[(k_off+1)*n + i_off] = 1; w[k_off*n + i_off] = -1
        add_term(u, v, w)
        
        # M6
        u = zero_u(); v = zero_v(); w = zero_w()
        u[(i_off+1)*m + j_off] = 1; u[i_off*m + j_off] = -1
        v[j_off*p + k_off] = 1; v[j_off*p + (k_off+1)] = 1
        w[(k_off+1)*n + (i_off+1)] = 1
        add_term(u, v, w)
        
        # M7
        u = zero_u(); v = zero_v(); w = zero_w()
        u[i_off*m + (j_off+1)] = 1; u[(i_off+1)*m + (j_off+1)] = -1
        v[(j_off+1)*p + k_off] = 1; v[(j_off+1)*p + (k_off+1)] = 1
        w[k_off*n + i_off] = 1
        add_term(u, v, w)

    for i_b in range(2):
        for j_b in range(2):
            strassen_2x2_block(i_b*2, j_b*2, 4)
            
    # --- Part 3: 4x4x1 Schoolbook (Col 6) ---
    for i in range(n):
        for j in range(m):
            u = zero_u(); v = zero_v(); w = zero_w()
            u[i*m + j] = 1
            v[j*p + 6] = 1
            w[6*n + i] = 1
            add_term(u, v, w)
            
    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
