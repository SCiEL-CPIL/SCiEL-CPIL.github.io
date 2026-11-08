import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 7
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))

    # Strategy: 
    # 1. Head: 4x4 * 4x4 using recursive Strassen (49 terms)
    # 2. Mid: 4x4 * 4x2 using embedded Strassen (28 terms)
    # 3. Tail: 4x4 * 4x1 using schoolbook (16 terms)
    # Total Rank: 93

    # Part 1: First 4 columns (4x4 * 4x4)
    def strassen_4x4_head():
        """Generate 49 terms for 4x4 * 4x4 * 4x4 (first 4 cols)"""
        outer = [
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
            ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
            ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
            ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
            ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
        ]
        
        inner = [
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
            ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
            ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
            ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
            ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
        ]
        
        def get_global_idx(block_idx, local_idx, is_A):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            row = br * 2 + lr
            col = bc * 2 + lc
            if is_A:
                return row * m + col
            else:
                return row * p + col
        
        def get_w_idx(block_idx_c, local_idx_c):
            br = block_idx_c // 2
            bc = block_idx_c % 2
            lr = local_idx_c // 2
            lc = local_idx_c % 2
            i = br * 2 + lr
            k = bc * 2 + lc
            return k * n + i
        
        for oa, ob, oc in outer:
            for ia, ib, ic in inner:
                u = [0.0] * (n * m)
                v = [0.0] * (m * p)
                w = [0.0] * (p * n)
                
                for a_blk, a_coeff in oa:
                    for x_loc, x_coeff in ia:
                        idx = get_global_idx(a_blk, x_loc, True)
                        u[idx] += a_coeff * x_coeff
                
                for b_blk, b_coeff in ob:
                    for y_loc, y_coeff in ib:
                        idx = get_global_idx(b_blk, y_loc, False)
                        v[idx] += b_coeff * y_coeff
                
                for c_blk, c_coeff in oc:
                    for z_loc, z_coeff in ic:
                        idx = get_w_idx(c_blk, z_loc)
                        w[idx] += c_coeff * z_coeff
                
                add_term(u, v, w)

    # Part 2: Cols 4, 5 (4x4 * 4x2)
    # We use the 4 outer Strassen terms that are non-zero when B cols 2,3 are 0
    # M1, M2, M4, M7 from the 4x4*4x4 analysis
    def strassen_4x4_mid(k_off):
        """Generate 28 terms for 4x4 * 4x2 (cols k_off, k_off+1)"""
        # k_off is 4
        # We map local cols 0,1 to global cols 4,5
        # We map local rows 0..3 to global rows 0..3
        
        # Outer terms needed:
        # M1: (A00+A11)*(B00+B11) -> B11 is 0, so (A00+A11)*B00
        #     C00 += M1, C11 += M1 (C11 is global col 6,7? No, C11 is local cols 2,3 which are global 6,7? 
        #     Wait, the mid part is 4x4 * 4x2.
        #     B is 4x2. Embedded in 4x4 as B' = [B 0].
        #     B00 is B (cols 0,1 of B'). Global cols 4,5.
        #     B11 is 0 (cols 2,3 of B'). Global cols 6,7.
        #     So M1 = (A00+A11) * B.
        #     C00 is C (cols 0,1 of C'). Global cols 4,5.
        #     C11 is C (cols 2,3 of C'). Global cols 6,7.
        #     We only need C00.
        
        # M2: (A10+A11)*B00 -> C10 += M2, C11 -= M2
        #     C10 is C (cols 0,1). Global 4,5.
        #     We need C10.
        
        # M4: A11*(B10-B00) -> C00 += M4, C10 += M4
        #     B10 is B (cols 0,1). B00 is B (cols 0,1).
        #     Wait, in 4x4 embedding:
        #     B' = [B 0]
        #     B00 = B (rows 0-1, cols 0-1)
        #     B10 = B (rows 2-3, cols 0-1)
        #     B01 = 0
        #     B11 = 0
        #     So M4 = A11 * (B10 - B00).
        #     We need C00 and C10.
        
        # M7: (A01-A11)*(B10+B11) -> C00 += M7
        #     B11=0. So (A01-A11)*B10.
        #     We need C00.
        
        # We need to generate these 4 outer terms, each with 7 inner terms.
        
        outer_mid = [
            # M1
            ([(0,1),(3,1)], [(0,1)], [(0,1)]), # Only C00 needed
            # M2
            ([(2,1),(3,1)], [(0,1)], [(2,1)]), # Only C10 needed
            # M4
            ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]), # C00 and C10 needed
            # M7
            ([(1,1),(3,-1)], [(2,1)], [(0,1)]), # Only C00 needed
        ]
        
        inner = [
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
            ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
            ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
            ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
            ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
        ]
        
        def get_global_idx_mid(block_idx, local_idx, is_A):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            row = br * 2 + lr
            col = bc * 2 + lc
            if is_A:
                return row * m + col
            else:
                # B col offset by k_off
                return row * p + (col + k_off)
        
        def get_w_idx_mid(block_idx_c, local_idx_c):
            br = block_idx_c // 2
            bc = block_idx_c % 2
            lr = local_idx_c // 2
            lc = local_idx_c % 2
            i = br * 2 + lr
            k = bc * 2 + lc
            # W col offset by k_off
            return (k + k_off) * n + i
        
        for oa, ob, oc in outer_mid:
            for ia, ib, ic in inner:
                u = [0.0] * (n * m)
                v = [0.0] * (m * p)
                w = [0.0] * (p * n)
                
                for a_blk, a_coeff in oa:
                    for x_loc, x_coeff in ia:
                        idx = get_global_idx_mid(a_blk, x_loc, True)
                        u[idx] += a_coeff * x_coeff
                
                for b_blk, b_coeff in ob:
                    for y_loc, y_coeff in ib:
                        idx = get_global_idx_mid(b_blk, y_loc, False)
                        v[idx] += b_coeff * y_coeff
                
                for c_blk, c_coeff in oc:
                    for z_loc, z_coeff in ic:
                        idx = get_w_idx_mid(c_blk, z_loc)
                        w[idx] += c_coeff * z_coeff
                
                add_term(u, v, w)

    # Part 3: Col 6 (4x4 * 4x1)
    def schoolbook_tail_1col(k_off):
        """Generate 16 terms for 4x4 * 4x1 (col k_off) using schoolbook"""
        for i in range(n):
            for j in range(m):
                u = [0.0] * (n * m)
                v = [0.0] * (m * p)
                w = [0.0] * (p * n)
                u[i * m + j] = 1.0
                v[j * p + k_off] = 1.0
                w[k_off * n + i] = 1.0
                add_term(u, v, w)
    
    # Execute
    strassen_4x4_head()
    
    if time.time() > deadline:
        pass
    
    strassen_4x4_mid(4)
    
    if time.time() > deadline:
        pass
    
    schoolbook_tail_1col(6)
    
    elapsed = time.time() - start_time
    if elapsed > 140:
        return {"U": [], "V": [], "W": []}
    
    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
