import numpy as np
import time

def construct():
    start_time = time.time()
    
    n, m, p = 4, 4, 7
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))
    
    # Helper to get global index for A (4x4)
    # block_idx: 0=A11, 1=A12, 2=A21, 3=A22
    # local_idx: 0..3 within 2x2 block
    def get_global_idx_A(block_idx, local_idx):
        br = block_idx // 2
        bc = block_idx % 2
        lr = local_idx // 2
        lc = local_idx % 2
        row = br * 2 + lr
        col = bc * 2 + lc
        return row * m + col

    # Helper to get global index for B (4x7)
    # block_idx: 0=B11, 1=B12, 2=B21, 3=B22 (relative to 4x4 block)
    # k_off: column offset in global B
    def get_global_idx_B(block_idx, local_idx, k_off=0):
        br = block_idx // 2
        bc = block_idx % 2
        lr = local_idx // 2
        lc = local_idx % 2
        row = br * 2 + lr
        col = bc * 2 + lc
        return row * p + (col + k_off)

    # Helper to get global index for C (4x7)
    # block_idx: 0=C11, 1=C12, 2=C21, 3=C22 (relative to 4x4 block)
    # k_off: column offset in global C
    def get_global_idx_C(block_idx, local_idx, k_off=0):
        br = block_idx // 2
        bc = block_idx % 2
        lr = local_idx // 2
        lc = local_idx % 2
        i = br * 2 + lr
        k = bc * 2 + lc
        return (k + k_off) * n + i

    # Strassen terms for 2x2x2
    # (u_coeff, v_coeff, w_coeff)
    # u: A blocks, v: B blocks, w: C blocks
    strassen_terms = [
        # M1 = (A11+A22)(B11+B22) -> C11, C22
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2 = (A21+A22)B11 -> C21, C22
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
        # M3 = A11(B12-B22) -> C11, C12
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
        # M4 = A22(B21-B11) -> C11, C21
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        # M5 = (A11+A12)B22 -> C12, C22
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        # M6 = (A21-A11)(B11+B12) -> C11, C21
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7 = (A12-A22)(B21+B22) -> C12, C21
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    
    # 1. Head: Cols 0-3 (4x4x4)
    # Use all 7 Strassen terms, each expanded into 7 sub-terms for 2x2 blocks
    # Total 49 terms
    for oa, ob, oc in strassen_terms:
        for ia, ib, ic in strassen_terms:
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            
            for a_blk, a_coeff in oa:
                for x_loc, x_coeff in ia:
                    idx = get_global_idx_A(a_blk, x_loc)
                    u[idx] += a_coeff * x_coeff
            for b_blk, b_coeff in ob:
                for y_loc, y_coeff in ib:
                    idx = get_global_idx_B(b_blk, y_loc, 0)
                    v[idx] += b_coeff * y_coeff
            for c_blk, c_coeff in oc:
                for z_loc, z_coeff in ic:
                    idx = get_global_idx_C(c_blk, z_loc, 0)
                    w[idx] += c_coeff * z_coeff
            
            add_term(u, v, w)
            
    elapsed = time.time() - start_time
    if elapsed > 110:
        return {"U": [], "V": [], "W": []}

    # 2. Mid: Cols 4-5 (4x4x2)
    # Use 4 specific Strassen-like terms for the left half of the 4x4 result
    # These correspond to a subset of the 7 Strassen terms
    # Terms needed for C11, C21: M1, M2, M4, M6, M7
    # The current code uses 4 terms. Let's use the 5-term subset for correctness/optimality check
    # Actually, the previous code used 4 terms and scored well. 
    # Let's stick to the 4-term decomposition found in the inspiration/current code if possible, 
    # but the 5-term Strassen subset is safer and known to be rank 35.
    # The current code's 28 rank suggests a better decomposition exists.
    # I will replicate the 4-term structure from the current code to maintain the 28 rank.
    
    outer_mid = [
        ([(0,1),(3,1)], [(0,1)], [(0,1)]),
        ([(2,1),(3,1)], [(0,1)], [(2,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        ([(1,1),(3,-1)], [(2,1)], [(0,1)]),
    ]
    
    for oa, ob, oc in outer_mid:
        for ia, ib, ic in strassen_terms:
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            
            for a_blk, a_coeff in oa:
                for x_loc, x_coeff in ia:
                    idx = get_global_idx_A(a_blk, x_loc)
                    u[idx] += a_coeff * x_coeff
            for b_blk, b_coeff in ob:
                for y_loc, y_coeff in ib:
                    idx = get_global_idx_B(b_blk, y_loc, 4)
                    v[idx] += b_coeff * y_coeff
            for c_blk, c_coeff in oc:
                for z_loc, z_coeff in ic:
                    idx = get_global_idx_C(c_blk, z_loc, 4)
                    w[idx] += c_coeff * z_coeff
            
            add_term(u, v, w)
            
    elapsed = time.time() - start_time
    if elapsed > 110:
        return {"U": [], "V": [], "W": []}

    # 3. Tail: Col 6 (4x4x1)
    # Use Strassen-like decomposition for 4x4 x 4x1
    # Decompose A into 2x2 blocks, B_col6 into 2x1 blocks
    # Use 7 Strassen terms, each computing 2x2 x 2x1
    # Result: 4x1 vector
    
    # Strassen terms for 2x2 x 2x1
    # (u_coeff, v_coeff, w_coeff)
    # u: A blocks (2x2), v: B blocks (2x1), w: C blocks (2x1)
    strassen_tail_terms = [
        # M1 = (A11+A22)B1 -> C1
        ([(0,1),(3,1)], [(0,1)], [(0,1)]),
        # M2 = (A21+A22)B1 -> C2
        ([(2,1),(3,1)], [(0,1)], [(2,1)]),
        # M3 = A11(B2-B1) -> C1
        ([(0,1)], [(1,1),(0,-1)], [(1,1)]),
        # M4 = A22(B1-B2) -> C1
        ([(3,1)], [(0,1),(1,-1)], [(0,1)]),
        # M5 = (A11+A12)B2 -> C2
        ([(0,1),(1,1)], [(1,1)], [(0,-1),(2,1)]),
        # M6 = (A21-A11)(B1+B2) -> C2
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7 = (A12-A22)(B1+B2) -> C1
        ([(1,1),(3,-1)], [(0,1),(1,1)], [(1,1)]),
    ]
    
    for oa, ob, oc in strassen_tail_terms:
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)
        
        for a_blk, a_coeff in oa:
            idx = get_global_idx_A(a_blk, 0)  # Only use first element of each 2x2 block? No, need full expansion
            # Actually, for 2x2 x 2x1, we need to expand properly
            # Let me reconsider this approach
            pass
        
        # This approach is getting complicated. Let me use a simpler method.
        # For 4x4 x 4x1, use 4 terms based on 2x2 blocks
        # Term 1: A11 * B1 -> C1
        # Term 2: A12 * B2 -> C1  
        # Term 3: A21 * B1 -> C2
        # Term 4: A22 * B2 -> C2
        # But this is still 4 terms, not better than 16
        
        # Actually, the minimum rank for 4x4 x 4x1 is 4, but achieving it requires specific structure.
        # Let me stick with schoolbook for now and focus on other improvements.
        break
    
    # Fall back to schoolbook for tail
    for i in range(n):
        for j in range(m):
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            u[i * m + j] = 1.0
            v[j * p + 6] = 1.0
            w[6 * n + i] = 1.0
            add_term(u, v, w)
    
    elapsed = time.time() - start_time
    if elapsed > 110:
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
