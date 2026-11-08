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
    # 1. For the first 4 columns (4x4 * 4x4), use recursive Strassen (49 terms)
    # 2. For the last 3 columns (4x4 * 4x3), use a direct decomposition
    
    # Part 1: First 4 columns using recursive Strassen
    # We'll build this using block decomposition
    # A is 4x4, B_first is 4x4, C_first is 4x4
    
    def strassen_4x4_head():
        """Generate 49 terms for 4x4 * 4x4 * 4x4 (first 4 cols)"""
        # Outer Strassen 2x2 (where each "element" is a 2x2 block)
        # A blocks: A00, A01, A10, A11
        # B blocks: B00, B01, B10, B11
        # C blocks: C00, C01, C10, C11
        
        # We'll generate the 49 terms by combining outer and inner Strassen
        # For efficiency, we'll pre-compute the coefficient patterns
        
        # Outer Strassen terms (for 2x2 matrix multiplication of blocks)
        # Each term: (A_coeffs, B_coeffs, C_coeffs) where coeffs are lists of (block_idx, coeff)
        outer = [
            # M1: (A00+A11)*(B00+B11) -> C00+=M1, C11+=M1
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
            # M2: (A10+A11)*B00 -> C10+=M2, C11-=M2
            ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
            # M3: A00*(B01-B11) -> C01+=M3, C11+=M3
            ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
            # M4: A11*(B10-B00) -> C00+=M4, C10+=M4
            ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
            # M5: (A00+A01)*B11 -> -C00, +C01
            ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
            # M6: (A10-A00)*(B00+B01) -> C11+=M6
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
            # M7: (A01-A11)*(B10+B11) -> C00+=M7
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
        ]
        
        # Inner Strassen terms (for 2x2 * 2x2 * 2x2)
        inner = [
            # M1: (x00+x11)*(y00+y11) -> z00+=M1, z11+=M1
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
            # M2: (x10+x11)*y00 -> z10+=M2, z11-=M2
            ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
            # M3: x00*(y01-y11) -> z01+=M3, z11+=M3
            ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
            # M4: x11*(y10-y00) -> z00+=M4, z10+=M4
            ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
            # M5: (x00+x01)*y11 -> -z00, +z01
            ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
            # M6: (x10-x00)*(y00+y01) -> z11+=M6
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
            # M7: (x01-x11)*(y10+y11) -> z00+=M7
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
        ]
        
        def get_global_idx(block_idx, local_idx, is_A):
            """
            Get global flat index.
            block_idx: 0..3 (block position in 4x4)
            local_idx: 0..3 (position within 2x2 block)
            is_A: True for U (A), False for V (B)
            """
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            
            row = br * 2 + lr
            col = bc * 2 + lc
            
            if is_A:
                # U: A[i,j] -> index i*m + j
                return row * m + col
            else:
                # V: B[i,j] -> index i*p + j
                # For head, j is 0..3
                return row * p + col
        
        def get_w_idx(block_idx_c, local_idx_c):
            """
            Get W flat index for C[k,i] with trace convention W[k*n+i].
            block_idx_c: block position in 4x4 C (rows)
            local_idx_c: position within 2x2 block
            """
            br = block_idx_c // 2
            bc = block_idx_c % 2
            lr = local_idx_c // 2
            lc = local_idx_c % 2
            
            i = br * 2 + lr  # row of C
            k = bc * 2 + lc  # col of C
            
            return k * n + i
        
        for oa, ob, oc in outer:
            for ia, ib, ic in inner:
                u = [0.0] * (n * m)
                v = [0.0] * (m * p)
                w = [0.0] * (p * n)
                
                # Accumulate U
                for a_blk, a_coeff in oa:
                    for x_loc, x_coeff in ia:
                        idx = get_global_idx(a_blk, x_loc, True)
                        u[idx] += a_coeff * x_coeff
                
                # Accumulate V
                for b_blk, b_coeff in ob:
                    for y_loc, y_coeff in ib:
                        idx = get_global_idx(b_blk, y_loc, False)
                        v[idx] += b_coeff * y_coeff
                
                # Accumulate W
                for c_blk, c_coeff in oc:
                    for z_loc, z_coeff in ic:
                        idx = get_w_idx(c_blk, z_loc)
                        w[idx] += c_coeff * z_coeff
                
                # Convert to int if possible
                add_term([float(x) for x in u], [float(x) for x in v], [float(x) for x in w])
    
    # Part 2: Last 3 columns (4x4 * 4x3)
    # We'll use a direct approach
    # For each of the 3 columns, we can use Strassen-like or direct decomposition
    # Let's try: split 4x4 into 2x2 blocks, multiply by 2x1 or 2x3 columns
    
    # Actually, let's use a simple but effective approach:
    # For cols 4,5,6, we treat them as separate 4x4 * 4x1 problems
    # Each 4x4 * 4x1 can be done with 16 schoolbook terms
    # Total: 3 * 16 = 48 terms
    # Total rank: 49 + 48 = 97
    
    # But we can do better. Let's group cols 4,5 as 4x4 * 4x2
    # And col 6 as 4x4 * 4x1
    
    # For 4x4 * 4x2: split into 2 blocks of 2x2 * 2x2 * 2x1? No.
    # Let's use: 4x4 * 4x2 can be done by treating it as 4x4 * [4x1 | 4x1]
    # Or use Strassen on rows.
    
    # Actually, let's just use schoolbook for all 3 tail columns to be safe and fast
    # That gives 49 + 48 = 97, which is worse than the current 93.
    
    # The current program uses 93. Let's replicate that structure.
    # Current: 49 (head) + 28 (tail cols 4,5) + 16 (tail col 6) = 93
    
    # For tail cols 4,5: 4 blocks of 2x2 * 2x2 * 2x2 using Strassen = 4*7 = 28
    # For tail col 6: schoolbook = 16
    
    def strassen_tail_2cols(k_off):
        """Generate 28 terms for 4x4 * 4x2 (cols k_off, k_off+1) using block Strassen"""
        for i_b in range(2):
            for j_b in range(2):
                i_off = i_b * 2
                j_off = j_b * 2
                _strassen_2x2(i_off, j_off, k_off)
    
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
    
    def _strassen_2x2(i_off, j_off, k_off, a_scale=1.0, b_scale=1.0, c_scale=1.0):
        """
        Adds 7 Strassen terms for a 2x2 * 2x2 block multiplication.
        """
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
        u[i_off * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v[j_off * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w[k_off * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u[:], v[:], w[:])

        # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[(i_off+1) * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v[j_off * p + k_off] = b_scale
        w[k_off * n + (i_off+1)] = c_scale
        w[(k_off+1) * n + (i_off+1)] = -c_scale
        add_term(u, v, w)

        # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[i_off * m + j_off] = a_scale
        v[j_off * p + (k_off+1)] = b_scale
        v[(j_off+1) * p + (k_off+1)] = -b_scale
        w[(k_off+1) * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v[(j_off+1) * p + k_off] = b_scale
        v[j_off * p + k_off] = -b_scale
        w[k_off * n + i_off] = c_scale
        w[k_off * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        # M5 = (A11+A12)*B22 -> C12 += M5, C11 -= M5
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[i_off * m + j_off] = a_scale
        u[i_off * m + (j_off+1)] = a_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w[(k_off+1) * n + i_off] = c_scale
        w[k_off * n + i_off] = -c_scale
        add_term(u, v, w)

        # M6 = (A21-A11)*(B11+B12) -> C22 += M6
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[(i_off+1) * m + j_off] = a_scale
        u[i_off * m + j_off] = -a_scale
        v[j_off * p + k_off] = b_scale
        v[j_off * p + (k_off+1)] = b_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        # M7 = (A12-A22)*(B21+B22) -> C11 += M7
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[i_off * m + (j_off+1)] = a_scale
        u[(i_off+1) * m + (j_off+1)] = -a_scale
        v[(j_off+1) * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w[k_off * n + i_off] = c_scale
        add_term(u, v, w)
    
    # Execute
    strassen_4x4_head()
    
    if time.time() > deadline:
        pass  # Continue if fast enough
    
    strassen_tail_2cols(4)  # cols 4, 5
    
    if time.time() > deadline:
        pass
    
    schoolbook_tail_1col(6)  # col 6
    
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
