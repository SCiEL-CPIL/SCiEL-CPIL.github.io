import numpy as np

def construct():
    n, m, p = 3, 4, 8
    
    # Recursive builder
    def build(n, m, p):
        # Returns list of (u, v, w) where u is np.array length n*m, etc.
        if n == 2 and m == 2 and p == 2:
            # Strassen
            terms = []
            # M1
            u = np.array([1, 0, 1, 0])
            v = np.array([1, 1, 0, 0])
            w = np.array([1, 0, 0, 1])
            terms.append((u, v, w))
            # M2
            u = np.array([0, 0, 1, 1])
            v = np.array([1, 0, 0, 0])
            w = np.array([0, 0, 1, 0])
            terms.append((u, v, w))
            # M3
            u = np.array([1, 0, 0, 0])
            v = np.array([0, 1, 0, -1])
            w = np.array([0, 1, 0, 1])
            terms.append((u, v, w))
            # M4
            u = np.array([0, 0, 0, 1])
            v = np.array([-1, 0, 1, 0])
            w = np.array([1, 0, 1, 0])
            terms.append((u, v, w))
            # M5
            u = np.array([1, 1, 0, 0])
            v = np.array([0, 0, 0, 1])
            w = np.array([-1, 0, 0, 0])
            terms.append((u, v, w))
            # M6
            u = np.array([-1, 0, 1, 0])
            v = np.array([1, 1, 0, 0])
            w = np.array([0, 0, 0, 1])
            terms.append((u, v, w))
            # M7
            u = np.array([0, 1, 0, -1])
            v = np.array([0, 0, 1, 1])
            w = np.array([1, 0, 0, 0])
            terms.append((u, v, w))
            return terms

        # Recursive case
        # Strategy:
        # If p is even, split B into two halves.
        # If m is even, split A into two halves?
        # If n is even, split C?
        
        # We use the specific splits derived:
        # M(3,4,8) -> 2 * M(3,4,4) [Split B]
        # M(3,4,4) -> M(2,4,4) + M(1,4,4) [Split A]
        # M(2,4,4) -> 2 * M(2,2,4) [Split A and B]
        # M(2,2,4) -> 2 * M(2,2,2) [Split B]
        # M(1,4,4) -> 4 * M(1,1,1)? No, M(1,4,4) is 4 dot products.
        
        # Let's generalize the split logic to cover these cases.
        
        terms = []
        
        if n == 1 and m == 1 and p == 1:
            u = np.array([1])
            v = np.array([1])
            w = np.array([1])
            terms.append((u, v, w))
            return terms
            
        if n == 1 and m == 4 and p == 4:
            # 4 dot products of length 4
            # C_j = sum_k A_k B_kj
            # A is 1x4, B is 4x4
            # Indices for A: 0..3
            # Indices for B: row k, col j -> k*4+j
            # Indices for C: j -> j
            for j in range(4):
                for k in range(4):
                    u = np.zeros(4)
                    u[k] = 1
                    v = np.zeros(16)
                    v[k*4+j] = 1
                    w = np.zeros(4)
                    w[j] = 1
                    terms.append((u, v, w))
            return terms

        if n == 2 and m == 2 and p == 4:
            # Split B into B1 (2x2) and B2 (2x2)
            # A is 2x2
            # C is 2x4
            # C = A B1 + A B2
            # Part 1: M(2,2,2) on A, B1
            # Part 2: M(2,2,2) on A, B2
            
            # Local terms for M(2,2,2)
            local_terms = build(2, 2, 2)
            
            for (lu, lv, lw) in local_terms:
                # Map to global
                # u is same (A is 2x2)
                gu = lu.copy()
                
                # v is B1 (2x2) mapped to B (2x4)
                # B1 is cols 0,1 of B.
                gv = np.zeros(8)
                gv[0:4] = lv # B1 indices 0..3
                # B2 is cols 2,3 of B.
                # Wait, B is 2x4. Indices 0..7.
                # B1 is rows 0,1 cols 0,1. Indices 0,1,2,3.
                # B2 is rows 0,1 cols 2,3. Indices 4,5,6,7.
                
                # We need to add terms for B2 separately.
                # This logic is getting complex.
                pass
                
        # Let's implement a more generic recursive splitter.
        
        # Helper to embed a local tensor into a global one
        def embed(terms_local, n_loc, m_loc, p_loc, row_offset_u, col_offset_u, row_offset_v, col_offset_v, row_offset_w, col_offset_w):
            # This is too complex for dynamic offsets.
            # Better to build the full vectors directly in the recursion.
            pass

        # Let's stick to the specific cases and implement them explicitly.
        
        if n == 2 and m == 2 and p == 4:
            # Split B into B1 (2x2) and B2 (2x2)
            # A (2x2)
            # C (2x4)
            # C1 = A B1 (2x2), C2 = A B2 (2x2)
            
            local_terms = build(2, 2, 2)
            
            for (lu, lv, lw) in local_terms:
                # Case 1: B1
                gu = lu.copy()
                gv = np.zeros(8)
                gv[0:4] = lv
                gw = np.zeros(8) # 2x4 = 8 elements
                # lw is 2x2 = 4 elements. C1 is left half of C.
                # C indices: 0,1 (row 0), 2,3 (row 1)?
                # W index convention: k*n + i.
                # For 2x2, k in 0,1, i in 0,1.
                # k=0: i=0,1 -> 0,1
                # k=1: i=0,1 -> 2,3
                # So C1 occupies indices 0,1,2,3 in W?
                # C is 2x4. k in 0,1. i in 0,1.
                # C[k, i] index is k*2 + i.
                # C1 is C[:, 0:2].
                # C1[k, i] is C[k, i]. Index k*2+i.
                # So lw maps directly to gw[0:4].
                gw[0:4] = lw
                terms.append((gu, gv, gw))
                
                # Case 2: B2
                gu2 = lu.copy()
                gv2 = np.zeros(8)
                # B2 is right half of B.
                # B is 2x4. B2 is cols 2,3.
                # B indices: row j, col k. j*p + k.
                # B1: j in 0,1, k in 0,1. Indices 0,1,2,3.
                # B2: j in 0,1, k in 2,3. Indices 4,5,6,7.
                gv2[4:8] = lv
                gw2 = np.zeros(8)
                # C2 is C[:, 2:4].
                # C2[k, i] is C[k, i+2].
                # Index of C[k, i+2] is k*2 + (i+2) = k*2 + i + 2.
                # So lw[i] maps to gw2[k*2+i+2].
                # lw is flattened 2x2.
                # lw[0] -> C[0,0] -> C2[0,0] -> C[0,2] -> idx 2
                # lw[1] -> C[0,1] -> C2[0,1] -> C[0,3] -> idx 3
                # lw[2] -> C[1,0] -> C2[1,0] -> C[1,2] -> idx 6
                # lw[3] -> C[1,1] -> C2[1,1] -> C[1,3] -> idx 7
                gw2[2:4] = lw[0:2]
                gw2[6:8] = lw[2:4]
                terms.append((gu2, gv2, gw2))
            return terms

        if n == 2 and m == 4 and p == 4:
            # Split A into A1 (2x2) and A2 (2x2)
            # Split B into B1 (2x4) and B2 (2x4)
            # C = A1 B1 + A2 B2
            # A1 B1 is M(2,2,4)
            # A2 B2 is M(2,2,4)
            
            local_terms = build(2, 2, 4)
            
            for (lu, lv, lw) in local_terms:
                # Case 1: A1, B1
                # A1 is left half of A (2x4).
                # A indices: i*m + j. i in 0,1, j in 0,1.
                # A1: j in 0,1. Indices 0,1,2,3.
                # A2: j in 2,3. Indices 4,5,6,7.
                
                # B1 is top half of B (4x4).
                # B indices: j*p + k. j in 0,1, k in 0,3.
                # B1: j in 0,1. Indices 0..7.
                # B2: j in 2,3. Indices 8..15.
                
                # C is 2x4.
                # lw is 2x4.
                
                gu = np.zeros(8)
                gu[0:4] = lu
                gv = np.zeros(16)
                gv[0:8] = lv
                gw = lw.copy()
                terms.append((gu, gv, gw))
                
                # Case 2: A2, B2
                gu2 = np.zeros(8)
                gu2[4:8] = lu
                gv2 = np.zeros(16)
                gv2[8:16] = lv
                gw2 = lw.copy()
                terms.append((gu2, gv2, gw2))
            return terms

        if n == 3 and m == 4 and p == 4:
            # Split A into A1 (2x4) and A2 (1x4)
            # C = A1 B + A2 B
            # A1 B is M(2,4,4)
            # A2 B is M(1,4,4)
            
            terms1 = build(2, 4, 4)
            terms2 = build(1, 4, 4)
            
            for (lu, lv, lw) in terms1:
                # A1 is top 2 rows of A (3x4).
                # A indices: i*4 + j.
                # A1: i in 0,1. Indices 0..7.
                # A2: i in 2. Indices 8..11.
                
                # B is 4x4. Same.
                # C is 3x4.
                # lw is 2x4.
                # C1 is top 2 rows of C.
                # C indices: k*3 + i.
                # C1: i in 0,1.
                # For k=0: i=0,1 -> 0,1.
                # For k=1: i=0,1 -> 3,4.
                # For k=2: i=0,1 -> 6,7.
                # lw is flattened 2x4.
                # lw[0] -> C[0,0] -> idx 0
                # lw[1] -> C[0,1] -> idx 1
                # lw[2] -> C[0,2] -> idx 2
                # lw[3] -> C[0,3] -> idx 3
                # lw[4] -> C[1,0] -> idx 3? No.
                # C is 3x4. k in 0,1,2. i in 0,1,2.
                # Index k*3 + i.
                # C1 is rows 0,1 of C.
                # C1[k, i] is C[k, i] for k in 0,1.
                # So lw[k*4 + i] maps to C[k, i].
                # C[k, i] index is k*3 + i.
                
                gw = np.zeros(12)
                for k in range(2):
                    for i in range(4):
                        gw[k*3 + i] = lw[k*4 + i]
                        
                gu = np.zeros(12)
                gu[0:8] = lu
                gv = lv.copy()
                terms.append((gu, gv, gw))
                
            for (lu, lv, lw) in terms2:
                # A2 is bottom 1 row of A.
                # lu is length 4.
                # B is 4x4.
                # C is 3x4.
                # lw is 1x4.
                # C2 is bottom 1 row of C.
                # C2[i] is C[2, i].
                # Index 2*3 + i = 6 + i.
                
                gw = np.zeros(12)
                for i in range(4):
                    gw[6 + i] = lw[i]
                    
                gu = np.zeros(12)
                gu[8:12] = lu
                gv = lv.copy()
                terms.append((gu, gv, gw))
                
            return terms

        if n == 3 and m == 4 and p == 8:
            # Split B into B1 (4x4) and B2 (4x4)
            # C = A B1 + A B2
            # A B1 is M(3,4,4)
            # A B2 is M(3,4,4)
            
            local_terms = build(3, 4, 4)
            
            for (lu, lv, lw) in local_terms:
                # B1 is left half of B (4x8).
                # B indices: j*8 + k.
                # B1: k in 0,1,2,3.
                # B2: k in 4,5,6,7.
                
                # C is 3x8.
                # lw is 3x4.
                # C1 is left half of C.
                # C1[k, i] is C[k, i] for k in 0..3.
                # C indices: k*3 + i.
                # C1 indices: k*3 + i for k in 0..3.
                # lw is flattened 3x4.
                # lw[k*4 + i] maps to C[k, i].
                
                gw = np.zeros(24)
                for k in range(4):
                    for i in range(3):
                        gw[k*3 + i] = lw[k*4 + i]
                        
                gu = lu.copy()
                gv = np.zeros(32)
                gv[0:16] = lv
                terms.append((gu, gv, gw))
                
                # Case 2: B2
                gu2 = lu.copy()
                gv2 = np.zeros(32)
                # B2 is right half.
                # B1 indices 0..15.
                # B2 indices 16..31.
                gv2[16:32] = lv
                gw2 = np.zeros(24)
                # C2 is right half of C.
                # C2[k, i] is C[k, i+4].
                # Index k*3 + (i+4) = k*3 + i + 4.
                for k in range(4):
                    for i in range(3):
                        gw2[k*3 + i + 4] = lw[k*4 + i]
                terms.append((gu2, gv2, gw2))
            return terms

        raise ValueError(f"Unimplemented case: {n}x{m}x{p}")

    terms = build(3, 4, 8)
    
    U = np.zeros((12, len(terms)))
    V = np.zeros((32, len(terms)))
    W = np.zeros((24, len(terms)))
    
    for idx, (u, v, w) in enumerate(terms):
        U[:, idx] = u
        V[:, idx] = v
        W[:, idx] = w
        
    return {"U": U, "V": V, "W": W}
