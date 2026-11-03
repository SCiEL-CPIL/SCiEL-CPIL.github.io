"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6
    
    # We will decompose the tensor T_{i,j,k} = delta for (A_{i,j} * B_{j,k}).
    # Indices:
    # i in [0, n)
    # j in [0, m)
    # k in [0, p)
    # U is n*m x rank, indexed by (i, j)
    # V is m*p x rank, indexed by (j, k)
    # W is p*n x rank, indexed by (k, i)
    
    U = []
    V = []
    W = []
    
    def add_term(u_vals, v_vals, w_vals):
        """
        u_vals: dict {(i,j): coeff}
        v_vals: dict {(j,k): coeff}
        w_vals: dict {(k,i): coeff}
        """
        U.append(u_vals)
        V.append(v_vals)
        W.append(w_vals)
    
    # --- Part 1: Decompose the 4x4x4 block (k in 0..3) using Strassen-like rank 49 ---
    # We can use the fact that 4x4x4 tensor has rank <= 49 via recursive Strassen.
    # Actually, the exact rank of 4x4x4 is not fully known, but 49 is a valid upper bound from 2-level Strassen.
    # Level 1: Split into 2x2 blocks of size 2.
    # 2x2x2 has rank 7.
    # 4x4x4 = (2x2) x (2x2) x (2x2).
    # This yields 7 * 7 * 7 = 343? No, Strassen reduces multiplication.
    # The 4x4x4 tensor can be decomposed into 49 rank-1 terms? 
    # Let's check: Standard Strassen for 4x4x4 uses 7 multiplications of size 2x2x2.
    # Each 2x2x2 uses 7 rank-1 terms.
    # So 7 * 7 = 49 rank-1 terms.
    
    # We need to map the local 2x2x2 coordinates to global 4x4x4 coordinates.
    # Let A be 4x4, split into A11, A12, A21, A22 (each 2x2).
    # Let B be 4x4, split into B11, B12, B21, B22 (each 2x2).
    # Let C be 4x4, split into C11, C12, C21, C22 (each 2x2).
    
    # Strassen for 4x4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Now, each M_s is a 2x2x2 multiplication tensor.
    # We can decompose each M_s into 7 rank-1 terms for 2x2x2.
    
    # 2x2x2 Strassen decomposition (rank 7):
    # Inputs a, b (2x2), Output c (2x2).
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # We need to define the 7 vectors for the 2x2x2 case.
    # Let's index a, b, c by (r, col) in 0..1.
    # For a specific M_s, we have coefficients for a, b, c.
    # Let's define a function to add the 7 terms for a 2x2x2 sub-block.
    
    def add_2x2x2_strassen(a_offsets, b_offsets, c_offsets, sign=1):
        """
        Adds the 7 rank-1 terms for a 2x2x2 multiplication.
        a_offsets: dict {(r,c): (i,j)} mapping local 2x2 a indices to global (i,j)
        b_offsets: dict {(r,c): (j,k)} mapping local 2x2 b indices to global (j,k)
        c_offsets: dict {(r,c): (k,i)} mapping local 2x2 c indices to global (k,i)
        sign: multiplies all coefficients
        """
        # The 7 vectors for 2x2x2:
        # v1: a=(1,1,1,1)? No, a is 2x2.
        # Let's list u, v, w for each of the 7 terms.
        # u is in R^4 (a-space), v in R^4 (b-space), w in R^4 (c-space).
        
        # Term 1: m1 = (a00+a11)(b00+b11)
        u1 = {(0,0):1, (1,1):1}
        v1 = {(0,0):1, (1,1):1}
        w1 = {(0,0):1, (1,1):1} # c00 and c11 get +1
        
        # Term 2: m2 = (a10+a11)b00
        u2 = {(1,0):1, (1,1):1}
        v2 = {(0,0):1}
        w2 = {(1,0):1, (1,1):1} # c10 and c11 get +1 (from c21=m2+m4, c22=m1-m2+m3+m6)
        # Wait, let's be precise with c coefficients.
        # c00: m1, m4, -m5, m7
        # c01: m3, m5
        # c10: m2, m4
        # c11: m1, -m2, m3, m6
        
        # Let's define the 7 (u,v,w) tuples for 2x2x2.
        # u is coeff for a, v for b, w for c.
        
        terms_2x2x2 = [
            # 1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # 2
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}), # c10: +1, c11: -1
            # 3
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),  # c01: +1, c11: +1
            # 4
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),   # c00: +1, c10: +1
            # 5
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),   # c00: -1, c01: +1
            # 6
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),   # c11: +1
            # 7
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})    # c00: +1
        ]
        
        for u, v, w in terms_2x2x2:
            global_u = {}
            global_v = {}
            global_w = {}
            
            for (lr, lc), coeff in u.items():
                gi, gj = a_offsets[(lr, lc)]
                key = (gi, gj)
                global_u[key] = global_u.get(key, 0) + coeff * sign
            
            for (lr, lc), coeff in v.items():
                gj, gk = b_offsets[(lr, lc)]
                key = (gj, gk)
                global_v[key] = global_v.get(key, 0) + coeff * sign
            
            for (lr, lc), coeff in w.items():
                gk, gi = c_offsets[(lr, lc)]
                key = (gk, gi)
                global_w[key] = global_w.get(key, 0) + coeff * sign
            
            if global_u and global_v and global_w:
                add_term(global_u, global_v, global_w)
    
    # Define offsets for the 4 2x2 blocks of A, B, C.
    # A blocks:
    # A11: i=0,1 j=0,1
    # A12: i=0,1 j=2,3
    # A21: i=2,3 j=0,1
    # A22: i=2,3 j=2,3
    
    # B blocks:
    # B11: j=0,1 k=0,1
    # B12: j=0,1 k=2,3
    # B21: j=2,3 k=0,1
    # B22: j=2,3 k=2,3
    
    # C blocks:
    # C11: k=0,1 i=0,1
    # C12: k=0,1 i=2,3
    # C21: k=2,3 i=0,1
    # C22: k=2,3 i=2,3
    
    # Helper to create offset dict for a block.
    def make_offsets(base_i=0, base_j=0, base_k=0, is_a=False, is_b=False, is_c=False):
        # is_a: (i,j)
        # is_b: (j,k)
        # is_c: (k,i)
        off = {}
        for lr in range(2):
            for lc in range(2):
                if is_a:
                    off[(lr, lc)] = (base_i + lr, base_j + lc)
                elif is_b:
                    off[(lr, lc)] = (base_j + lr, base_k + lc)
                elif is_c:
                    off[(lr, lc)] = (base_k + lr, base_i + lc)
        return off
    
    # Strassen formulas for 4x4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to add the 7 terms for each M_s.
    # But M_s is a product of sums of blocks.
    # E.g., M1 = A11*B11 + A11*B22 + A22*B11 + A22*B22.
    # Each of these is a 2x2x2 multiplication.
    # So M1 contributes 4 * 7 = 28 rank-1 terms?
    # No, the rank-1 decomposition of M1 is the sum of the rank-1 decompositions of its parts.
    # Since the parts are disjoint in terms of which A and B blocks they involve, we can just add the terms.
    
    # Let's define the 7 terms for each M_s explicitly.
    
    # M1: (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # Each is a 2x2x2 multiplication contributing to C11 and C22.
    # C11 += M1, C22 += M1.
    # So for A11B11, the C offset is for C11 (k=0,1 i=0,1) and C22 (k=2,3 i=2,3)?
    # Wait, Strassen says C11 = M1 + ... and C22 = M1 - ...
    # So M1 contributes to both C11 and C22.
    
    # Let's just add the 7 terms for each of the 4 sub-products in M1, with the correct C offsets.
    
    def add_subprod(a_block, b_block, c_blocks):
        """
        a_block: e.g., (0,0) for A11
        b_block: e.g., (0,0) for B11
        c_blocks: list of (c_block, sign) e.g., [( (0,0), 1 ), ( (1,1), 1 )]
        """
        ai, aj = a_block
        bi, bj = b_block
        a_off = make_offsets(base_i=ai*2, base_j=aj*2, is_a=True)
        b_off = make_offsets(base_j=bi*2, base_k=bj*2, is_b=True)
        
        # We need to add the 7 terms for this 2x2x2 sub-product.
        # The c offsets are different for each c_block.
        # We can call add_2x2x2_strassen for each c_block?
        # No, the u and v are the same, only w changes.
        # So we can add 7 terms for the first c_block, and 7 terms for the second, etc.
        # This is correct because the tensor is linear.
        
        for cb, sign in c_blocks:
            ci, cj = cb
            c_off = make_offsets(base_k=ci*2, base_i=cj*2, is_c=True)
            add_2x2x2_strassen(a_off, b_off, c_off, sign=1)
    
    # M1: (A11+A22)(B11+B22)
    # C11 += M1, C22 += M1
    add_subprod((0,0), (0,0), [((0,0), 1), ((1,1), 1)]) # A11 B11
    add_subprod((0,0), (1,1), [((0,0), 1), ((1,1), 1)]) # A11 B22
    add_subprod((1,1), (0,0), [((0,0), 1), ((1,1), 1)]) # A22 B11
    add_subprod((1,1), (1,1), [((0,0), 1), ((1,1), 1)]) # A22 B22
    
    # M2: (A21+A22)B11
    # C21 += M2, C22 -= M2
    add_subprod((1,0), (0,0), [((1,0), 1), ((1,1), -1)]) # A21 B11
    add_subprod((1,1), (0,0), [((1,0), 1), ((1,1), -1)]) # A22 B11
    
    # M3: A11(B12-B22)
    # C12 += M3, C22 += M3
    add_subprod((0,0), (0,1), [((0,1), 1), ((1,1), 1)]) # A11 B12
    add_subprod((0,0), (1,1), [((0,1), -1), ((1,1), -1)]) # A11 (-B22)
    
    # M4: A22(B21-B11)
    # C11 += M4, C21 += M4
    add_subprod((1,1), (1,0), [((0,0), 1), ((1,0), 1)]) # A22 B21
    add_subprod((1,1), (0,0), [((0,0), -1), ((1,0), -1)]) # A22 (-B11)
    
    # M5: (A11+A12)B22
    # C11 -= M5, C12 += M5
    add_subprod((0,0), (1,1), [((0,0), -1), ((0,1), 1)]) # A11 B22
    add_subprod((0,1), (1,1), [((0,0), -1), ((0,1), 1)]) # A12 B22
    
    # M6: (A21-A11)(B11+B12)
    # C22 += M6
    add_subprod((1,0), (0,0), [((1,1), 1)]) # A21 B11
    add_subprod((1,0), (0,1), [((1,1), 1)]) # A21 B12
    add_subprod((0,0), (0,0), [((1,1), -1)]) # (-A11) B11
    add_subprod((0,0), (0,1), [((1,1), -1)]) # (-A11) B12
    
    # M7: (A12-A22)(B21+B22)
    # C11 += M7
    add_subprod((0,1), (1,0), [((0,0), 1)]) # A12 B21
    add_subprod((0,1), (1,1), [((0,0), 1)]) # A12 B22
    add_subprod((1,1), (1,0), [((0,0), -1)]) # (-A22) B21
    add_subprod((1,1), (1,1), [((0,0), -1)]) # (-A22) B22
    
    # --- Part 2: Decompose the 4x4x2 block (k in 4..5) using schoolbook (rank 32) ---
    # For k=4,5, we can use schoolbook for the 4x4x2 multiplication.
    # This is 4*4*2 = 32 rank-1 terms.
    
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                # A[i,j] * B[j,k] -> C[k,i]
                u = {(i,j): 1}
                v = {(j,k): 1}
                w = {(k,i): 1}
                add_term(u, v, w)
    
    # Convert the list of dicts to matrices
    n_rows_u = n * m
    n_rows_v = m * p
    n_rows_w = p * n
    rank = len(U)
    
    U_mat = [[0] * rank for _ in range(n_rows_u)]
    V_mat = [[0] * rank for _ in range(n_rows_v)]
    W_mat = [[0] * rank for _ in range(n_rows_w)]
    
    for c in range(rank):
        for (i,j), coeff in U[c].items():
            U_mat[i*m+j][c] = coeff
        for (j,k), coeff in V[c].items():
            V_mat[j*p+k][c] = coeff
        for (k,i), coeff in W[c].items():
            W_mat[k*n+i][c] = coeff
            
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
