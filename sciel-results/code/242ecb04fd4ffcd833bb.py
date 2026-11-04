"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build U, V, W as lists of lists.
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U = []
    V = []
    W = []
    
    def add_term(u_vals, v_vals, w_vals):
        """
        Adds a term to the decomposition.
        u_vals: dict mapping (i, j) to value for U[i*m + j, col]
        v_vals: dict mapping (j, k) to value for V[j*p + k, col]
        w_vals: dict mapping (k, i) to value for W[k*n + i, col]
        """
        col = len(U)
        # Initialize new column
        U.append([0] * (n * m))
        V.append([0] * (m * p))
        W.append([0] * (p * n))
        
        for (i, j), val in u_vals.items():
            U[col][i * m + j] = val
        for (j, k), val in v_vals.items():
            V[col][j * p + k] = val
        for (k, i), val in w_vals.items():
            W[col][k * n + i] = val

    def schoolbook_2x2(u_offset_i, u_offset_j, v_offset_j, v_offset_k, w_offset_k, w_offset_i, n_local, m_local, p_local, U_local, V_local, W_local):
        """
        Adds schoolbook terms for a n_local x m_local x p_local multiplication.
        Indices are offset by u_offset_i, etc.
        """
        for i in range(n_local):
            for j in range(m_local):
                for k in range(p_local):
                    u_vals = {(i, j): 1}
                    v_vals = {(j, k): 1}
                    w_vals = {(k, i): 1}
                    # Map local indices to global
                    gu_vals = {(u_offset_i + i, u_offset_j + j): 1}
                    gv_vals = {(v_offset_j + j, v_offset_k + k): 1}
                    gw_vals = {(w_offset_k + k, w_offset_i + i): 1}
                    add_term(gu_vals, gv_vals, gw_vals)

    def strassen_2x2(u_off_i, u_off_j, v_off_j, v_off_k, w_off_k, w_off_i):
        """
        Adds Strassen terms for a 4x4x4 multiplication (2x2 blocks of 2x2).
        u_off_i, u_off_j: top-left corner of A block in global A
        v_off_j, v_off_k: top-left corner of B block in global B
        w_off_k, w_off_i: top-left corner of C block in global C
        """
        # A is 4x4, split into 2x2 blocks A11, A12, A21, A22
        # B is 4x4, split into 2x2 blocks B11, B12, B21, B22
        # C is 4x4, split into 2x2 blocks C11, C12, C21, C22
        
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # We need to expand each Mi into schoolbook 2x2x2 terms.
        # Each Mi is a product of two 2x2 matrices.
        # Let's create a helper to add a product of two 2x2 matrices defined by linear combinations of blocks.
        
        def add_product_2x2(A_coeffs, B_coeffs, sign=1):
            """
            A_coeffs: list of (block, coeff) for A (2x2 blocks)
            B_coeffs: list of (block, coeff) for B (2x2 blocks)
            block is (bi, bj) where bi, bj in {0, 1}
            """
            # Expand (sum ca * Ablock) * (sum cb * Bblock)
            # = sum ca cb (Ablock * Bblock)
            for (ai, aj), ca in A_coeffs:
                for (bi, bj), cb in B_coeffs:
                    coeff = sign * ca * cb
                    if coeff == 0: continue
                    
                    # A block (ai, aj) is at rows u_off_i + 2*ai, cols u_off_j + 2*aj
                    # B block (bi, bj) is at rows v_off_j + 2*bi, cols v_off_k + 2*bj
                    # C block (ai, bj) is at rows w_off_k + 2*bj, cols w_off_i + 2*ai
                    
                    a_i = u_off_i + 2 * ai
                    a_j = u_off_j + 2 * aj
                    b_j = v_off_j + 2 * bi
                    b_k = v_off_k + 2 * bj
                    c_k = w_off_k + 2 * bj
                    c_i = w_off_i + 2 * ai
                    
                    # Schoolbook for 2x2x2
                    for li in range(2):
                        for lj in range(2):
                            for lk in range(2):
                                gu = {(a_i + li, a_j + lj): coeff}
                                gv = {(b_j + lj, b_k + lk): 1} # Wait, V is multiplied by coeff?
                                # Actually, we can put coeff in U or V or W.
                                # Let's put it in U.
                                gw = {(c_k + lk, c_i + li): 1}
                                add_term(gu, gv, gw)

        # M1
        add_product_2x2([(0,0, 1), (1,1, 1)], [(0,0, 1), (1,1, 1)])
        # M2
        add_product_2x2([(1,0, 1), (1,1, 1)], [(0,0, 1)])
        # M3
        add_product_2x2([(0,0, 1)], [(0,1, 1), (1,1, -1)])
        # M4
        add_product_2x2([(1,1, 1)], [(1,0, 1), (0,0, -1)])
        # M5
        add_product_2x2([(0,0, 1), (0,1, 1)], [(1,1, 1)])
        # M6
        add_product_2x2([(1,0, 1), (0,0, -1)], [(0,0, 1), (0,1, 1)])
        # M7
        add_product_2x2([(0,1, 1), (1,1, -1)], [(1,0, 1), (1,1, 1)])

    # Term 1: A1 (4x4) * B11 (4x4) -> C11 (4x4)
    # A1 is A[0:4, 0:4]
    # B11 is B[0:4, 0:4]
    # C11 is C[0:4, 0:4]
    strassen_2x2(0, 0, 0, 0, 0, 0)

    # Term 2: A1 (4x4) * B12 (4x1) -> C12 (4x1)
    # A1 is A[0:4, 0:4]
    # B12 is B[0:4, 4]
    # C12 is C[0:4, 4]
    # Schoolbook 4x4x1
    schoolbook_2x2(0, 0, 0, 4, 4, 0, 4, 4, 1, None, None, None)
    # Wait, schoolbook_2x2 helper was not fully implemented for offsets in the way I called it.
    # Let's just implement it inline or fix the helper.
    # The helper `schoolbook_2x2` above is a bit messy. Let's just write the loop.
    
    # Clear the previous call if it was wrong? No, I didn't call it yet in the final code structure.
    # I will rewrite the term additions clearly.

    # Reset U, V, W to be safe
    U = []
    V = []
    W = []

    def add_term_simple(u_map, v_map, w_map):
        col = len(U)
        U.append([0] * (n * m))
        V.append([0] * (m * p))
        W.append([0] * (p * n))
        for idx, val in u_map.items():
            U[col][idx] = val
        for idx, val in v_map.items():
            V[col][idx] = val
        for idx, val in w_map.items():
            W[col][idx] = val

    # Term 1: Strassen 4x4x4
    # Indices: A[0:4, 0:4], B[0:4, 0:4], C[0:4, 0:4]
    # We use the strassen_2x2 logic but with flat indices.
    
    def add_strassen_4x4x4(a_i0, a_j0, b_j0, b_k0, c_k0, c_i0):
        # A is 4x4 at a_i0, a_j0
        # B is 4x4 at b_j0, b_k0
        # C is 4x4 at c_k0, c_i0
        
        # Blocks of A:
        # A11: a_i0..a_i0+1, a_j0..a_j0+1
        # A12: a_i0..a_i0+1, a_j0+2..a_j0+3
        # A21: a_i0+2..a_i0+3, a_j0..a_j0+1
        # A22: a_i0+2..a_i0+3, a_j0+2..a_j0+3
        
        # Same for B, C.
        
        def add_prod(A_blocks, B_blocks, sign=1):
            for (ai, aj), ca in A_blocks:
                for (bi, bj), cb in B_blocks:
                    coeff = sign * ca * cb
                    if coeff == 0: continue
                    a_i = a_i0 + 2 * ai
                    a_j = a_j0 + 2 * aj
                    b_j = b_j0 + 2 * bi
                    b_k = b_k0 + 2 * bj
                    c_k = c_k0 + 2 * bj
                    c_i = c_i0 + 2 * ai
                    for li in range(2):
                        for lj in range(2):
                            for lk in range(2):
                                u_idx = (a_i + li) * m + (a_j + lj)
                                v_idx = (b_j + lj) * p + (b_k + lk)
                                w_idx = (c_k + lk) * n + (c_i + li)
                                add_term_simple({u_idx: coeff}, {v_idx: 1}, {w_idx: 1})

        # M1
        add_prod([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
        # M2
        add_prod([(1,0,1), (1,1,1)], [(0,0,1)])
        # M3
        add_prod([(0,0,1)], [(0,1,1), (1,1,-1)])
        # M4
        add_prod([(1,1,1)], [(1,0,1), (0,0,-1)])
        # M5
        add_prod([(0,0,1), (0,1,1)], [(1,1,1)])
        # M6
        add_prod([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)])
        # M7
        add_prod([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)])

    add_strassen_4x4x4(0, 0, 0, 0, 0, 0)

    # Term 2: A1 (4x4) * B12 (4x1) -> C12 (4x1)
    # A[0:4, 0:4], B[0:4, 4], C[0:4, 4]
    for i in range(4):
        for j in range(4):
            u_idx = i * m + j
            v_idx = j * p + 4
            w_idx = 4 * n + i
            add_term_simple({u_idx: 1}, {v_idx: 1}, {w_idx: 1})

    # Term 3: A2 (4x1) * B21 (1x4) -> C21 (4x4)
    # A[0:4, 4], B[4, 0:4], C[0:4, 0:4]
    # Rank 1
    u_map = {}
    v_map = {}
    w_map = {}
    for i in range(4):
        u_map[i * m + 4] = 1
    for k in range(4):
        v_map[4 * p + k] = 1
    for i in range(4):
        for k in range(4):
            w_map[k * n + i] = 1
    add_term_simple(u_map, v_map, w_map)

    # Term 4: A2 (4x1) * B22 (1x1) -> C22 (4x1)
    # A[0:4, 4], B[4, 4], C[0:4, 4]
    for i in range(4):
        u_idx = i * m + 4
        v_idx = 4 * p + 4
        w_idx = 4 * n + i
        add_term_simple({u_idx: 1}, {v_idx: 1}, {w_idx: 1})

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
