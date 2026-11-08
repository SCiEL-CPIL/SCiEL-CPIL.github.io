import time

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen-based block decomposition with rank 114:
    - Two 4x4*4x4 Strassen blocks: 49 + 49 = 98
    - One 4x4*4x1 schoolbook: 16
    - Total: 114
    """
    n, m, p = 4, 4, 9  # A is n x m, B is m x p, C is n x p
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen 2x2 * 2x2 terms:
    # Each term: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict from (i,j) to coeff for A[i][j]
    # B_coeffs: dict from (j,k) to coeff for B[j][k]
    # C_coeffs: dict from (i,k) to coeff for C[i][k]
    
    S = [
        # m1 = (a+d)(e+h), c11 += 1, c22 += 1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # m2 = (c+d)*e, c21 += 1, c22 -= 1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # m3 = a*(f-h), c12 += 1, c22 += 1
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # m4 = d*(g-e), c11 += 1, c21 += 1
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # m5 = (a+b)*h, c11 -= 1, c12 += 1
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # m6 = (c-a)*(e+f), c22 += 1
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # m7 = (b-d)*(g+h), c11 += 1
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    def add_strassen_2x2(A_i0, A_j0, B_j0, B_k0, C_i0, C_k0, A_scale=1, B_scale=1, C_scale=1):
        """Add 7 Strassen rank-1 terms for a 2x2*2x2 block."""
        for A_c, B_c, C_c in S:
            u = [0] * (n*m)
            v = [0] * (m*p)
            w = [0] * (p*n)
            
            for (li, lj), coeff in A_c.items():
                gi = A_i0 + li
                gj = A_j0 + lj
                u[gi*m + gj] = coeff * A_scale
            
            for (lj, lk), coeff in B_c.items():
                gj = B_j0 + lj
                gk = B_k0 + lk
                v[gj*p + gk] = coeff * B_scale
            
            for (li, lk), coeff in C_c.items():
                gi = C_i0 + li
                gk = C_k0 + lk
                w[gk*n + gi] = coeff * C_scale
            
            add_term(u, v, w)
    
    def strassen_4x4(A_i0, A_j0, B_j0, B_k0, C_i0, C_k0):
        """Add 49 Strassen rank-1 terms for a 4x4*4x4 block."""
        # Block structure:
        # A = [[A11, A12], [A21, A22]] each 2x2
        # B = [[B11, B12], [B21, B22]] each 2x2
        # C = [[C11, C12], [C21, C22]] each 2x2
        #
        # Block positions (offsets from A_i0, A_j0):
        # A11: (0,0), A12: (0,2), A21: (2,0), A22: (2,2)
        # B11: (0,0), B12: (0,2), B21: (2,0), B22: (2,2)
        # C11: (0,0), C12: (0,2), C21: (2,0), C22: (2,2)
        
        # The 7 Strassen multiplications at block level:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Each Mi is a product of 2x2 matrices. We decompose each with Strassen.
        # But the inputs to each Strassen are sums/differences of A-blocks and B-blocks.
        
        # For each Strassen term in the inner 2x2 multiplication, we need to expand
        # the "sum of blocks" into individual entries.
        
        # Define helper: given a list of (block_row, block_col, coeff) for A-side,
        # and a Strassen A-coeff dict, expand into global u_vec entries.
        
        def expand_A(A_side, local_c):
            """Expand A-side contributions into u_vec."""
            u = [0] * (n*m)
            for (li, lj), meta_coeff in local_c.items():
                for (br, bc, block_coeff) in A_side:
                    gi = A_i0 + br*2 + li
                    gj = A_j0 + bc*2 + lj
                    u[gi*m + gj] += meta_coeff * block_coeff
            return u
        
        def expand_B(B_side, local_c):
            """Expand B-side contributions into v_vec."""
            v = [0] * (m*p)
            for (lj, lk), meta_coeff in local_c.items():
                for (br, bc, block_coeff) in B_side:
                    gj = B_j0 + br*2 + lj
                    gk = B_k0 + bc*2 + lk
                    v[gj*p + gk] += meta_coeff * block_coeff
            return v
        
        def expand_C(C_side, local_c):
            """Expand C-side contributions into w_vec."""
            w = [0] * (p*n)
            for (li, lk), meta_coeff in local_c.items():
                for (br, bc, block_coeff) in C_side:
                    gi = C_i0 + br*2 + li
                    gk = C_k0 + bc*2 + lk
                    w[gk*n + gi] += meta_coeff * block_coeff
            return w
        
        # Define the 7 block-level multiplications with their C-side contributions:
        # Each entry: (A_side, B_side, C_side_dict)
        # A_side: list of (block_row, block_col, coeff)
        # B_side: list of (block_row, block_col, coeff)
        # C_side: dict from (block_row, block_col) to coeff
        
        block_mults = [
            # M1 = (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], {(0,0):1, (1,1):1}),
            # M2 = (A21+A22)B11 -> C21 += 1, C22 -= 1
            ([(1,0,1), (1,1,1)], [(0,0,1)], {(1,0):1, (1,1):-1}),
            # M3 = A11(B12-B22) -> C12 += 1, C22 += 1
            ([(0,0,1)], [(0,1,1), (1,1,-1)], {(0,1):1, (1,1):1}),
            # M4 = A22(B21-B11) -> C11 += 1, C21 += 1
            ([(1,1,1)], [(1,0,1), (0,0,-1)], {(0,0):1, (1,0):1}),
            # M5 = (A11+A12)B22 -> C11 -= 1, C12 += 1
            ([(0,0,1), (0,1,1)], [(1,1,1)], {(0,0):-1, (0,1):1}),
            # M6 = (A21-A11)(B11+B12) -> C22 += 1
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], {(1,1):1}),
            # M7 = (A12-A22)(B21+B22) -> C11 += 1
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], {(0,0):1}),
        ]
        
        for A_side, B_side, C_side in block_mults:
            for A_c, B_c, C_c in S:
                u = expand_A(A_side, A_c)
                v = expand_B(B_side, B_c)
                w = expand_C(C_side, C_c)
                add_term(u, v, w)
    
    # Part 1: A * B[:, 0:4] (4x4 * 4x4)
    strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # Part 2: A * B[:, 4:8] (4x4 * 4x4)
    strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # Part 3: A * B[:, 8:9] (4x4 * 4x1) - schoolbook, 16 terms
    for i in range(n):
        for j in range(m):
            u = [0] * (n*m)
            v = [0] * (m*p)
            w = [0] * (p*n)
            u[i*m + j] = 1
            v[j*p + 8] = 1
            w[8*n + i] = 1
            add_term(u, v, w)
    
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n*m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m*p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
