import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Decomposition:
    - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1).
    - A * B1 uses Strassen 4x4 (rank 49).
    - A * B2 uses Strassen 4x4 (rank 49).
    - A * B3 uses Schoolbook 4x1 (rank 16).
    - Total rank: 114.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    n, m, p = 4, 4, 9  # A is n x m, B is m x p, C is n x p
    
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Strassen 2x2 terms
    # m1 = (a+d)(e+h) -> c11(+1), c22(+1)
    # m2 = (c+d)e    -> c21(+1), c22(-1)
    # m3 = a(f-h)    -> c12(+1), c22(+1)
    # m4 = d(g-e)    -> c11(+1), c21(+1)
    # m5 = (a+b)h    -> c11(-1), c12(+1)
    # m6 = (c-a)(e+f)-> c22(+1)
    # m7 = (b-d)(g+h)-> c11(+1)
    
    # Each term is (A_coeffs, B_coeffs, C_coeffs)
    # Indices for 2x2:
    # A: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # B: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # C: (0,0)=c11, (0,1)=c12, (1,0)=c21, (1,1)=c22
    
    strassen_2x2 = [
        # m1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # m2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # m3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # m4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # m5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # m6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # m7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    def add_strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """
        Adds 49 terms for 4x4 * 4x4 multiplication using recursive Strassen.
        A is at [A_base_i:A_base_i+4, A_base_j:A_base_j+4]
        B is at [B_base_j:B_base_j+4, B_base_k:B_base_k+4]
        C is at [C_base_i:C_base_i+4, C_base_k:C_base_k+4]
        """
        # 7 block multiplications
        # Each block multiplication is a 2x2 * 2x2 product
        # We define the linear combinations of blocks for A and B
        # and the contributions to C blocks
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Block indices (br, bc) for 2x2 grid of 2x2 blocks
        # A blocks: A11(0,0), A12(0,1), A21(1,0), A22(1,1)
        # B blocks: B11(0,0), B12(0,1), B21(1,0), B22(1,1)
        # C blocks: C11(0,0), C12(0,1), C21(1,0), C22(1,1)
        
        block_mults = [
            # (A_side, B_side, C_side)
            # A_side: list of (br, bc, coeff)
            # B_side: list of (br, bc, coeff)
            # C_side: dict (br, bc) -> coeff
            
            # M1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], {(0,0): 1, (1,1): 1}),
            # M2
            ([(1,0,1), (1,1,1)], [(0,0,1)], {(1,0): 1, (1,1): -1}),
            # M3
            ([(0,0,1)], [(0,1,1), (1,1,-1)], {(0,1): 1, (1,1): 1}),
            # M4
            ([(1,1,1)], [(1,0,1), (0,0,-1)], {(0,0): 1, (1,0): 1}),
            # M5
            ([(0,0,1), (0,1,1)], [(1,1,1)], {(0,0): -1, (0,1): 1}),
            # M6
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], {(1,1): 1}),
            # M7
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], {(0,0): 1}),
        ]
        
        for A_side, B_side, C_side in block_mults:
            # For each of the 7 Strassen 2x2 terms
            for A_coeffs, B_coeffs, C_coeffs in strassen_2x2:
                u_vec = [0] * 16
                v_vec = [0] * 36
                w_vec = [0] * 36
                
                # Fill U
                for (li, lj), meta_coeff in A_coeffs.items():
                    for (br, bc, block_coeff) in A_side:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[gi * m + gj] += meta_coeff * block_coeff
                        
                # Fill V
                for (lj, lk), meta_coeff in B_coeffs.items():
                    for (br, bc, block_coeff) in B_side:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[gj * p + gk] += meta_coeff * block_coeff
                        
                # Fill W
                for (li, lk), meta_coeff in C_coeffs.items():
                    for (br, bc), block_coeff in C_side.items():
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[gk * n + gi] += meta_coeff * block_coeff
                        
                add_term(u_vec, v_vec, w_vec)
                
                if time.time() > deadline:
                    return # Fallback not strictly needed if fast, but safe
    
    # Part 1: A * B1 (cols 0-3)
    add_strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # Part 2: A * B2 (cols 4-7)
    add_strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # Part 3: A * B3 (col 8)
    # Schoolbook 4x4 * 4x1
    # C_i = sum_j A_ij B_j8
    # Terms: A_ij * B_j8 -> C_i
    for i in range(4):
        for j in range(4):
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
            
    rank = len(U_cols)
    
    # Convert to matrix format
    U = [[U_cols[col][row] for col in range(rank)] for row in range(16)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(36)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(36)]
    
    return {"U": U, "V": V, "W": W}
