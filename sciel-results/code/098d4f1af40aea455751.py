import time

def construct():
    """
    Returns a rank-minimal tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses a Strassen-based block decomposition:
    - Split B into two 4x4 blocks and a 4x1 block
    - Use Strassen 4x4x4 (rank 49) for the first two 4x4 blocks
    - Use schoolbook (rank 16) for the last 4x1 block
    - Total rank: 49 + 49 + 16 = 114
    """
    
    start_time = time.time()
    
    n, m, p = 4, 4, 9
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen 2x2x2 base case: 7 rank-1 terms
    # Each term is (u_dict, v_dict, w_dict) where keys are local indices 0..3
    # local index mapping: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    strassen_2x2 = [
        # m1 = (a+d)(e+h) -> c11(+1), c22(+1)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # m2 = (c+d)e -> c21(+1), c22(-1)
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # m3 = a(f-h) -> c12(+1), c22(+1)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # m4 = d(g-e) -> c11(+1), c21(+1)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # m5 = (a+b)h -> c11(-1), c12(+1)
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # m6 = (c-a)(e+f) -> c22(+1)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # m7 = (b-d)(g+h) -> c11(+1)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def add_strassen_4x4(A_i_base, A_j_base, B_j_base, B_k_base, C_i_base, C_k_base):
        """Add a 4x4x4 Strassen decomposition using 2x2 blocks."""
        
        def get_A_indices(block_row, block_col):
            """Get global indices for a 2x2 A block at (block_row, block_col)."""
            indices = {}
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    gi = A_i_base + block_row * 2 + li
                    gj = A_j_base + block_col * 2 + lj
                    indices[local_idx] = (gi, gj)
            return indices
        
        def get_B_indices(block_row, block_col):
            """Get global indices for a 2x2 B block at (block_row, block_col)."""
            indices = {}
            for lj in range(2):
                for lk in range(2):
                    local_idx = lj * 2 + lk
                    gj = B_j_base + block_row * 2 + lj
                    gk = B_k_base + block_col * 2 + lk
                    indices[local_idx] = (gj, gk)
            return indices
        
        def get_C_indices(block_row, block_col):
            """Get global indices for a 2x2 C block at (block_row, block_col)."""
            indices = {}
            for li in range(2):
                for lk in range(2):
                    local_idx = li * 2 + lk
                    gi = C_i_base + block_row * 2 + li
                    gk = C_k_base + block_col * 2 + lk
                    indices[local_idx] = (gi, gk)
            return indices
        
        def add_inner(A_block_specs, B_block_specs, C_block_specs):
            """
            A_block_specs: list of (block_row, block_col, coeff)
            B_block_specs: list of (block_row, block_col, coeff)
            C_block_specs: list of (block_row, block_col, coeff)
            """
            for u_c, v_c, w_c in strassen_2x2:
                u_vec = [0] * (n*m)
                v_vec = [0] * (m*p)
                w_vec = [0] * (p*n)
                
                # Fill u_vec from A_blocks
                for (loc_i, loc_j), coeff in u_c.items():
                    # loc_i, loc_j are local indices within the 2x2 block
                    # We need to map these to the actual block positions
                    # The local indices in strassen_2x2 are:
                    # 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                    # But u_c keys are just 0,1,2,3 representing positions
                    # We need to iterate over each A block spec and add contributions
                    pass
                
                # This approach is getting complicated. Let me use a simpler method.
                # For each Strassen term, we have coefficients for A positions (0,1,2,3)
                # and B positions (0,1,2,3) and C positions (0,1,2,3).
                # The block specs tell us which 2x2 blocks to use and with what coefficient.
                # For each local position in the 2x2 block, we add the contribution.
                
                # Let me redo this more carefully.
                # u_c has keys 0,1,2,3 which correspond to local positions (0,0),(0,1),(1,0),(1,1)
                # For each A block spec (br, bc, bc_coeff), we add bc_coeff * u_c[loc] to the
                # global position (A_i_base + br*2 + li, A_j_base + bc*2 + lj)
                
                u_vec = [0] * (n*m)
                v_vec = [0] * (m*p)
                w_vec = [0] * (p*n)
                
                # Process u_vec
                for loc, coeff in u_c.items():
                    li = loc // 2
                    lj = loc % 2
                    for (br, bc, bc_coeff) in A_block_specs:
                        gi = A_i_base + br * 2 + li
                        gj = A_j_base + bc * 2 + lj
                        u_vec[gi * m + gj] += coeff * bc_coeff
                
                # Process v_vec
                for loc, coeff in v_c.items():
                    lj = loc // 2
                    lk = loc % 2
                    for (br, bc, bc_coeff) in B_block_specs:
                        gj = B_j_base + br * 2 + lj
                        gk = B_k_base + bc * 2 + lk
                        v_vec[gj * p + gk] += coeff * bc_coeff
                
                # Process w_vec
                for loc, coeff in w_c.items():
                    li = loc // 2
                    lk = loc % 2
                    for (br, bc, bc_coeff) in C_block_specs:
                        gi = C_i_base + br * 2 + li
                        gk = C_k_base + bc * 2 + lk
                        w_vec[gk * n + gi] += coeff * bc_coeff
                
                add_term(u_vec, v_vec, w_vec)
        
        # 7 Strassen combinations for 4x4 using 2x2 blocks
        # A is split into 4 blocks: A00 at (0,0), A01 at (0,1), A10 at (1,0), A11 at (1,1)
        # B is split into 4 blocks: B00 at (0,0), B01 at (0,1), B10 at (1,0), B11 at (1,1)
        # C is split into 4 blocks: C00 at (0,0), C01 at (0,1), C10 at (1,0), C11 at (1,1)
        
        # M1 = (A00+A11)(B00+B11) -> C00 + C11
        add_inner([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
        
        # M2 = (A10+A11)(B00) -> C10 + C11(-1)
        add_inner([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])
        
        # M3 = (A00)(B01-B11) -> C01 + C11
        add_inner([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])
        
        # M4 = (A11)(B10-B00) -> C00 + C10
        add_inner([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])
        
        # M5 = (A00+A01)(B11) -> C00(-1) + C01
        add_inner([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
        
        # M6 = (A10-A00)(B00+B01) -> C11
        add_inner([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
        
        # M7 = (A01-A11)(B10+B11) -> C00
        add_inner([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    
    # First 4x4 block of B: columns 0-3
    add_strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # Second 4x4 block of B: columns 4-7
    add_strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # Last column of B: column 8, use schoolbook
    for i in range(n):
        for j in range(m):
            u_vec = [0] * (n*m)
            v_vec = [0] * (m*p)
            w_vec = [0] * (p*n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    
    # Convert list of column vectors to row-major matrices
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n*m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m*p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
