import time

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Returns a rank-minimal tensor decomposition for multiplying a 4x4 matrix by a 4x9 matrix.
    
    Strategy:
    1. Split the 4x9 matrix B into three parts: B1 (4x4), B2 (4x4), B3 (4x1).
    2. Compute C1 = A * B1 using a rank-49 Strassen-based algorithm.
    3. Compute C2 = A * B2 using a rank-49 Strassen-based algorithm.
    4. Compute C3 = A * B3 using a rank-16 schoolbook algorithm.
    5. Total rank = 49 + 49 + 16 = 114.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Dimensions for vectors
    dim_u = n * m  # 16
    dim_v = m * p  # 36
    dim_w = p * n  # 36

    # Pre-compute the 49 terms for 4x4 * 4x4 matrix multiplication
    # Each term is a tuple (u_coords, v_coords, w_coords)
    # u_coords: list of (index, coeff) for A (4x4)
    # v_coords: list of (index, coeff) for B (4x4)
    # w_coords: list of (index, coeff) for C (4x4)
    
    # We use the standard Strassen recursion for 4x4:
    # Split 4x4 into 2x2 blocks.
    # There are 7 Strassen multiplications of 2x2 blocks.
    # Each 2x2 multiplication has 7 terms.
    # Total 7 * 7 = 49 terms.
    
    # Indices for 2x2 blocks:
    # A: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
    # C: C00 (0,0), C01 (0,1), C10 (1,0), C11 (1,1)
    
    # Strassen 2x2 terms (u, v, w) where indices are 0,1,2,3 for (0,0), (0,1), (1,0), (1,1)
    strassen_2x2 = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Block-level Strassen structure for 4x4
    # A_side: list of (block_idx, coeff)
    # B_side: list of (block_idx, coeff)
    # C_side: list of (block_idx, coeff)
    # Block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    block_strassen_4x4 = [
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (A10+A11)B00 -> C10, -C11
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = A00(B01-B11) -> C01, C11
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = A11(B10-B00) -> C00, C10
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (A00+A01)B11 -> -C00, C01
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6 = (A10-A00)(B00+B01) -> C11
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (A01-A11)(B10+B11) -> C00
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    def map_block_index_to_global(block_idx, offset):
        """
        Maps a 2x2 block index (0,1,2,3) and its top-left offset (row, col) 
        to a list of global indices for the 4x4 matrix.
        Returns list of (global_index, coeff_placeholder) is not quite right, 
        we need to map the 4 elements of the 2x2 block to global 4x4 indices.
        """
        br = block_idx // 2
        bc = block_idx % 2
        base_r = br * 2
        base_c = bc * 2
        # Global indices for the 4 elements of this 2x2 block
        # 0: (base_r, base_c)
        # 1: (base_r, base_c+1)
        # 2: (base_r+1, base_c)
        # 3: (base_r+1, base_c+1)
        return [
            (base_r * 4 + base_c, 0),
            (base_r * 4 + base_c + 1, 1),
            ((base_r + 1) * 4 + base_c, 2),
            ((base_r + 1) * 4 + base_c + 1, 3)
        ]

    # Generate the 49 terms for 4x4 * 4x4
    terms_4x4 = []
    for A_side, B_side, C_side in block_strassen_4x4:
        for u_meta, v_meta, w_meta in strassen_2x2:
            u_coords = {}
            v_coords = {}
            w_coords = {}
            
            # Process U (A side)
            for block_idx, coeff in A_side:
                # Get mapping for this block
                block_map = map_block_index_to_global(block_idx, 0)
                for local_idx, meta_coeff in u_meta.items():
                    # local_idx is 0,1,2,3 corresponding to the 4 elements in the 2x2 block
                    # We need to map local_idx to the correct global index
                    # block_map gives us the global index for each of the 4 local positions
                    global_idx, local_pos = block_map[local_idx]
                    # Wait, block_map returns (global_idx, local_pos). 
                    # My map_block_index_to_global returns a list of tuples (global_idx, local_pos_in_2x2_block)?
                    # No, let's refine map_block_index_to_global to return a dict {local_pos: global_idx}
                    pass
            
            # Let's redo the mapping logic cleanly
            # For a 4x4 matrix, index is r*4+c.
            # Block (br, bc) contains elements at (br*2 + dr, bc*2 + dc) for dr,dc in {0,1}.
            # Local index l in 2x2 block corresponds to dr=l//2, dc=l%2.
            
            for block_idx, coeff in A_side:
                br = block_idx // 2
                bc = block_idx % 2
                for local_idx, meta_coeff in u_meta.items():
                    dr = local_idx // 2
                    dc = local_idx % 2
                    gr = br * 2 + dr
                    gc = bc * 2 + dc
                    global_idx = gr * 4 + gc
                    u_coords[global_idx] = u_coords.get(global_idx, 0) + coeff * meta_coeff

            for block_idx, coeff in B_side:
                br = block_idx // 2
                bc = block_idx % 2
                for local_idx, meta_coeff in v_meta.items():
                    dr = local_idx // 2
                    dc = local_idx % 2
                    gr = br * 2 + dr
                    gc = bc * 2 + dc
                    global_idx = gr * 4 + gc
                    v_coords[global_idx] = v_coords.get(global_idx, 0) + coeff * meta_coeff

            for block_idx, coeff in C_side:
                br = block_idx // 2
                bc = block_idx % 2
                for local_idx, meta_coeff in w_meta.items():
                    dr = local_idx // 2
                    dc = local_idx % 2
                    gr = br * 2 + dr
                    gc = bc * 2 + dc
                    global_idx = gr * 4 + gc
                    w_coords[global_idx] = w_coords.get(global_idx, 0) + coeff * meta_coeff

            terms_4x4.append((u_coords, v_coords, w_coords))
            
            if time.time() > deadline:
                return {"U": [], "V": [], "W": []}

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_dict, v_dict, w_dict, col_offset_u=0, col_offset_v=0, col_offset_w=0):
        u_vec = [0] * dim_u
        v_vec = [0] * dim_v
        w_vec = [0] * dim_w
        
        for idx, val in u_dict.items():
            u_vec[idx + col_offset_u] = val
        for idx, val in v_dict.items():
            v_vec[idx + col_offset_v] = val
        for idx, val in w_dict.items():
            w_vec[idx + col_offset_w] = val
            
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Part 1: A * B1 (cols 0-3 of B, rows 0-3 of C)
    # B1 is columns 0-3 of B. In V (size 36, 4x9), indices 0-15 correspond to B1.
    # C1 is rows 0-3 of C. In W (size 36, 9x4), indices 0-15 correspond to C1.
    # Note: W index is k*n + i. For C1 (i=0..3, k=0..3), indices are 0..15.
    for u_d, v_d, w_d in terms_4x4:
        # v_d keys are 0..15 (4x4). These map directly to B1 in V (cols 0-3).
        # w_d keys are 0..15 (4x4). These map directly to C1 in W (rows 0-3).
        # u_d keys are 0..15 (4x4). These map directly to A in U.
        add_term(u_d, v_d, w_d)
        
    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}

    # Part 2: A * B2 (cols 4-7 of B, rows 4-7 of C)
    # B2 is columns 4-7 of B. In V, indices 16-31 correspond to B2.
    # C2 is rows 4-7 of C. In W, indices 16-31 correspond to C2.
    for u_d, v_d, w_d in terms_4x4:
        # Shift v_d indices by 16 (4 * 4 = 16)
        v_shifted = {k + 16: val for k, val in v_d.items()}
        # Shift w_d indices by 16
        w_shifted = {k + 16: val for k, val in w_d.items()}
        add_term(u_d, v_shifted, w_shifted)

    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}

    # Part 3: A * B3 (col 8 of B, row 8 of C)
    # Schoolbook for 4x4 * 4x1
    for i in range(4):
        for j in range(4):
            u_d = {i * 4 + j: 1}
            # B3 is column 8. In V (4x9), index is j*9 + 8.
            v_d = {j * 9 + 8: 1}
            # C3 is row 8. In W (9x4), index is 8*4 + i = 32 + i.
            w_d = {32 + i: 1}
            add_term(u_d, v_d, w_d)

    # Build final matrices
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(dim_u)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(dim_v)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(dim_w)]
    
    return {"U": U, "V": V, "W": W}
