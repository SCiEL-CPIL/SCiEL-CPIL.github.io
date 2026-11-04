import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 49 + 28 = 77
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a 2x2x2 Strassen decomposition at specific offsets
    def add_strassen_222(u_offset, v_offset, w_offset, u_block_map, v_block_map, w_block_map):
        # u_block_map: dict from block name '00', '01', etc to (row_start, col_start) in U
        # Actually, it's easier to pass the indices directly.
        pass
    
    # Helper to add a 2x2x2 Strassen decomposition at specific offsets
    def add_strassen_222(u_offset, v_offset, w_offset, u_block_map, v_block_map, w_block_map):
        # u_block_map: dict from block name '00', '01', etc to (row_start, col_start) in U
        # Actually, it's easier to pass the indices directly.
        pass

    # Let's define the 2x2x2 Strassen steps
    # A is 2x2, B is 2x2, C is 2x2
    # Steps: (alpha, beta, gamma)
    # alpha, beta, gamma are dicts mapping block pos to coeff
    # Block pos: (i, j) where i, j in {0, 1}
    strassen_steps_222 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    # Part 1: 4x4x4 part (cols 0-48)
    # A is 4x4, B1 is 4x4 (cols 0-3 of B), C1 is 4x4 (cols 0-3 of C)
    # Partition A into 2x2 blocks of size 2x2.
    # Partition B1 into 2x2 blocks of size 2x2.
    # Partition C1 into 2x2 blocks of size 2x2.
    
    col_start_1 = 0
    
    # For each of the 7 Strassen steps for the 4x4 level
    for s in range(7):
        alpha_4, beta_4, gamma_4 = strassen_steps_222[s]
        # For each of the 7 Strassen steps for the 2x2 level (inside each block multiplication)
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
            
            col = col_start_1 + s * 7 + s2
            
            # Process U (A)
            for (bi, bj), val_a4 in alpha_4.items():
                # bi, bj are block indices in 4x4 A (0 or 1)
                # The block is 2x2.
                # Inside this block, we have 2x2 sub-blocks (0,0), (0,1), (1,0), (1,1)
                for (si, sj), val_a2 in alpha_2.items():
                    # Global row index in A
                    # Block (bi, bj) starts at row bi*2, col bj*2
                    # Sub-block (si, sj) starts at row si, col sj within the block
                    # So global row = bi*2 + si, global col = bj*2 + sj
                    r = (bi * 2 + si) * m + (bj * 2 + sj)
                    U[r][col] += val_a4 * val_a2
                    
            # Process V (B1)
            for (bi, bj), val_b4 in beta_4.items():
                for (si, sj), val_b2 in beta_2.items():
                    # B1 is 4x4.
                    # Block (bi, bj) starts at row bi*2, col bj*2
                    # Sub-block (si, sj) starts at row si, col sj
                    # Global row = bi*2 + si, global col = bj*2 + sj
                    # Note: V index is row * p + col.
                    # But B1 is only the first 4 columns of B.
                    # So col index in B is bj*2 + sj.
                    r = (bi * 2 + si) * p + (bj * 2 + sj)
                    V[r][col] += val_b4 * val_b2
                    
            # Process W (C1)
            for (bi, bj), val_c4 in gamma_4.items():
                for (si, sj), val_c2 in gamma_2.items():
                    # C1 is 4x4.
                    # Block (bi, bj) starts at row bi*2, col bj*2
                    # Sub-block (si, sj) starts at row si, col sj
                    # Global row = bi*2 + si, global col = bj*2 + sj
                    # W index is col * n + row.
                    r = (bj * 2 + sj) * n + (bi * 2 + si)
                    W[r][col] += val_c4 * val_c2

    # Part 2: 4x4x2 part (cols 49-76)
    # A is 4x4, B2 is 4x2 (cols 4-5 of B), C2 is 4x2 (cols 4-5 of C)
    # Partition A into 2x2 blocks of size 2x2.
    # Partition B2 into 2 blocks of size 2x2 (top and bottom).
    # Partition C2 into 2 blocks of size 2x2 (top and bottom).
    
    col_start_2 = 49
    
    # We need 4 multiplications:
    # M1: A00 * B0
    # M2: A01 * B1
    # M3: A10 * B0
    # M4: A11 * B1
    
    mults = [
        ((0,0), (0,0)), # A00, B0 -> C0
        ((0,1), (1,0)), # A01, B1 -> C0
        ((1,0), (0,0)), # A10, B0 -> C1
        ((1,1), (1,0))  # A11, B1 -> C1
    ]
    
    # Wait, the mapping of B and C blocks:
    # B2 is 4x2.
    # B0 is rows 0-1, cols 0-1 (relative to B2, so global cols 4-5).
    # B1 is rows 2-3, cols 0-1 (relative to B2, so global cols 4-5).
    # C2 is 4x2.
    # C0 is rows 0-1, cols 0-1 (relative to C2, so global cols 4-5).
    # C1 is rows 2-3, cols 0-1 (relative to C2, so global cols 4-5).
    
    # A blocks:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # For each of the 4 multiplications:
    for m_idx, (a_block, b_block) in enumerate(mults):
        # a_block: (bi, bj) in A
        # b_block: (bi, bj) in B2? No, B2 is split vertically.
        # Let's define b_block as (bi, 0) where bi is 0 or 1.
        # And c_block is determined by a_block's bi?
        # C0 comes from A00, A01 (bi=0).
        # C1 comes from A10, A11 (bi=1).
        # So c_block bi is same as a_block bi.
        
        c_block = (a_block[0], 0) # bi same, bj=0 (since C2 is 2x2 blocks? No, C2 is 4x2, split into two 2x2 blocks vertically).
        # Actually, C2 is 4x2.
        # C0 (top) is rows 0-1, cols 0-1.
        # C1 (bottom) is rows 2-3, cols 0-1.
        
        # For each of the 7 Strassen steps inside this 2x2x2 multiplication:
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
            col = col_start_2 + m_idx * 7 + s2
            
            # Process U (A)
            for (si, sj), val_a2 in alpha_2.items():
                # A block is a_block (bi, bj).
                # Sub-block (si, sj).
                # Global row = a_block[0]*2 + si
                # Global col = a_block[1]*2 + sj
                r = (a_block[0] * 2 + si) * m + (a_block[1] * 2 + sj)
                U[r][col] += val_a2
                
            # Process V (B2)
            for (si, sj), val_b2 in beta_2.items():
                # B2 block is b_block (bi, 0).
                # Sub-block (si, sj).
                # Global row = b_block[0]*2 + si
                # Global col = 4 + sj (since B2 starts at col 4)
                r = (b_block[0] * 2 + si) * p + (4 + sj)
                V[r][col] += val_b2
                
            # Process W (C2)
            for (si, sj), val_c2 in gamma_2.items():
                # C2 block is c_block (bi, 0).
                # Sub-block (si, sj).
                # Global row = c_block[0]*2 + si
                # Global col = 4 + sj
                # W index is col * n + row
                r = (4 + sj) * n + (c_block[0] * 2 + si)
                W[r][col] += val_c2

    # Convert to numpy arrays for efficient rank reduction check
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check for redundant columns
    # Flatten tensors for each rank index
    # T_r = U[:, r] (outer) V[:, r] (outer) W[:, r]
    # We want to see if T_r is in the span of {T_k | k != r}
    
    R = rank
    # Precompute flattened tensors
    tensors = []
    for r in range(R):
        t = np.outer(U_np[:, r], V_np[:, r])
        t = np.outer(t, W_np[:, r])
        tensors.append(t.flatten())
    
    tensors_np = np.array(tensors) # Shape (R, N)
    
    # Remove redundant columns
    # Iterate through columns, if one is dependent on others, remove it
    # To be safe, we do this iteratively
    keep_indices = list(range(R))
    changed = True
    while changed:
        changed = False
        for i in range(len(keep_indices)):
            idx = keep_indices[i]
            if len(keep_indices) <= 1:
                break
            # Create matrix of other tensors
            others = [keep_indices[j] for j in range(len(keep_indices)) if j != i]
            if not others:
                continue
            A_mat = tensors_np[others] # Shape (K, N)
            b_vec = tensors_np[idx]
            
            # Check if b_vec is in the column space of A_mat
            # Compute rank of A_mat and rank of [A_mat | b_vec]
            rank_A = np.linalg.matrix_rank(A_mat)
            rank_AB = np.linalg.matrix_rank(np.hstack([A_mat, b_vec.reshape(-1, 1)]))
            
            if rank_A == rank_AB:
                # Redundant
                keep_indices.pop(i)
                changed = True
                break # Restart loop as indices changed

    # Reconstruct U, V, W with kept indices
    if len(keep_indices) < R:
        U_np = U_np[:, keep_indices]
        V_np = V_np[:, keep_indices]
        W_np = W_np[:, keep_indices]
        
        # Convert back to list of lists
        U = U_np.tolist()
        V = V_np.tolist()
        W = W_np.tolist()
        
    return {"U": U, "V": V, "W": W}
