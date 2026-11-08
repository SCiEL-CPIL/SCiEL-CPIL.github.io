"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-65 exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Strategy: Split the 4x5 matrix B into B1 (4x4) and B2 (4x1).
    C[:, 0:4] = A * B1  (4x4 * 4x4, rank 49 via recursive Strassen)
    C[:, 4]   = A * B2  (4x4 * 4x1, rank 16 via schoolbook)
    Total rank = 49 + 16 = 65.
    """
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    # --- Helper: Generate Strassen factors for 2x2 * 2x2 (rank 7) ---
    def strassen_2x2():
        """
        Returns U, V, W for 2x2 * 2x2 -> 2x2.
        U: (2*2, 7) = (4, 7)
        V: (2*2, 7) = (4, 7)
        W: (2*2, 7) = (4, 7)
        Indexing: U[i*m+j], V[j*p+k], W[k*n+i] with n=m=p=2.
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Strassen's 7 products:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        terms = [
            # (U indices, V indices, W coefficients)
            # M1
            ([0, 3], [0, 3], {0: 1, 3: -1}),  # C00: +1, C11: -1
            # M2
            ([2, 3], [0], {2: 1, 3: 1}),       # C10: +1, C11: +1
            # M3
            ([0], [1, 3], {1: 1, 3: 1}),       # C01: +1, C11: +1
            # M4
            ([3], [2, 0], {0: 1, 2: 1}),       # C00: +1, C10: +1
            # M5
            ([0, 1], [3], {0: -1, 3: -1}),     # C00: -1, C11: -1
            # M6
            ([2, 0], [0, 1], {1: 1, 3: 1}),    # C01: +1, C11: +1
            # M7
            ([1, 3], [2, 3], {0: 1, 3: 1}),    # C00: +1, C11: +1
        ]
        
        for r in range(7):
            u_idx, v_idx, w_coeff = terms[r]
            for u in u_idx:
                U[u][r] = 1
            for v in v_idx:
                V[v][r] = 1
            for w_idx, coeff in w_coeff.items():
                W[w_idx][r] = coeff
                
        return U, V, W
    
    # --- Recursive block construction ---
    def strassen_block(n_size, U_2x2, V_2x2, W_2x2):
        """
        Recursively build Strassen factors for n_size x n_size * n_size x n_size.
        Returns U, V, W of shape (n_size^2, rank).
        """
        if n_size == 2:
            return U_2x2, V_2x2, W_2x2
        
        half = n_size // 2
        U_half, V_half, W_half = strassen_block(half, U_2x2, V_2x2, W_2x2)
        rank_half = len(U_half[0])
        
        U = [[0]*(7*rank_half) for _ in range(n_size*n_size)]
        V = [[0]*(7*rank_half) for _ in range(n_size*n_size)]
        W = [[0]*(7*rank_half) for _ in range(n_size*n_size)]
        
        # 7 Strassen terms for the block level
        # Each term combines half-block factors with specific indices
        # Term structure:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # For each of the 7 terms, we combine the half-block factors
        # The U, V, W matrices for the half-block operate on indices within the half-block.
        # We need to map the half-block indices to the full block indices.
        
        # Define the 7 combinations for U and V sides
        # Each entry is a list of (block_position, sign) for U and V
        U_combos = [
            [(0, 1), (3, 1)],   # M1: A00 + A11
            [(2, 1), (3, 1)],   # M2: A10 + A11
            [(0, 1)],           # M3: A00
            [(3, 1)],           # M4: A11
            [(0, 1), (1, 1)],   # M5: A00 + A01
            [(2, 1), (0, -1)],  # M6: A10 - A00
            [(1, 1), (3, -1)],  # M7: A01 - A11
        ]
        V_combos = [
            [(0, 1), (3, 1)],   # M1: B00 + B11
            [(0, 1)],           # M2: B00
            [(1, 1), (3, -1)],  # M3: B01 - B11
            [(2, 1), (0, -1)],  # M4: B10 - B00
            [(3, 1)],           # M5: B11
            [(0, 1), (1, 1)],   # M6: B00 + B01
            [(2, 1), (3, 1)],   # M7: B10 + B11
        ]
        W_combos = [
            [(0, 1), (3, -1)],   # M1: C00 + C11
            [(2, 1), (3, 1)],    # M2: C10 + C11
            [(1, 1), (3, 1)],    # M3: C01 + C11
            [(0, 1), (2, 1)],    # M4: C00 + C10
            [(0, -1), (3, -1)],  # M5: -C00 - C11
            [(1, 1), (3, 1)],    # M6: C01 + C11
            [(0, 1), (3, 1)],    # M7: C00 + C11
        ]
        
        # Helper to get row index in the half-block matrix
        def half_row(block_pos, local_i, local_j, size):
            # block_pos: 0=A00, 1=A01, 2=A10, 3=A11
            # local_i, local_j: indices within the half-block (0 to half-1)
            row_block = block_pos // 2  # 0 or 1
            col_block = block_pos % 2   # 0 or 1
            global_i = row_block * half + local_i
            global_j = col_block * half + local_j
            return global_i * size + global_j
        
        # Build the full U, V, W
        for term in range(7):
            base_col = term * rank_half
            # U side
            for (block_pos, sign) in U_combos[term]:
                for local_i in range(half):
                    for local_j in range(half):
                        full_u_row = half_row(block_pos, local_i, local_j, half)
                        for r in range(rank_half):
                            U[full_u_row][base_col + r] += sign * U_half[local_i*half + local_j][r]
            # V side
            for (block_pos, sign) in V_combos[term]:
                for local_i in range(half):
                    for local_j in range(half):
                        full_v_row = half_row(block_pos, local_i, local_j, half)
                        for r in range(rank_half):
                            V[full_v_row][base_col + r] += sign * V_half[local_i*half + local_j][r]
            # W side
            for (block_pos, sign) in W_combos[term]:
                for local_i in range(half):
                    for local_j in range(half):
                        full_w_row = half_row(block_pos, local_i, local_j, half)
                        for r in range(rank_half):
                            W[full_w_row][base_col + r] += sign * W_half[local_i*half + local_j][r]
        
        return U, V, W
    
    # --- Build 4x4 * 4x4 factors (rank 49) ---
    U_2, V_2, W_2 = strassen_2x2()
    U_44, V_44, W_44 = strassen_block(4, U_2, V_2, W_2)
    rank_44 = len(U_44[0])  # Should be 49
    
    # --- Build 4x4 * 4x1 factors (rank 16) ---
    # Schoolbook for 4x4 * 4x1
    U_441 = [[0]*16 for _ in range(16)]  # 4*4 = 16 rows
    V_441 = [[0]*16 for _ in range(4)]   # 4*1 = 4 rows
    W_441 = [[0]*16 for _ in range(4)]   # 1*4 = 4 rows (output is 4x1, so p=1, n=4, W is 4x4)
    
    col = 0
    for i in range(4):
        for j in range(4):
            U_441[i*4 + j][col] = 1
            V_441[j][col] = 1  # B is 4x1, so index is j*1 + 0 = j
            W_441[i][col] = 1  # C is 4x1, output index is 0*4 + i = i
            col += 1
    
    # --- Combine into final 4x4 * 4x5 factors ---
    # Final shapes:
    # U: (n*m, rank) = (4*4, 49+16) = (16, 65)
    # V: (m*p, rank) = (4*5, 49+16) = (20, 65)
    # W: (p*n, rank) = (5*4, 49+16) = (20, 65)
    
    total_rank = rank_44 + 16
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(20)]
    W = [[0]*total_rank for _ in range(20)]
    
    # First 49 columns: 4x4 * 4x4 part
    # U_44 is (16, 49) -> copy directly to U[:, 0:49]
    for i in range(16):
        for r in range(rank_44):
            U[i][r] = U_44[i][r]
    
    # V_44 is (16, 49) -> maps to V[:, 0:16] (B1 is 4x4, columns 0-3 of B)
    # V index in 4x5: j*5 + k where j=0..3, k=0..3 for B1
    for i in range(16):
        j = i // 4
        k = i % 4
        for r in range(rank_44):
            V[j*5 + k][r] = V_44[i][r]
    
    # W_44 is (16, 49) -> maps to W[:, 0:16] (C1 is 4x4, rows 0-3 of C)
    # W index in 5x4: k*4 + i where k=0..3, i=0..3 for C1
    for i in range(16):
        k = i // 4
        idx = i % 4
        for r in range(rank_44):
            W[k*4 + idx][r] = W_44[i][r]
    
    # Next 16 columns: 4x4 * 4x1 part
    # U_441 is (16, 16) -> U[:, 49:65]
    for i in range(16):
        for r in range(16):
            U[i][rank_44 + r] = U_441[i][r]
    
    # V_441 is (4, 16) -> V[:, 16:20] (B2 is column 4 of B, so k=4)
    for i in range(4):
        for r in range(16):
            V[i*5 + 4][rank_44 + r] = V_441[i][r]
    
    # W_441 is (4, 16) -> W[:, 16:20] (C2 is row 4 of C, so k=4)
    for i in range(4):
        for r in range(16):
            W[4*4 + i][rank_44 + r] = W_441[i][r]
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 140:
        # Should not happen, but safety guard
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
