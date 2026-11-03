"""Strassen-based decomposition for 4x4 * 4x8 matrix multiplication."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    # We decompose into two independent 4x4*4x4 blocks (columns 0-3 and 4-7 of B)
    # Each 4x4*4x4 uses Strassen recursion: rank 49 per block, total rank 98.
    
    # Strassen for 2x2: 7 terms
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B21-B22)
    # M4 = A22*(B12-B21)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x4*4x4, we use 2x2 blocks of size 2x2, and apply Strassen to each 2x2 sub-block.
    # The 4x4 Strassen has 7 terms, each involving sums/diffs of 2x2 blocks.
    # Each 2x2 block multiplication can be done with 7 scalar Strassen terms.
    # So total rank for 4x4*4x4 is 7*7 = 49.
    
    # Let's define the 4x4 Strassen structure:
    # Divide A (4x4) into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # Divide B (4x4) into 2x2 blocks: B11, B12, B21, B22 (each 2x2)
    # Divide C (4x4) into 2x2 blocks: C11, C12, C21, C22 (each 2x2)
    
    # 4x4 Strassen terms (each is a 2x2*2x2 product):
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B21-B22)
    # M4 = A22*(B12-B21)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # Each of these 7 terms requires a 2x2*2x2 multiplication, which uses 7 scalar terms.
    # Total: 49 scalar terms for 4x4*4x4.
    
    # For 4x4*4x8, we have two such blocks (columns 0-3 and 4-7), total rank 98.
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * rank for _ in range(p * n)]  # 32 x 98
    
    # Define 2x2 Strassen terms for scalar level
    # Each term is (U_entries, V_entries, W_entries)
    # where entries are (index, coeff) pairs
    # U index: i*2+j for 2x2 A
    # V index: j*2+k for 2x2 B
    # W index: k*2+i for 2x2 C
    
    strassen_2x2 = [
        # M1 = (A11+A22)*(B11+B22)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (A21+A22)*B11
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, 1)]),
        # M3 = A11*(B21-B22)
        ([(0, 1)], [(2, 1), (3, -1)], [(0, 1), (1, -1)]),
        # M4 = A22*(B12-B21)
        ([(3, 1)], [(1, 1), (2, -1)], [(1, 1), (2, -1)]),
        # M5 = (A11+A12)*B22
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 1), (1, 1)]),
        # M6 = (A21-A11)*(B11+B12)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(2, 1), (3, 1)]),
        # M7 = (A12-A22)*(B21+B22)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (1, 1)]),
    ]
    
    # 4x4 Strassen structure: 7 terms, each specifies which 2x2 blocks of A and B to use
    # and with what signs
    # Each term: (A_block_specs, B_block_specs)
    # A_block_specs: list of (block_idx, coeff) where block_idx: 0=A11, 1=A12, 2=A21, 3=A22
    # B_block_specs: list of (block_idx, coeff)
    
    # Block index to (row_range, col_range) in 4x4:
    # 0: A11 -> rows 0-1, cols 0-1
    # 1: A12 -> rows 0-1, cols 2-3
    # 2: A21 -> rows 2-3, cols 0-1
    # 3: A22 -> rows 2-3, cols 2-3
    
    def block_to_indices(bidx):
        """Return (row_start, col_start) for block index in 4x4 matrix."""
        r = (bidx // 2) * 2
        c = (bidx % 2) * 2
        return (r, c)
    
    # 4x4 Strassen terms
    strassen_4x4 = [
        # M1 = (A11+A22)*(B11+B22)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (A21+A22)*B11
        ([(2, 1), (3, 1)], [(0, 1)]),
        # M3 = A11*(B21-B22)
        ([(0, 1)], [(2, 1), (3, -1)]),
        # M4 = A22*(B12-B21)
        ([(3, 1)], [(1, 1), (2, -1)]),
        # M5 = (A11+A12)*B22
        ([(0, 1), (1, 1)], [(3, 1)]),
        # M6 = (A21-A11)*(B11+B12)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        # M7 = (A12-A22)*(B21+B22)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),
    ]
    
    col = 0
    # For each half of the 8 columns (0-3 and 4-7)
    for half in range(2):
        k_offset = half * 4  # 0 or 4
        
        # For each 4x4 Strassen term
        for a_specs, b_specs in strassen_4x4:
            # For each 2x2 Strassen term within
            for (u_entries_2x2, v_entries_2x2, w_entries_2x2) in strassen_2x2:
                # Build the full U, V, W for this scalar term
                # A is 4x4, so U has indices i*4+j where i,j in 0..3
                # B is 4x8, so V has indices j*8+k where j in 0..3, k in 0..7
                # C is 8x4, so W has indices k*4+i where k in 0..7, i in 0..3
                
                # A contributions: for each (block_idx, coeff_a) in a_specs
                # and each (local_i, local_j, coeff_inner) in u_entries_2x2
                # The global i = block_row + local_i, global j = block_col + local_j
                for (a_bidx, a_coeff) in a_specs:
                    a_row, a_col = block_to_indices(a_bidx)
                    for (li, lj, inner_coeff) in u_entries_2x2:
                        gi = a_row + li
                        gj = a_col + lj
                        U[gi * m + gj][col] += a_coeff * inner_coeff
                
                # B contributions: for each (block_idx, coeff_b) in b_specs
                # and each (local_j, local_k, coeff_inner) in v_entries_2x2
                # The global j = block_row + local_j, global k = k_offset + block_col + local_k
                for (b_bidx, b_coeff) in b_specs:
                    b_row, b_col = block_to_indices(b_bidx)
                    for (lj, lk, inner_coeff) in v_entries_2x2:
                        gj = b_row + lj
                        gk = k_offset + b_col + lk
                        V[gj * p + gk][col] += b_coeff * inner_coeff
                
                # W contributions: the output C block structure
                # For each 4x4 Strassen term, the output C block is determined
                # We need to figure out which C blocks get contributions
                
                # Actually, the 4x4 Strassen gives us C11, C12, C21, C22 as combinations of M1-M7
                # But here we're building individual rank-1 terms, not the final combination
                # Each scalar term from the 2x2 Strassen within a 4x4 Strassen term contributes
                # to specific C blocks based on the 4x4 Strassen structure
                
                # Let me reconsider. The 4x4 Strassen computes:
                # C11 = M1 + M4 - M5 + M7
                # C12 = M3 + M5
                # C21 = M2 + M4
                # C22 = M1 - M2 + M3 + M6
                #
                # But in our tensor decomposition, each "M" is itself a 2x2*2x2 product
                # which is decomposed into 7 scalar terms.
                #
                # The issue is that the 4x4 Strassen terms M1-M7 are 2x2*2x2 products,
                # and each is placed in specific C blocks with signs.
                #
                # So for each 4x4 Strassen term t (0-6) and each inner 2x2 Strassen term s (0-6),
                # we need to know which C blocks it contributes to and with what sign.
                
                # Let's define the C block contributions for each 4x4 Strassen term:
                # term 0 (M1): C11(+1), C22(+1)
                # term 1 (M2): C21(+1), C22(-1)
                # term 2 (M3): C12(+1), C22(+1)
                # term 3 (M4): C11(+1), C21(+1)
                # term 4 (M5): C11(-1), C12(+1)
                # term 5 (M6): C22(+1)
                # term 6 (M7): C11(+1)
                
                c_contribs = [
                    [(0, 1), (3, 1)],   # M1: C11, C22
                    [(2, 1), (3, -1)],  # M2: C21, C22
                    [(1, 1), (3, 1)],   # M3: C12, C22
                    [(0, 1), (2, 1)],   # M4: C11, C21
                    [(0, -1), (1, 1)],  # M5: C11, C12
                    [(3, 1)],           # M6: C22
                    [(0, 1)],           # M7: C11
                ]
                
                # Get the C contributions for this 4x4 Strassen term
                # We need to map back to the term index
                # a_specs and b_specs correspond to a specific term in strassen_4x4
                term_idx = strassen_4x4.index((a_specs, b_specs))
                c_specs = c_contribs[term_idx]
                
                for (c_bidx, c_coeff) in c_specs:
                    c_row, c_col = block_to_indices(c_bidx)
                    for (lk, li, inner_coeff) in w_entries_2x2:
                        gk = k_offset + c_row + lk
                        gi = c_col + li
                        W[gk * n + gi][col] += c_coeff * inner_coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
