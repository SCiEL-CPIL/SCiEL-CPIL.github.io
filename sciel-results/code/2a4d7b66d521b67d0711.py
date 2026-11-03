def add_strassen_term(U, V, W, rank_offset, A_block_idx, B_block_idx, C_block_idx):
    # A_block_idx: (i_start, j_start) for 2x2 block
    # B_block_idx: (j_start, k_start) for 2x2 block
    # C_block_idx: (i_start, k_start) for 2x2 block
    
    i0, j0 = A_block_idx
    j1, k0 = B_block_idx
    i1, k1 = C_block_idx
    
    # Strassen formulas
    # P1 = (a00+a11)(b00+b11)
    # a00 is A[i0, j0], a11 is A[i0+1, j0+1]
    # b00 is B[j1, k0], b11 is B[j1+1, k0+1]
    
    def add_product(a_indices, b_indices, c_index, coeff=1):
        # a_indices: list of (i, j)
        # b_indices: list of (j, k)
        # c_index: (i, k)
        # This adds a term to U, V, W
        # U[a_idx, r] += coeff
        # V[b_idx, r] += coeff
        # W[c_idx, r] += coeff
        pass
