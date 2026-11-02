def strassen_decomposition(n):
    if n == 1:
        return np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]])
    
    k = n // 2
    U_k, V_k, W_k = strassen_decomposition(k)
    r_k = U_k.shape[1]
    r_n = 7 * r_k
    
    U = np.zeros((n*n, r_n))
    V = np.zeros((n*n, r_n))
    W = np.zeros((n*n, r_n))
    
    # Helper to add a term
    # term_idx: 0..6
    # For each term, we define which sub-blocks of A, B, C are involved and with what signs.
    
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # C blocks: 00, 01, 10, 11
    
    # Strassen formulas:
    # M0 = (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
    # M1 = (A10+A11)B11 -> C01 += M1
    # M2 = A00(B10-B11) -> C10 += M2
    # M3 = (A00+A01)(B00-B10) -> C00 += M3
    # M4 = A11(B00+B01) -> C01 += M4, C11 += M4
    # M5 = (A00-A10)(B10+B11) -> C10 += M5
    # M6 = (A01-A11)(B00+B10) -> C00 += M6, C11 += M6
    
    # We need to map the sub-decomposition (U_k, V_k, W_k) to the global indices.
    
    # For a term t (0..6):
    # We have a set of (A_block, sign) and (B_block, sign)
    # And a set of (C_block, sign)
    
    # Let's define the structure.
    
    def map_block_indices(block_type, block_id, n, k):
        # Returns (row_start, row_end, col_start, col_end) for the block in the n x n matrix
        # block_id: 00, 01, 10, 11
        if block_id == 0: # 00
            return 0, k, 0, k
        elif block_id == 1: # 01
            return 0, k, k, n
        elif block_id == 2: # 10
            return k, n, 0, k
        elif block_id == 3: # 11
            return k, n, k, n
