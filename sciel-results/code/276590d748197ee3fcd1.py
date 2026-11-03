    # Assemble full 3x3x8 decomposition
    # 2 blocks of 3x3x3 (rank 26) + 1 block of 3x3x2 (rank 17)
    rank_333 = 26
    rank_332 = 17
    rank_total = 2 * rank_333 + rank_332
    
    U = np.zeros((9, rank_total))
    V = np.zeros((24, rank_total))
    W = np.zeros((24, rank_total))
    
    # Block 1: 3x3x3, k=0..2
    U333_1, V333_1, W333_1 = get_26_rank_block_3x3x3()
    col_start = 0
    col_end = rank_333
    U[:, col_start:col_end] = U333_1
    for c in range(rank_333):
        # Map V: local (j, k) -> global (j, k)
        for j in range(3):
            for k in range(3):
                V[j*8 + k, col_start + c] = V333_1[j*3 + k, c]
        # Map W: local (k, i) -> global (k, i). Since k_offset=0, indices match.
        W[0:9, col_start + c] = W333_1[:, c]
    
    # Block 2: 3x3x3, k=3..5
    U333_2, V333_2, W333_2 = get_26_rank_block_3x3x3()
    col_start = rank_333
    col_end = 2 * rank_333
    U[:, col_start:col_end] = U333_2
    for c in range(rank_333):
        # Map V: local (j, k) -> global (j, k+3)
        for j in range(3):
            for k in range(3):
                V[j*8 + (k + 3), col_start + c] = V333_2[j*3 + k, c]
        # Map W: local (k, i) -> global (k+3, i)
        W[9:18, col_start + c] = W333_2[:, c]
