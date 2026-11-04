def _block_decomposition_5x5x5():
    n = 5
    # Splits: [2, 2, 1]
    splits = [2, 2, 1]
    starts = [0, 2, 4]
    
    # Pre-compute ranks and decompositions for each shape
    # Shapes: (n_i, m_k, p_j)
    # We need U, V, W for each shape.
    
    def get_decomp(n_i, m_k, p_j):
        if n_i == 2 and m_k == 2 and p_j == 2:
            return _strassen_2x2x2()
        else:
            rank = n_i * m_k * p_j
            U = np.zeros((n_i * m_k, rank))
            V = np.zeros((m_k * p_j, rank))
            W = np.zeros((p_j * n_i, rank))
            col = 0
            for i in range(n_i):
                for j in range(m_k):
                    for k in range(p_j):
                        U[i * m_k + j, col] = 1.0
                        V[j * p_j + k, col] = 1.0
                        W[k * n_i + i, col] = 1.0
                        col += 1
            return U, V, W

    total_rank = 0
    decomps = {}
    for i_idx in range(3):
        for k_idx in range(3):
            for j_idx in range(3):
                n_i = splits[i_idx]
                m_k = splits[k_idx]
                p_j = splits[j_idx]
                key = (n_i, m_k, p_j)
                if key not in decomps:
                    decomps[key] = get_decomp(n_i, m_k, p_j)
                total_rank += decomps[key][0].shape[1]

    U = np.zeros((25, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((25, total_rank))
    
    col = 0
    for i_idx in range(3):
        i_start = starts[i_idx]
        n_i = splits[i_idx]
        for k_idx in range(3):
            k_start = starts[k_idx]
            m_k = splits[k_idx]
            for j_idx in range(3):
                j_start = starts[j_idx]
                p_j = splits[j_idx]
                
                key = (n_i, m_k, p_j)
                Ub, Vb, Wb = decomps[key]
                rank_b = Ub.shape[1]
                
                # Fill U
                # Global row: i_start + i, col: k_start + j
                # Local row: i, col: j
                for i in range(n_i):
                    for j in range(m_k):
                        global_row = (i_start + i) * 5 + (k_start + j)
                        local_row = i * m_k + j
                        for r in range(rank_b):
                            U[global_row, col + r] = Ub[local_row, r]
                
                # Fill V
                # Global row: k_start + j, col: j_start + k
                # Local row: j, col: k
                for j in range(m_k):
                    for k in range(p_j):
                        global_row = (k_start + j) * 5 + (j_start + k)
                        local_row = j * p_j + k
                        for r in range(rank_b):
                            V[global_row, col + r] = Vb[local_row, r]
                
                # Fill W
                # Global row: j_start + k, col: i_start + i
                # Local row: k, col: i
                for k in range(p_j):
                    for i in range(n_i):
                        global_row = (j_start + k) * 5 + (i_start + i)
                        local_row = k * n_i + i
                        for r in range(rank_b):
                            W[global_row, col + r] = Wb[local_row, r]
                
                col += rank_b
                
    return U, V, W
