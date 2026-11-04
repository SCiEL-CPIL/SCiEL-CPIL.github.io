import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We will use a block decomposition strategy.
    # Split m=4 into two blocks of 2.
    # Split p=7 into blocks of 2, 2, 3.
    # For each m-block (size 2), we have a 3x2x7 problem.
    # We split the 3x2x7 problem into:
    #   - Two 3x2x2 sub-problems (for p-blocks of size 2)
    #   - One 3x2x3 sub-problem (for p-block of size 3)
    
    # For a 3x2x2 sub-problem:
    # Split n=3 into 2+1.
    #   - 2x2x2 uses Strassen (rank 7)
    #   - 1x2x2 uses schoolbook (rank 4)
    # Total rank for 3x2x2 is 11.
    
    # For a 3x2x3 sub-problem:
    # Split n=3 into 2+1.
    #   - 2x2x3: Split p=3 into 2+1.
    #       - 2x2x2 uses Strassen (rank 7)
    #       - 2x2x1 uses schoolbook (rank 4)
    #       Total 2x2x3 rank is 11.
    #   - 1x2x3 uses schoolbook (rank 6)
    # Total rank for 3x2x3 is 11 + 6 = 17.
    
    # Total rank per m-block (3x2x7): 11 + 11 + 17 = 39.
    # Total rank for two m-blocks: 39 * 2 = 78.
    
    total_rank = 78
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    rank_offset = 0
    
    # Helper to add Strassen 2x2x2 block
    def add_strassen_222(i_off, j_off, k_off, r_off):
        # Strassen's 7 terms for 2x2x2
        # U, V, W are 2x2 matrices for each term
        terms = [
            # (U, V, W)
            (np.array([[1, 0], [0, 1]], dtype=float), np.array([[1, 0], [0, 1]], dtype=float), np.array([[1, 0], [0, 1]], dtype=float)), # M1
            (np.array([[0, 0], [1, 1]], dtype=float), np.array([[1, 0], [0, 0]], dtype=float), np.array([[0, 0], [1, -1]], dtype=float)), # M2
            (np.array([[1, 0], [0, 0]], dtype=float), np.array([[0, 1], [0, -1]], dtype=float), np.array([[1, 0], [0, 1]], dtype=float)), # M3
            (np.array([[0, 0], [0, 1]], dtype=float), np.array([[-1, 0], [1, 0]], dtype=float), np.array([[1, 0], [1, 0]], dtype=float)), # M4
            (np.array([[1, 1], [0, 0]], dtype=float), np.array([[0, 0], [0, 1]], dtype=float), np.array([[-1, 1], [0, 0]], dtype=float)), # M5
            (np.array([[-1, 0], [1, 0]], dtype=float), np.array([[1, 1], [0, 0]], dtype=float), np.array([[0, 0], [0, 1]], dtype=float)), # M6
            (np.array([[0, 1], [0, -1]], dtype=float), np.array([[0, 1], [1, 1]], dtype=float), np.array([[1, 0], [0, 0]], dtype=float)), # M7
        ]
        
        for t in range(7):
            Ut, Vt, Wt = terms[t]
            col = r_off + t
            for i in range(2):
                for j in range(2):
                    U[(i_off + i) * m + (j_off + j), col] = Ut[i, j]
            for j in range(2):
                for k in range(2):
                    V[(j_off + j) * p + (k_off + k), col] = Vt[j, k]
            for k in range(2):
                for i in range(2):
                    W[(k_off + k) * n + (i_off + i), col] = Wt[k, i]
        return 7

    # Helper to add schoolbook block
    def add_schoolbook(i_off, j_off, k_off, r_off, n_len, m_len, p_len):
        for i in range(n_len):
            for j in range(m_len):
                for k in range(p_len):
                    U[(i_off + i) * m + (j_off + j), r_off] = 1.0
                    V[(j_off + j) * p + (k_off + k), r_off] = 1.0
                    W[(k_off + k) * n + (i_off + i), r_off] = 1.0
                    r_off += 1
        return r_off - (r_off - 0) # Just return count? No, return new offset.
    
    # Let's rewrite add_schoolbook to return the new rank offset
    def add_schoolbook(i_off, j_off, k_off, r_off, n_len, m_len, p_len):
        start_r = r_off
        for i in range(n_len):
            for j in range(m_len):
                for k in range(p_len):
                    U[(i_off + i) * m + (j_off + j), r_off] = 1.0
                    V[(j_off + j) * p + (k_off + k), r_off] = 1.0
                    W[(k_off + k) * n + (i_off + i), r_off] = 1.0
                    r_off += 1
        return r_off

    # Iterate over m-blocks (j_off = 0, 2)
    for j_block in range(2):
        j_off = j_block * 2
        # Iterate over p-blocks
        # p-blocks: 0 (size 2), 1 (size 2), 2 (size 3)
        p_sizes = [2, 2, 3]
        p_offs = [0, 2, 4]
        
        for p_idx in range(3):
            k_off = p_offs[p_idx]
            k_len = p_sizes[p_idx]
            
            if k_len == 2:
                # 3x2x2 block
                # Split n=3 into 2+1
                # 2x2x2 part
                rank_offset += add_strassen_222(0, j_off, k_off, rank_offset)
                # 1x2x2 part (i_off=2)
                rank_offset = add_schoolbook(2, j_off, k_off, rank_offset, 1, 2, 2)
            else: # k_len == 3
                # 3x2x3 block
                # Split n=3 into 2+1
                # 2x2x3 part (i_off=0)
                # Split p=3 into 2+1
                # 2x2x2 part
                rank_offset += add_strassen_222(0, j_off, k_off, rank_offset)
                # 2x2x1 part (k_off+2)
                rank_offset = add_schoolbook(0, j_off, k_off + 2, rank_offset, 2, 2, 1)
                # 1x2x3 part (i_off=2)
                rank_offset = add_schoolbook(2, j_off, k_off, rank_offset, 1, 2, 3)

    return {"U": U, "V": V, "W": W}
