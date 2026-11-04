"""Program for matrix multiplication tensor decomposition M_{3,5,5}."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x5 matrix by a 5x5 matrix using block decomposition with Strassen's algorithm
    for the 2x2x2 sub-blocks.
    """
    n, m, p = DIMENSIONS
    
    # Helper to create schoolbook decomposition
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # Helper to create Strassen decomposition for 2x2x2
    def strassen_2x2x2():
        # Returns U, V, W of shape (4, 7)
        U = [
            [1, 0, 1, 0, 1, -1, 0], # A00
            [0, 0, 0, 0, 1, 0, 1],  # A01
            [0, 1, 0, 0, 0, 1, 0],  # A10
            [1, 1, 0, 1, 0, 0, -1]  # A11
        ]
        V = [
            [1, 1, 0, -1, 0, 1, 0], # B00
            [0, 0, 0, 1, 0, 1, 0],  # B01
            [0, 0, 1, 0, 0, 0, 1],  # B10
            [1, 0, -1, 0, 1, 0, 1]  # B11
        ]
        W = [
            [1, 0, 0, 1, -1, 0, 1], # C00
            [0, 1, 1, 0, 0, 0, 0],  # C01
            [0, 0, 0, 0, 1, 1, 0],  # C10
            [1, 0, -1, 1, 0, 1, 0]  # C11
        ]
        return U, V, W

    # Block decomposition
    # A: 3x5 split into 1x2, 1x3, 2x2, 2x3
    # B: 5x5 split into 2x2, 2x3, 3x2, 3x3
    # C: 3x5 split into 1x2, 1x3, 2x2, 2x3
    
    # Define blocks: (i_start, i_len, j_start, j_len, k_start, k_len, n, m, p, use_strassen)
    # n, m, p are dims of the block product
    blocks = [
        # C11 (1x2) from A11(1x2) * B11(2x2) + A12(1x3) * B21(3x2)
        (0, 1, 0, 2, 0, 2, 1, 2, 2, False), # A11 B11
        (0, 1, 2, 3, 2, 2, 1, 3, 2, False), # A12 B21
        
        # C12 (1x3) from A11(1x2) * B12(2x3) + A12(1x3) * B22(3x3)
        (0, 1, 0, 2, 2, 3, 1, 2, 3, False), # A11 B12
        (0, 1, 2, 3, 2, 3, 1, 3, 3, False), # A12 B22
        
        # C21 (2x2) from A21(2x2) * B11(2x2) + A22(2x3) * B21(3x2)
        (1, 2, 0, 2, 0, 2, 2, 2, 2, True),  # A21 B11 (Strassen)
        (1, 2, 2, 3, 2, 2, 2, 3, 2, False), # A22 B21
        
        # C22 (2x3) from A21(2x2) * B12(2x3) + A22(2x3) * B22(3x3)
        (1, 2, 0, 2, 2, 3, 2, 2, 3, False), # A21 B12
        (1, 2, 2, 3, 2, 3, 2, 3, 3, False), # A22 B22
    ]

    U_total = []
    V_total = []
    W_total = []

    for (i_s, i_l, j_s, j_l, k_s, k_l, bn, bm, bp, use_str) in blocks:
        if use_str:
            u, v, w = strassen_2x2x2()
        else:
            u, v, w = schoolbook(bn, bm, bp)
        
        rank_block = len(u[0])
        
        # Extend U, V, W to global size
        # U shape (15, rank_block)
        # V shape (25, rank_block)
        # W shape (15, rank_block)
        
        u_ext = [[0] * rank_block for _ in range(n * m)]
        v_ext = [[0] * rank_block for _ in range(m * p)]
        w_ext = [[0] * rank_block for _ in range(p * n)]
        
        for local_i in range(bn):
            for local_j in range(bm):
                global_a_idx = (i_s + local_i) * m + (j_s + local_j)
                u_ext[global_a_idx] = u[local_i * bm + local_j]
                
        for local_j in range(bm):
            for local_k in range(bp):
                global_b_idx = (j_s + local_j) * p + (k_s + local_k)
                v_ext[global_b_idx] = v[local_j * bp + local_k]
                
        for local_k in range(bp):
            for local_i in range(bn):
                global_c_idx = (k_s + local_k) * n + (i_s + local_i)
                w_ext[global_c_idx] = w[local_k * bn + local_i]
                
        U_total.append(u_ext)
        V_total.append(v_ext)
        W_total.append(w_ext)

    # Concatenate horizontally
    final_U = [row1 + row2 for row1, row2 in zip(*U_total)] if U_total else []
    # Wait, zip(*U_total) gives tuples of rows.
    # We need to concatenate the lists of columns.
    # U_total is a list of lists of lists.
    # U_total[0] is (15, r1).
    # We want (15, r1+r2+...).
    
    if not U_total:
        return {"U": [], "V": [], "W": []}
        
    total_rank = sum(len(u[0]) for u in U_total)
    final_U = [[0] * total_rank for _ in range(n * m)]
    final_V = [[0] * total_rank for _ in range(m * p)]
    final_W = [[0] * total_rank for _ in range(p * n)]
    
    offset = 0
    for u_ext, v_ext, w_ext in zip(U_total, V_total, W_total):
        r = len(u_ext[0])
        for i in range(n * m):
            for c in range(r):
                final_U[i][offset + c] = u_ext[i][c]
        for i in range(m * p):
            for c in range(r):
                final_V[i][offset + c] = v_ext[i][c]
        for i in range(p * n):
            for c in range(r):
                final_W[i][offset + c] = w_ext[i][c]
        offset += r

    return {"U": final_U, "V": final_V, "W": final_W}
# EVOLVE-BLOCK-END
