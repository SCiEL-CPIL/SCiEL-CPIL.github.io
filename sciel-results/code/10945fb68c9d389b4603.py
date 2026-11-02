"""Matrix multiplication 4,4,8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 8

    # Strassen's 7-term decomposition for 2x2x2 matrix multiplication
    # Each term is (u, v, w) where u, v, w are 4-element vectors
    # representing flattened 2x2 matrices (row-major: 00, 01, 10, 11)
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1 = (a+d)(e+h)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2 = (c+f)e
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3 = a(g-h)
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),   # M4 = d(f-h)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5 = (a+b)(h-d)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6 = (c-d)(e+f)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])    # M7 = (a-c)(e+h)
    ]

    # For 4x4x8 multiplication, we decompose into block structure.
    # A is 4x4, B is 4x8, C is 4x8.
    # We view B and C as having two 4x4 blocks along the column dimension.
    # B = [B_0 | B_1] where B_0, B_1 are 4x4
    # C = [C_0 | C_1] where C_0, C_1 are 4x4
    # So C_0 = A * B_0 and C_1 = A * B_1
    
    # For each 4x4x4 multiplication (A * B_k -> C_k), we use the standard
    # block decomposition with 2x2 blocks, giving 8 products of 2x2 matrices.
    # Each 2x2x2 product uses 7 scalar multiplications (Strassen).
    # So each 4x4x4 uses 8 * 7 = 56 scalar products.
    # Total for 4x4x8: 2 * 56 = 112 scalar products.

    rank = 112

    # Precompute block index mappings for 4x4 matrices
    # Block (bi, bj) of a 4x4 matrix: rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # Flattened index = row * 4 + col
    def block_2x2_indices(bi, bj):
        indices = []
        for dr in range(2):
            for dc in range(2):
                r = 2 * bi + dr
                c = 2 * bj + dc
                indices.append(r * 4 + c)
        return indices

    # Precompute for all 4 blocks
    bi_indices = {(bi, bj): block_2x2_indices(bi, bj) for bi in range(2) for bj in range(2)}

    # The 8 products for 4x4x4: (a_block, b_block, c_block)
    # C_00 = A_00*B_00 + A_01*B_10
    # C_01 = A_00*B_01 + A_01*B_11
    # C_10 = A_10*B_00 + A_11*B_10
    # C_11 = A_10*B_01 + A_11*B_11
    products = [
        ((0, 0), (0, 0), (0, 0)),
        ((0, 1), (1, 0), (0, 0)),
        ((0, 0), (0, 1), (0, 1)),
        ((0, 1), (1, 1), (0, 1)),
        ((1, 0), (0, 0), (1, 0)),
        ((1, 1), (1, 0), (1, 0)),
        ((1, 0), (0, 1), (1, 1)),
        ((1, 1), (1, 1), (1, 1))
    ]

    # Initialize factor matrices
    # U: (n*m, rank) = (16, 112)
    # V: (m*p, rank) = (32, 112)
    # W: (p*n, rank) = (32, 112)
    U = [[0.0] * rank for _ in range(16)]
    V = [[0.0] * rank for _ in range(32)]
    W = [[0.0] * rank for _ in range(32)]

    for block in range(2):  # Two 4x4 blocks for the 4x8 dimension
        base_col = block * 56
        for prod_idx, (a_blk, b_blk, c_blk) in enumerate(products):
            for s_idx, (u_2, v_2, w_2) in enumerate(strassen_2x2):
                col = base_col + prod_idx * 7 + s_idx

                # Fill U: A is 4x4, index = row * 4 + col
                u_idx_list = bi_indices[a_blk]
                for k, flat_idx in enumerate(u_idx_list):
                    U[flat_idx][col] = u_2[k]

                # Fill V: B is 4x8, index = row * 8 + col
                # For block k, the columns are 4*block to 4*block+3
                v_idx_list = bi_indices[b_blk]
                for k, flat_idx in enumerate(v_idx_list):
                    b_row = flat_idx // 4
                    b_col_local = flat_idx % 4
                    b_col_global = 4 * block + b_col_local
                    v_flat = b_row * 8 + b_col_global
                    V[v_flat][col] = v_2[k]

                # Fill W: C is 4x8, but W convention is W[k*n + i, r]
                # where k is column index (0..7) and i is row index (0..3)
                # So W flat index = col * 4 + row
                w_idx_list = bi_indices[c_blk]
                for k, flat_idx in enumerate(w_idx_list):
                    c_row = flat_idx // 4
                    c_col_local = flat_idx % 4
                    c_col_global = 4 * block + c_col_local
                    w_flat = c_col_global * 4 + c_row
                    W[w_flat][col] = w_2[k]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
