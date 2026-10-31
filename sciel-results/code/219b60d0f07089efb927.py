"""4x4x4 matrix multiplication tensor decomposition, rank 49 (Bini et al. via Strassen tensor product)."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 rank-1 terms (local indices 0,1,2,3 for 2x2 matrix)
    # Local index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),                # M2
        ({0: 1}, {2: 1, 3: -1}, {1: 1}),               # M3
        ({3: 1}, {1: 1, 2: -1}, {0: 1}),               # M4
        ({0: 1, 1: 1}, {3: 1}, {0: 1}),                # M5
        ({2: 1, 0: -1}, {0: 1, 2: 1}, {3: 1}),         # M6
        ({1: 1, 3: -1}, {1: 1, 3: 1}, {1: 1}),         # M7
    ]

    # Block-level Strassen terms: (A_blocks, B_blocks, C_blocks)
    # Block indices: 0=00, 1=01, 2=10, 3=11
    block_strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),                # M2
        ({0: 1}, {2: 1, 3: -1}, {1: 1}),               # M3
        ({3: 1}, {1: 1, 2: -1}, {0: 1}),               # M4
        ({0: 1, 1: 1}, {3: 1}, {0: 1}),                # M5
        ({2: 1, 0: -1}, {0: 1, 2: 1}, {3: 1}),         # M6
        ({1: 1, 3: -1}, {1: 1, 3: 1}, {1: 1}),         # M7
    ]

    def block_to_base(block):
        """Convert block index (0,1,2,3) to base index in 4x4 matrix."""
        row_block = block // 2  # 0 or 1
        col_block = block % 2   # 0 or 1
        return row_block * 4 + col_block * 2

    def local_to_offset(local):
        """Convert local 2x2 index (0,1,2,3) to offset within block."""
        row_local = local // 2  # 0 or 1
        col_local = local % 2   # 0 or 1
        return row_local * 2 + col_local

    col = 0
    for b in range(7):
        A_blocks, B_blocks, C_blocks = block_strassen_terms[b]
        for s in range(7):
            u_local, v_local, w_local = strassen_terms[s]
            for a_block, a_sign in A_blocks.items():
                for a_local, a_lsign in u_local.items():
                    global_a = block_to_base(a_block) + local_to_offset(a_local)
                    U[global_a][col] += a_sign * a_lsign
            for b_block, b_sign in B_blocks.items():
                for b_local, b_lsign in v_local.items():
                    global_b = block_to_base(b_block) + local_to_offset(b_local)
                    V[global_b][col] += b_sign * b_lsign
            for c_block, c_sign in C_blocks.items():
                for c_local, c_lsign in w_local.items():
                    global_c = block_to_base(c_block) + local_to_offset(c_local)
                    W[global_c][col] += c_sign * c_lsign
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
