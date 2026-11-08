"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Uses a block decomposition:
  - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
  - A*B1 and A*B2 use a rank-49 bilinear algorithm for 4x4*4x4
  - A*B3 uses schoolbook for 4x4*4x1 (rank 16)
  - Total rank: 49 + 49 + 16 = 114
"""

import time

DIMENSIONS = [4, 4, 9]


def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses block decomposition with rank 114.
    """
    start_time = time.time()
    deadline = start_time + 120

    n, m, p = DIMENSIONS  # n=4, m=4, p=9

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term to the decomposition."""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Strassen decomposition for 2x2 * 2x2
    strassen_2x2_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    def strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """
        Add rank-49 Strassen decomposition for 4x4 * 4x4.
        """
        
        def get_A_entry(gi, gj):
            return gi * m + gj

        def get_B_entry(gj, gk):
            return gj * p + gk

        def get_C_entry(gi, gk):
            return gk * n + gi

        def add_block_mult(A_blocks, B_blocks, C_blocks):
            for u_coeffs, v_coeffs, w_coeffs in strassen_2x2_terms:
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)

                for local_idx, meta_coeff in u_coeffs.items():
                    li = local_idx // 2
                    lj = local_idx % 2
                    for (br, bc, block_coeff) in A_blocks:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[get_A_entry(gi, gj)] += meta_coeff * block_coeff

                for local_idx, meta_coeff in v_coeffs.items():
                    lj = local_idx // 2
                    lk = local_idx % 2
                    for (br, bc, block_coeff) in B_blocks:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[get_B_entry(gj, gk)] += meta_coeff * block_coeff

                for local_idx, meta_coeff in w_coeffs.items():
                    li = local_idx // 2
                    lk = local_idx % 2
                    for (br, bc, block_coeff) in C_blocks:
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[get_C_entry(gi, gk)] += meta_coeff * block_coeff

                add_term(u_vec, v_vec, w_vec)

        # M1 = (A11+A22)(B11+B22)
        add_block_mult(
            A_blocks=[(0, 0, 1), (1, 1, 1)],
            B_blocks=[(0, 0, 1), (1, 1, 1)],
            C_blocks=[(0, 0, 1), (1, 1, 1)]
        )

        # M2 = (A21+A22)B11
        add_block_mult(
            A_blocks=[(1, 0, 1), (1, 1, 1)],
            B_blocks=[(0, 0, 1)],
            C_blocks=[(1, 0, 1), (1, 1, -1)]
        )

        # M3 = A11(B12-B22)
        add_block_mult(
            A_blocks=[(0, 0, 1)],
            B_blocks=[(0, 1, 1), (1, 1, -1)],
            C_blocks=[(0, 1, 1), (1, 1, 1)]
        )

        # M4 = A22(B21-B11)
        add_block_mult(
            A_blocks=[(1, 1, 1)],
            B_blocks=[(1, 0, 1), (0, 0, -1)],
            C_blocks=[(0, 0, 1), (1, 0, 1)]
        )

        # M5 = (A11+A12)B22
        add_block_mult(
            A_blocks=[(0, 0, 1), (0, 1, 1)],
            B_blocks=[(1, 1, 1)],
            C_blocks=[(0, 0, -1), (0, 1, 1)]
        )

        # M6 = (A21-A11)(B11+B12)
        add_block_mult(
            A_blocks=[(1, 0, 1), (0, 0, -1)],
            B_blocks=[(0, 0, 1), (0, 1, 1)],
            C_blocks=[(1, 1, 1)]
        )

        # M7 = (A12-A22)(B21+B22)
        add_block_mult(
            A_blocks=[(0, 1, 1), (1, 1, -1)],
            B_blocks=[(1, 0, 1), (1, 1, 1)],
            C_blocks=[(0, 0, 1)]
        )

    # Part 1: A * B1 (cols 0-3)
    strassen_4x4(0, 0, 0, 0, 0, 0)

    # Part 2: A * B2 (cols 4-7)
    strassen_4x4(0, 0, 0, 4, 0, 4)

    # Part 3: A * B3 (col 8)
    for i in range(4):
        for j in range(4):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)

    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]

    return {"U": U, "V": V, "W": W}
