"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6

    # Use the known rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    # We decompose 4x4x6 into 6 independent 2x2x2 multiplications, but each block product
    # C[i_block, k_block] = A[i_block, 0]*B[0, k_block] + A[i_block, 1]*B[1, k_block]
    # requires two 2x2 matrix products, each of rank 7.
    # Total rank = 6 * 2 * 7 = 84.

    # Strassen 7 terms for 2x2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b00)
    # M4 = a11(b01-b11)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 - M5 + M4 + M7
    # C10 = M2 + M6
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6

    # U vectors (for A, 4 elements: a00, a01, a10, a11):
    U_strassen = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [1, 0, -1, 0], # M6
        [0, 1, 0, -1], # M7
    ]

    # V vectors (for B, 4 elements: b00, b01, b10, b11):
    V_strassen = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 0, 1, -1], # M3
        [0, 1, 0, -1], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1],  # M7
    ]

    # W vectors (for C, 4 elements in order: c00, c10, c01, c11)
    W_strassen = [
        [1, 0, 0, 1],   # M1
        [0, 1, 0, -1],  # M2
        [0, 0, 1, 1],   # M3
        [1, 0, 0, 0],   # M4
        [-1, 0, 1, 0],  # M5
        [0, 1, 0, 1],   # M6
        [1, 0, 0, 0],   # M7
    ]

    rank = 84
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(24)]
    W = [[0] * rank for _ in range(24)]

    col = 0
    for i_block in range(2):
        for k_block in range(3):
            for j_block in range(2):
                for s in range(7):
                    # U: A[i_block, j_block]
                    a_indices = [
                        (i_block * 2 + 0) * 4 + (j_block * 2 + 0),
                        (i_block * 2 + 0) * 4 + (j_block * 2 + 1),
                        (i_block * 2 + 1) * 4 + (j_block * 2 + 0),
                        (i_block * 2 + 1) * 4 + (j_block * 2 + 1),
                    ]
                    for a_idx, val in zip(a_indices, U_strassen[s]):
                        U[a_idx][col] += val

                    # V: B[j_block, k_block]
                    b_indices = [
                        (j_block * 2 + 0) * 6 + (k_block * 2 + 0),
                        (j_block * 2 + 0) * 6 + (k_block * 2 + 1),
                        (j_block * 2 + 1) * 6 + (k_block * 2 + 0),
                        (j_block * 2 + 1) * 6 + (k_block * 2 + 1),
                    ]
                    for b_idx, val in zip(b_indices, V_strassen[s]):
                        V[b_idx][col] += val

                    # W: C[i_block, k_block]
                    c_indices = [
                        (k_block * 2 + 0) * 4 + (i_block * 2 + 0),
                        (k_block * 2 + 0) * 4 + (i_block * 2 + 1),
                        (k_block * 2 + 1) * 4 + (i_block * 2 + 0),
                        (k_block * 2 + 1) * 4 + (i_block * 2 + 1),
                    ]
                    for c_idx, val in zip(c_indices, W_strassen[s]):
                        W[c_idx][col] += val

                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
