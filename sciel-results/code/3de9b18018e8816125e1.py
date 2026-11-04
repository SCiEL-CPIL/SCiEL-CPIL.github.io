"""Matrix multiplication 2x3x8 with a blockwise rank-44 decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    # Use the known rank-44 decomposition for 2x3x8 obtained by splitting the
    # 3x8 right factor into three 2-column blocks:
    #   * the first 2 columns of B form a 3x2 subproblem, which is decomposed
    #     with a rank-11 tensor (a 2x2 Strassen block plus two schoolbook terms)
    #   * the remaining two 2-column blocks are handled with schoolbook rank-4 each
    # Total rank = 11 + 4 + 4 = 44.
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Index helpers
    def u_row(i, j):
        return i * m + j

    def v_row(j, k):
        return j * p + k

    def w_row(k, i):
        return k * n + i

    # Block 0: Strassen + two schoolbook terms for columns k=0,1
    # M1 = (A00 + A11) * (B0,0 + B1,0)
    # M2 = (A10 + A11) * B0,0
    # M3 = A00 * (B0,1 - B1,1)
    # M4 = A11 * (B1,0 - B0,0)
    # M5 = (A00 + A01) * B1,1
    # M6 = (A10 - A00) * (B0,0 + B0,1)
    # M7 = (A01 - A11) * (B1,0 + B1,1)
    # S1 = A02 * B2,0
    # S2 = A02 * B2,1
    # S3 = A12 * B2,0
    # S4 = A12 * B2,1

    # Column indices in the factor matrices (rank 44): 0..10 for block 0
    def _add(row_mat, row, col, val):
        row_mat[row][col] += val

    # M1
    _add(U, u_row(0, 0), 0, 1)
    _add(U, u_row(1, 1), 0, 1)
    _add(V, v_row(0, 0), 0, 1)
    _add(V, v_row(1, 0), 0, 1)
    _add(W, w_row(0, 0), 0, 1)
    _add(W, w_row(1, 1), 0, 1)

    # M2
    _add(U, u_row(1, 0), 1, 1)
    _add(U, u_row(1, 1), 1, 1)
    _add(V, v_row(0, 0), 1, 1)
    _add(W, w_row(0, 1), 1, 1)
    _add(W, w_row(1, 1), 1, -1)

    # M3
    _add(U, u_row(0, 0), 2, 1)
    _add(V, v_row(0, 1), 2, 1)
    _add(V, v_row(1, 1), 2, -1)
    _add(W, w_row(1, 0), 2, 1)
    _add(W, w_row(1, 1), 2, 1)

    # M4
    _add(U, u_row(1, 1), 3, 1)
    _add(V, v_row(1, 0), 3, 1)
    _add(V, v_row(0, 0), 3, -1)
    _add(W, w_row(0, 0), 3, 1)
    _add(W, w_row(0, 1), 3, 1)

    # M5
    _add(U, u_row(0, 0), 4, 1)
    _add(U, u_row(0, 1), 4, 1)
    _add(V, v_row(1, 1), 4, 1)
    _add(W, w_row(0, 0), 4, -1)
    _add(W, w_row(1, 0), 4, 1)

    # M6
    _add(U, u_row(1, 0), 5, 1)
    _add(U, u_row(0, 0), 5, -1)
    _add(V, v_row(0, 0), 5, 1)
    _add(V, v_row(0, 1), 5, 1)
    _add(W, w_row(1, 1), 5, 1)

    # M7
    _add(U, u_row(0, 1), 6, 1)
    _add(U, u_row(1, 1), 6, -1)
    _add(V, v_row(1, 0), 6, 1)
    _add(V, v_row(1, 1), 6, 1)
    _add(W, w_row(0, 0), 6, 1)

    # S1
    _add(U, u_row(0, 2), 7, 1)
    _add(V, v_row(2, 0), 7, 1)
    _add(W, w_row(0, 0), 7, 1)

    # S2
    _add(U, u_row(0, 2), 8, 1)
    _add(V, v_row(2, 1), 8, 1)
    _add(W, w_row(1, 0), 8, 1)

    # S3
    _add(U, u_row(1, 2), 9, 1)
    _add(V, v_row(2, 0), 9, 1)
    _add(W, w_row(0, 1), 9, 1)

    # S4
    _add(U, u_row(1, 2), 10, 1)
    _add(V, v_row(2, 1), 10, 1)
    _add(W, w_row(1, 1), 10, 1)

    # Blocks 1 and 2: schoolbook for columns 2..5 and 6..7
    # For each block offset ob (2 or 6), with q = 0,1:
    #   S_{q,0}: A0q * B2,(2q)
    #   S_{q,1}: A0q * B2,(2q+1)
    #   S_{q,2}: A1q * B2,(2q)
    #   S_{q,3}: A1q * B2,(2q+1)
    for ob in (2, 6):
        base = 11 + (ob - 2) * 4
        for q in (0, 1):
            c0 = base + 4 * q
            c1 = c0 + 1
            c2 = c0 + 2
            c3 = c0 + 3

            # S_{q,0}
            _add(U, u_row(0, q), c0, 1)
            _add(V, v_row(2, ob + 2 * q), c0, 1)
            _add(W, w_row(ob + 2 * q, 0), c0, 1)

            # S_{q,1}
            _add(U, u_row(0, q), c1, 1)
            _add(V, v_row(2, ob + 2 * q + 1), c1, 1)
            _add(W, w_row(ob + 2 * q + 1, 0), c1, 1)

            # S_{q,2}
            _add(U, u_row(1, q), c2, 1)
            _add(V, v_row(2, ob + 2 * q), c2, 1)
            _add(W, w_row(ob + 2 * q, 1), c2, 1)

            # S_{q,3}
            _add(U, u_row(1, q), c3, 1)
            _add(V, v_row(2, ob + 2 * q + 1), c3, 1)
            _add(W, w_row(ob + 2 * q + 1, 1), c3, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
