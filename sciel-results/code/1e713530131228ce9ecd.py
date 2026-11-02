import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _verify(U, V, W, T):
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    T_rec = np.zeros_like(T, dtype=np.float64)
    for r in range(rank):
        T_rec += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_rec, T, atol=1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 4, 5

    # We build a rank-65 decomposition:
    #   49 terms: Strassen-based decomposition of the 4x4x4 matmul (first 4 cols of B)
    #   16 terms: schoolbook for the 4x4x1 slice (5th column of B)
    #
    # Total rank = 49 + 16 = 65
    #
    # The 4x4x4 part uses the 2-level Strassen decomposition:
    #   Level 1: 7 block terms, each multiplying two 2x2 blocks
    #   Level 2: each 2x2x2 multiplication uses 7 scalar Strassen terms
    #   Total: 7 * 7 = 49 scalar terms
    #
    # Index conventions:
    #   U index: i*m + j  (i = row of A, j = shared index)
    #   V index: j*p + k  (j = shared index, k = col of B)
    #   W index: k*n + i  (k = col of C, i = row of C)  -- trace convention

    rank = 65
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    # --- Part 1: 49 Strassen terms for 4x4x4 (columns 0..3 of B) ---
    #
    # 2x2 Strassen scalar terms:
    #   For X (2x2), Y (2x2) -> Z = X*Y
    #   X indices: 0=X00, 1=X01, 2=X10, 3=X11
    #   Y indices: 0=Y00, 1=Y01, 2=Y10, 3=Y11
    #   Z indices: 0=Z00, 1=Z01, 2=Z10, 3=Z11
    #
    #   t1: (X00+X11)(Y00+Y11) -> Z00 += 1, Z11 += 1
    #   t2: (X10+X11)Y00       -> Z10 += 1
    #   t3: X00(Y10-Y11)       -> Z00 += 1
    #   t4: X11(Y01-Y00)       -> Z11 += 1
    #   t5: (X00+X01)Y11       -> Z01 += 1
    #   t6: (X10-X00)(Y00+Y10) -> Z10 += 1
    #   t7: (X01-X11)(Y10+Y11) -> Z01 += 1

    # Scalar Strassen template: list of (x_coeffs, y_coeffs, z_coeffs)
    # x_coeffs: dict {local_x_idx: coeff}
    # y_coeffs: dict {local_y_idx: coeff}
    # z_coeffs: dict {local_z_idx: coeff}
    strassen_2x2 = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # t1
        ({2: 1, 3: 1}, {0: 1},       {2: 1}),             # t2
        ({0: 1},       {2: 1, 3: -1},{0: 1}),             # t3
        ({3: 1},       {1: 1, 0: -1},{3: 1}),             # t4
        ({0: 1, 1: 1}, {3: 1},       {1: 1}),             # t5
        ({2: 1, 0: -1},{0: 1, 2: 1}, {2: 1}),             # t6
        ({1: 1, 3: -1},{2: 1, 3: 1}, {1: 1}),             # t7
    ]

    # 2x2 block local indices: 0=00, 1=01, 2=10, 3=11
    # For a 2x2 block at global position (br, bc) in a 4x4 matrix:
    #   local 0 -> (br, bc), local 1 -> (br, bc+1)
    #   local 2 -> (br+1, bc), local 3 -> (br+1, bc+1)

    # A block positions (row_block, col_block) in 4x4:
    #   A00: (0,0), A01: (0,1), A10: (1,0), A11: (1,1)
    # B block positions: B00: (0,0), B01: (0,1), B10: (1,0), B11: (1,1)
    # C block positions: C00: (0,0), C01: (0,1), C10: (1,0), C11: (1,1)

    # The 7 Strassen block-level terms for 4x4 = (2x2 blocks):
    #   M1: X = A00+A11, Y = B00+B11
    #   M2: X = A10+A11, Y = B00
    #   M3: X = A00,       Y = B10-B11
    #   M4: X = A11,       Y = B01-B00
    #   M5: X = A00+A01,   Y = B11
    #   M6: X = A10-A00,   Y = B00+B10
    #   M7: X = A01-A11,   Y = B10+B11
    #
    # C00 = M1 - M2 + M6
    # C01 = M2 + M4
    # C10 = M3 + M7
    # C11 = M1 - M4 + M5 - M7

    # Each M term is a product of two 2x2 matrices (linear combos of A blocks and B blocks)
    # For each M term, we expand using the 7 scalar Strassen terms.
    #
    # For M term with X = sum_a alpha_a * A_block_a, Y = sum_b beta_b * B_block_b:
    #   The 2x2 product X*Y is computed via Strassen.
    #   For each scalar Strassen term s (with x_coeffs, y_coeffs, z_coeffs):
    #     U contribution: for each (local_x, cx) in x_coeffs:
    #         for each (block_a, alpha) in X_blocks:
    #             global (i,j) = block_a position + local_x offset
    #             U[i*m+j, col] += alpha * cx
    #     V contribution: for each (local_y, cy) in y_coeffs:
    #         for each (block_b, beta) in Y_blocks:
    #             global (j,k) = block_b position + local_y offset
    #             V[j*p+k, col] += beta * cy
    #     W contribution: for each (local_z, cz) in z_coeffs:
    #         for each (block_c, gamma) in C_blocks:
    #             global (i,k) = block_c position + local_z offset
    #             W[k*n+i, col] += gamma * cz

    # Define block positions
    # A blocks: (row_block, col_block) -> (i_offset, j_offset)
    #   A00: (0,0) -> i in {0,1}, j in {0,1}
    #   A01: (0,1) -> i in {0,1}, j in {2,3}
    #   A10: (1,0) -> i in {2,3}, j in {0,1}
    #   A11: (1,1) -> i in {2,3}, j in {2,3}

    def a_block_indices(block_id):
        """Return list of (i, j) for the 4 entries of A block."""
        br, bc = block_id  # row_block, col_block
        return [(br*2 + dr, bc*2 + dc) for dr in range(2) for dc in range(2)]

    def b_block_indices(block_id):
        """Return list of (j, k) for the 4 entries of B block (only cols 0..3 for 4x4 part)."""
        br, bc = block_id  # row_block, col_block
        return [(br*2 + dr, bc*2 + dc) for dr in range(2) for dc in range(2)]

    def c_block_indices(block_id):
        """Return list of (i, k) for the 4 entries of C block."""
        br, bc = block_id  # row_block, col_block
        return [(br*2 + dr, bc*2 + dc) for dr in range(2) for dc in range(2)]

    # Local 2x2 index to (dr, dc) offset
    local_offsets = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}

    # The 7 M terms: (X_blocks, Y_blocks, C_blocks)
    # X_blocks: list of (block_id, coeff)
    # Y_blocks: list of (block_id, coeff)
    # C_blocks: list of (block_id, coeff)
    m_terms = [
        # M1: X=A00+A11, Y=B00+B11, contributes to C00(+1), C11(+1)
        ([( (0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)]),
        # M2: X=A10+A11, Y=B00, contributes to C00(-1), C01(+1)
        ([((1,0), 1), ((1,1), 1)],
         [((0,0), 1)],
         [((0,0), -1), ((0,1), 1)]),
        # M3: X=A00, Y=B10-B11, contributes to C10(+1)
        ([((0,0), 1)],
         [((1,0), 1), ((1,1), -1)],
         [((1,0), 1)]),
        # M4: X=A11, Y=B01-B00, contributes to C01(+1), C11(-1)
        ([((1,1), 1)],
         [((0,1), 1), ((0,0), -1)],
         [((0,1), 1), ((1,1), -1)]),
        # M5: X=A00+A01, Y=B11, contributes to C11(+1)
        ([((0,0), 1), ((0,1), 1)],
         [((1,1), 1)],
         [((1,1), 1)]),
        # M6: X=A10-A00, Y=B00+B10, contributes to C00(+1)
        ([((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((1,0), 1)],
         [((0,0), 1)]),
        # M7: X=A01-A11, Y=B10+B11, contributes to C10(+1), C11(-1)
        ([((0,1), 1), ((1,1), -1)],
         [((1,0), 1), ((1,1), 1)],
         [((1,0), 1), ((1,1), -1)]),
    ]

    col = 0
    for m_idx, (X_blocks, Y_blocks, C_blocks) in enumerate(m_terms):
        for s_idx, (x_coeffs, y_coeffs, z_coeffs) in enumerate(strassen_2x2):
            # This is one scalar term (col) in the decomposition
            # U: for each (local_x, cx) in x_coeffs, for each (block_a, alpha) in X_blocks
            for local_x, cx in x_coeffs.items():
                dr, dc = local_offsets[local_x]
                for block_a, alpha in X_blocks:
                    br, bc = block_a
                    i = br * 2 + dr
                    j = bc * 2 + dc
                    U[i * m + j, col] += alpha * cx

            # V: for each (local_y, cy) in y_coeffs, for each (block_b, beta) in Y_blocks
            for local_y, cy in y_coeffs.items():
                dr, dc = local_offsets[local_y]
                for block_b, beta in Y_blocks:
                    br, bc = block_b
                    j = br * 2 + dr
                    k = bc * 2 + dc  # k is 0..3 (first 4 columns)
                    V[j * p + k, col] += beta * cy

            # W: for each (local_z, cz) in z_coeffs, for each (block_c, gamma) in C_blocks
            for local_z, cz in z_coeffs.items():
                dr, dc = local_offsets[local_z]
                for block_c, gamma in C_blocks:
                    br, bc = block_c
                    i = br * 2 + dr
                    k = bc * 2 + dc  # k is 0..3 (first 4 columns of C)
                    W[k * n + i, col] += gamma * cz

            col += 1

    # col should be 49 here
    assert col == 49, f"Expected 49 Strassen terms, got {col}"

    # --- Part 2: 16 schoolbook terms for 4x4x1 (column 4 of B) ---
    # For each (i, j): C[i, 4] += A[i, j] * B[j, 4]
    # U[i*m+j, col] = 1, V[j*p+4, col] = 1, W[4*n+i, col] = 1
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0
            W[4 * n + i, col] = 1.0
            col += 1

    assert col == 65, f"Expected rank 65, got {col}"

    # Verify the decomposition
    T = _build_tensor(n, m, p)
    if not _verify(U, V, W, T):
        # Fallback: pure schoolbook rank 80
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=np.float64)
        V_fb = np.zeros((m * p, rank_fb), dtype=np.float64)
        W_fb = np.zeros((p * n, rank_fb), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U_fb[i * m + j, c] = 1
                    V_fb[j * p + k, c] = 1
                    W_fb[k * n + i, c] = 1
        U, V, W = U_fb, V_fb, W_fb

    return {
        "U": [[float(x) for x in U[i]] for i in range(U.shape[0])],
        "V": [[float(x) for x in V[i]] for i in range(V.shape[0])],
        "W": [[float(x) for x in W[i]] for i in range(W.shape[0])]
    }
