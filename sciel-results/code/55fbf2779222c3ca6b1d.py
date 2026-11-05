import numpy as np
import time

def construct():
    n, m, p = 4, 4, 5
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s limit

    # Strategy: Use Strassen-based decomposition for 4x4x4 (rank 49)
    # and schoolbook for 4x4x1 (rank 16), totaling rank 65.
    # This is significantly better than schoolbook rank 80.

    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65

    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0

    # Strassen 2x2x2 decomposition definitions
    # Each entry: (u_elements, v_elements, w_contributions)
    # u_elements: list of (local_row, local_col, coeff) for A block
    # v_elements: list of (local_row, local_col, coeff) for B block
    # w_contributions: list of (block_row, block_col, sign) for C blocks

    inner_defs = [
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3 = A00(B10-B00) -> C01 += M3, C11 += M3
        ([(0,0,1)], [(1,0,1),(0,0,-1)], [(0,1,1),(1,1,1)]),
        # M4 = A11(B01-B11) -> C00 += M4, C10 += M4
        ([(1,1,1)], [(0,1,1),(1,1,-1)], [(0,0,1),(1,0,1)]),
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    # Super Strassen for 4x4x4: split into 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # We apply Strassen at the block level:
    # M1' = (A00+A11)(B00+B11)
    # M2' = (A10+A11)B00
    # M3' = A00(B10-B00)
    # M4' = A11(B01-B11)
    # M5' = (A00+A01)B11
    # M6' = (A10-A00)(B00+B01)
    # M7' = (A01-A11)(B10+B11)
    #
    # Each Mj' is a 2x2 matrix product of 2x2 matrices, which we compute
    # using the inner Strassen (7 multiplications per 2x2x2).
    # Total: 7 * 7 = 49 multiplications.

    # For each super-level Strassen multiplication, we need to express
    # the 2x2 block coefficients in terms of the 4x4 matrix elements.

    # Block index to (row_offset, col_offset) in 4x4:
    # Block 0 = A00: rows 0-1, cols 0-1
    # Block 1 = A01: rows 0-1, cols 2-3
    # Block 2 = A10: rows 2-3, cols 0-1
    # Block 3 = A11: rows 2-3, cols 2-3

    def get_block_elements(block_idx, local_r, local_c, coeff):
        """Get global (row, col) from block index and local position."""
        if block_idx == 0:  # top-left
            return (local_r, local_c)
        elif block_idx == 1:  # top-right
            return (local_r, 2 + local_c)
        elif block_idx == 2:  # bottom-left
            return (2 + local_r, local_c)
        elif block_idx == 3:  # bottom-right
            return (2 + local_r, 2 + local_c)

    # Super-level Strassen: define the 7 multiplications at block level
    # Each: (A_block_coeffs, B_block_coeffs, C_block_contribs)
    # A_block_coeffs: list of (block_idx, coeff)
    # B_block_coeffs: list of (block_idx, coeff)
    # C_block_contribs: list of (block_idx, sign)

    super_defs = [
        # M1' = (A00+A11)(B00+B11) -> C00 += M1', C11 += M1'
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2' = (A10+A11)B00 -> C10 += M2', C11 -= M2'
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
        # M3' = A00(B10-B00) -> C01 += M3', C11 += M3'
        ([(0,1)], [(2,1),(0,-1)], [(1,1),(3,1)]),
        # M4' = A11(B01-B11) -> C00 += M4', C10 += M4'
        ([(3,1)], [(1,1),(3,-1)], [(0,1),(2,1)]),
        # M5' = (A00+A01)B11 -> C00 -= M5', C01 += M5'
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        # M6' = (A10-A00)(B00+B01) -> C11 += M6'
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7' = (A01-A11)(B10+B11) -> C00 += M7'
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

    for s_idx, (a_blocks, b_blocks, c_blocks) in enumerate(super_defs):
        for i_idx, (u_elems, v_elems, w_contribs) in enumerate(inner_defs):
            u_list = []
            v_list = []
            w_list = []

            # Build U elements: combine super-level and inner-level
            for (block_idx, super_coeff) in a_blocks:
                for (lr, lc, inner_coeff) in u_elems:
                    gr, gc = get_block_elements(block_idx, lr, lc, super_coeff)
                    u_list.append((gr, gc, super_coeff * inner_coeff))

            # Build V elements: combine super-level and inner-level
            for (block_idx, super_coeff) in b_blocks:
                for (lr, lc, inner_coeff) in v_elems:
                    gr, gc = get_block_elements(block_idx, lr, lc, super_coeff)
                    v_list.append((gr, gc, super_coeff * inner_coeff))

            # Build W elements: combine super-level and inner-level
            for (c_block_idx, super_sign) in c_blocks:
                for (lr, lc, inner_sign) in w_contribs:
                    gr, gc = get_block_elements(c_block_idx, lr, lc, super_sign)
                    w_list.append((gc, gr, super_sign * inner_sign))

            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1

    # Part 2: 4x4x1 (rank 16) - schoolbook
    # C[i,4] = sum_j A[i,j]*B[j,4]
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1

    # Verify the decomposition
    if time.time() - start_time < DEADLINE - 10:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        rng = np.random.RandomState(42)
        A_test = rng.rand(4, 4)
        B_test = rng.rand(4, 5)

        C_expected = A_test @ B_test

        C_computed = np.zeros((4, 5))
        for alpha in range(rank):
            a_part = np.dot(A_test.flatten(), U_arr[:, alpha])
            b_part = np.dot(B_test.flatten(), V_arr[:, alpha])
            C_computed += a_part * b_part * W_arr.reshape(5, 4).T

        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback: schoolbook rank 80
            rank = 80
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(4):
                for j in range(4):
                    for k in range(5):
                        U[i * 4 + j][col] = 1
                        V[j * 5 + k][col] = 1
                        W[k * 4 + i][col] = 1
                        col += 1

    return {"U": U, "V": V, "W": W}
