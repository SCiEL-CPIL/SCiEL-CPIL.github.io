"""Rewritten program: 4x4x5 matrix multiplication tensor decomposition, rank 65.
Strategy: Strassen-based rank-49 for the 4x4x4 block, plus rank-16 schoolbook for the 4x4x1 strip."""

import numpy as np

DIMENSIONS = [4, 4, 5]


def construct():
    n, m, p = 4, 4, 5
    rank = 65

    # U: (n*m, rank) = (16, 65)
    # V: (m*p, rank) = (20, 65)
    # W: (p*n, rank) = (20, 65)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    def add_term(col, u_entries, v_entries, w_entries):
        for i, j, c in u_entries:
            U[i * m + j, col] += c
        for j, k, c in v_entries:
            V[j * p + k, col] += c
        for k, i, c in w_entries:
            W[k * n + i, col] += c

    # --- Part 1: 4x4 * 4x4 via Strassen (7x7 = 49 terms) ---
    #
    # The 4x4 product is decomposed into 2x2 blocks, each multiplied via Strassen (7 terms).
    # Super-level: 7 Strassen terms for the 2x2 block product.
    # Inner-level: 7 Strassen terms for each 2x2 scalar block product.
    #
    # A (4x4) split into blocks:
    #   A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    #   A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    #
    # B (4x4, first 4 cols) split into blocks:
    #   B00 = B[0:2, 0:2], B01 = B[0:2, 2:4]
    #   B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    #
    # Super Strassen terms:
    #   M0 = (A00+A11)(B00+B11)
    #   M1 = (A10+A11)B00
    #   M2 = A00(B01-B11)
    #   M3 = A11(B10-B00)
    #   M4 = (A00+A01)B11
    #   M5 = (A10-A00)(B00+B01)
    #   M6 = (A01-A11)(B10+B11)
    #
    # Result blocks:
    #   C00 = M0 + M3 - M4 + M6
    #   C01 = M2 + M4
    #   C10 = M1 + M3
    #   C11 = M0 - M1 + M2 + M5

    # For each super term, define the A-block coefficients and B-block coefficients.
    # Each coefficient is a 4-tuple: (cA00, cA01, cA10, cA11) for A, same for B.
    super_defs = [
        # (A_blocks, B_blocks)
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),   # M0
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0])),   # M1
        (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1])),  # M2
        (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0])),  # M3
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),   # M4
        (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0])),  # M5
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1])),  # M6
    ]

    # Inner Strassen: for P (2x2) * Q (2x2), 7 terms.
    # Each term multiplies a linear combo of P elements by a linear combo of Q elements.
    # P elements indexed 0..3: P[0,0], P[0,1], P[1,0], P[1,1]
    # Q elements indexed 0..3: Q[0,0], Q[0,1], Q[1,0], Q[1,1]
    #
    # Term 0: (P00+P11)(Q00+Q11) -> contrib to C: [1,0,0,1]
    # Term 1: (P10+P11)Q00       -> contrib to C: [0,0,1,-1]
    # Term 2: P00(Q01-Q11)       -> contrib to C: [0,1,0,1]
    # Term 3: P11(Q10-Q00)       -> contrib to C: [1,0,1,0]
    # Term 4: (P00+P01)Q11       -> contrib to C: [-1,1,0,0]
    # Term 5: (P10-P00)(Q00+Q01) -> contrib to C: [0,0,0,1]
    # Term 6: (P01-P11)(Q10+Q11) -> contrib to C: [1,0,0,0]

    inner_defs = [
        # (P_elem_coeffs, Q_elem_coeffs, C_block_contrib)
        # C_block_contrib: [C00, C01, C10, C11]
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),   # t0
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, -1])),  # t1
        (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 1, 0, 1])),  # t2
        (np.array([0, 0, 0, 1]), np.array([0, -1, 1, 0]), np.array([1, 0, 1, 0])),  # t3
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0])),  # t4
        (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),  # t5
        (np.array([0, 1, -1, 0]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0])),  # t6
    ]

    # Super-to-C-block mapping:
    #   C00 = M0 + M3 - M4 + M6
    #   C01 = M2 + M4
    #   C10 = M1 + M3
    #   C11 = M0 - M1 + M2 + M5
    super_c_map = [
        np.array([1, 0, 0, 1]),   # M0 -> C00+1, C11+1
        np.array([0, 0, 1, -1]),  # M1 -> C10+1, C11-1
        np.array([0, 1, 0, 1]),   # M2 -> C01+1, C11+1
        np.array([1, 0, 1, 0]),   # M3 -> C00+1, C10+1
        np.array([-1, 1, 0, 0]),  # M4 -> C00-1, C01+1
        np.array([0, 0, 0, 1]),   # M5 -> C11+1
        np.array([1, 0, 0, 0]),   # M6 -> C00+1
    ]

    col = 0
    for s in range(7):
        a_blocks, b_blocks = super_defs[s]
        for t in range(7):
            p_coeffs, q_coeffs, s_contrib = inner_defs[t]
            c_block_contrib = super_c_map[s]

            u_entries = []
            v_entries = []
            w_entries = []

            # Build U entries from P elements
            for pe in range(4):
                pc = p_coeffs[pe]
                if pc == 0:
                    continue
                pr = pe // 2
                pc_col = pe % 2
                for b_idx in range(4):
                    bc = a_blocks[b_idx]
                    if bc == 0:
                        continue
                    coeff = pc * bc
                    if b_idx == 0:
                        gi, gj = pr, pc_col
                    elif b_idx == 1:
                        gi, gj = pr, 2 + pc_col
                    elif b_idx == 2:
                        gi, gj = 2 + pr, pc_col
                    else:
                        gi, gj = 2 + pr, 2 + pc_col
                    u_entries.append((gi, gj, coeff))

            # Build V entries from Q elements
            for qe in range(4):
                qc = q_coeffs[qe]
                if qc == 0:
                    continue
                qr = qe // 2
                qc_col = qe % 2
                for b_idx in range(4):
                    bc = b_blocks[b_idx]
                    if bc == 0:
                        continue
                    coeff = qc * bc
                    if b_idx == 0:
                        gj, gk = qr, qc_col
                    elif b_idx == 1:
                        gj, gk = qr, 2 + qc_col
                    elif b_idx == 2:
                        gj, gk = 2 + qr, qc_col
                    else:
                        gj, gk = 2 + qr, 2 + qc_col
                    v_entries.append((gj, gk, coeff))

            # Build W entries from C block contributions
            for sp in range(4):
                sc = s_contrib[sp]
                if sc == 0:
                    continue
                sr = sp // 2
                sc_col = sp % 2
                for cb in range(4):
                    cc = c_block_contrib[cb]
                    if cc == 0:
                        continue
                    total = sc * cc
                    if cb == 0:
                        gi, gk = sr, sc_col
                    elif cb == 1:
                        gi, gk = sr, 2 + sc_col
                    elif cb == 2:
                        gi, gk = 2 + sr, sc_col
                    else:
                        gi, gk = 2 + sr, 2 + sc_col
                    w_entries.append((gk, gi, total))

            add_term(col, u_entries, v_entries, w_entries)
            col += 1

    # --- Part 2: 4x4 * 4x1 (column 4 of B) via schoolbook, rank 16 ---
    # C[i, 4] = sum_j A[i,j] * B[j, 4]
    for i in range(4):
        for j in range(4):
            u_entries = [(i, j, 1.0)]
            v_entries = [(j, 4, 1.0)]
            w_entries = [(4, i, 1.0)]
            add_term(col, u_entries, v_entries, w_entries)
            col += 1

    return {"U": U, "V": V, "W": W}
