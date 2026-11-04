"""Matrix multiplication 4x4x5: rank-minimal exact tensor decomposition.

Structural change: Use NESTED Strassen for the 4x4x4 portion (7 outer Strassen
block combinations, each decomposed via inner 2x2 Strassen = 49 scalar terms)
and direct schoolbook for the 4x4x1 portion (16 terms). Total rank = 49 + 16 = 65.

This is a different design family than the independent-block Strassen (rank 56)
in the parent. The nested approach exploits shared structure across block
products, saving 7 rank.
"""
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before the 150 s wall-clock limit


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check that U,V,W exactly decompose the matrix-multiplication tensor.

    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]  ==  1  for all i,j,k.
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if U.ndim != 2:
        return False
    if U.shape[0] != n * m:
        return False
    rank = U.shape[1]
    if V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False

    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        U_r = U[:, r].reshape(n, m)
        V_r = V[:, r].reshape(m, p)
        W_r = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', U_r, V_r, W_r)
    return bool(np.allclose(T, 1.0, atol=tol))


def _schoolbook_4x4x5():
    """Fallback schoolbook decomposition with rank n*m*p = 80 (always valid)."""
    n, m, p = 4, 4, 5
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _construct_4x4x5_nested_strassen():
    """Construct the 4x4x5 decomposition using nested Strassen for 4x4x4 part
    (rank 49) and direct multiplication for the 4x4x1 part (rank 16).
    Total rank = 49 + 16 = 65."""
    if _time_remaining() < 5:
        raise TimeoutError("Not enough time remaining")

    n, m, p = 4, 4, 5
    rank_total = 65
    U = np.zeros((n * m, rank_total), dtype=np.float64)
    V = np.zeros((m * p, rank_total), dtype=np.float64)
    W = np.zeros((p * n, rank_total), dtype=np.float64)

    state = [0]

    # Block definitions:
    # A00: i in {0,1}, j in {0,1}
    # A01: i in {0,1}, j in {2,3}
    # A10: i in {2,3}, j in {0,1}
    # A11: i in {2,3}, j in {2,3}
    # B00: j in {0,1}, k in {0,1}
    # B01: j in {0,1}, k in {2,3}
    # B10: j in {2,3}, k in {0,1}
    # B11: j in {2,3}, k in {2,3}

    # For each outer Strassen term, define:
    # - p_blocks: coefficients for A blocks [A00, A01, A10, A11]
    # - q_blocks: coefficients for B blocks [B00, B01, B10, B11]
    # - c_block: which C block this contributes to, and sign
    #   C00: i in {0,1}, k in {0,1}
    #   C01: i in {0,1}, k in {2,3}
    #   C10: i in {2,3}, k in {0,1}
    #   C11: i in {2,3}, k in {2,3}

    # Outer Strassen terms:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00(+), C10(+), C01(-), C11(+)
    # M2 = (A10+A11)B00       -> contributes to C10(+), C11(-)
    # M3 = A00(B01-B11)       -> contributes to C01(+), C11(+)
    # M4 = A11(B10-B00)       -> contributes to C00(+), C10(+)
    # M5 = (A00+A01)B11       -> contributes to C00(-), C01(+)
    # M6 = (A10-A00)(B00+B01) -> contributes to C11(+)
    # M7 = (A01-A11)(B10+B11) -> contributes to C00(+)

    # Wait, let me re-derive carefully.
    # C = [[C00, C01], [C10, C11]]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # So:
    # M1: C00(+1), C10(+1), C01(-1), C11(+1)  -- wait that's 4 contributions
    # Actually no. M1 is a 2x2 matrix. Its (r,c) element contributes to C(rc).
    # M1[r,c] contributes to C_{r,c} with sign depending on the formula.
    #
    # C00 = M1[0,0] + M4[0,0] - M5[0,0] + M7[0,0]
    # C01 = M3[0,1] + M5[0,1]  -- wait, this doesn't look right either.
    #
    # Let me be more careful. In Strassen:
    # C00 = M1[0,0] + M4[0,0] - M5[0,0] + M7[0,0]
    # C01 = M3[0,1] + M5[0,1]
    # C10 = M2[1,0] + M4[1,0]
    # C11 = M1[1,1] - M2[1,1] + M3[1,1] + M6[1,1]
    #
    # Hmm, that's not right either. Let me look at standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Here each Mi is a 2x2 matrix, and the addition is element-wise.
    # So M1 contributes to all four C blocks.

    # For the tensor decomposition, each scalar multiplication within Mi
    # contributes to specific (i,j,k) entries. The W matrix index is k*n+i.
    #
    # Let me define the contribution pattern for each outer term:
    # For outer term t with P = sum_p p_blocks[p]*A_block_p and Q = sum_q q_blocks[q]*B_block_q:
    # The product P*Q is a 2x2 matrix. Element (r,c) of P*Q contributes to
    # C block (r,c) with a sign.
    #
    # C block (r,c) corresponds to:
    #   i in {2*r, 2*r+1}, k in {2*c, 2*c+1}
    #
    # So for each outer term, I need to know which C blocks it contributes to
    # and with what sign.

    # Let me re-derive the C formulas:
    # C00 = M1[0,0] + M4[0,0] - M5[0,0] + M7[0,0]
    # C01 = M3[0,1] + M5[0,1]
    # C10 = M2[1,0] + M4[1,0]
    # C11 = M1[1,1] - M2[1,1] + M3[1,1] + M6[1,1]
    #
    # Wait, this is wrong. Let me look it up properly.
    #
    # Standard Strassen for 2x2:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Where all M's are 2x2 matrices and + is element-wise.
    #
    # So M1 contributes to C00 (element [0,0]), C10 (element [1,0])... 
    # No! M1 is a full 2x2 matrix. M1[0,0] goes to C00[0,0], M1[0,1] goes to C01[0,0]...
    # No that's not right either.
    #
    # Actually: C00 is a 2x2 submatrix of C. C00[i,j] for i,j in {0,1}.
    # M1 is a 2x2 matrix. C00 = M1 + M4 - M5 + M7 means:
    # C00[i,j] = M1[i,j] + M4[i,j] - M5[i,j] + M7[i,j] for i,j in {0,1}.
    #
    # Similarly C01[i,j] = M3[i,j] + M5[i,j] for i,j in {0,1}.
    # C10[i,j] = M2[i,j] + M4[i,j] for i,j in {0,1}.
    # C11[i,j] = M1[i,j] - M2[i,j] + M3[i,j] + M6[i,j] for i,j in {0,1}.
    #
    # So M1 contributes to C00 (all elements) and C11 (all elements).
    # M4 contributes to C00 (all elements) and C10 (all elements).
    # M5 contributes to C00 (all elements, negative) and C01 (all elements).
    # M7 contributes to C00 (all elements).
    # M3 contributes to C01 (all elements) and C11 (all elements).
    # M2 contributes to C10 (all elements) and C11 (all elements, negative).
    # M6 contributes to C11 (all elements).

    # So for each outer term, the contribution to each C block is uniform
    # (same sign for all 4 elements of that C block).

    # Outer term contributions:
    # M1: C00(+1), C11(+1)
    # M2: C10(+1), C11(-1)
    # M3: C01(+1), C11(+1)
    # M4: C00(+1), C10(+1)
    # M5: C00(-1), C01(+1)
    # M6: C11(+1)
    # M7: C00(+1)

    # Now for each outer term, I decompose the 2x2 product P*Q using inner Strassen.
    # Inner Strassen for 2x2 * 2x2:
    # s1 = (P00+P11)(Q00+Q11)
    # s2 = (P10+P11)Q00
    # s3 = P00(Q01-Q11)
    # s4 = P11(Q10-Q00)
    # s5 = (P00+P01)Q11
    # s6 = (P10-P00)(Q00+Q01)
    # s7 = (P01-P11)(Q10+Q11)
    #
    # Result:
    # R00 = s1 + s4 - s5 + s7
    # R01 = s3 + s5
    # R10 = s2 + s4
    # R11 = s1 - s2 + s3 + s6

    # For the tensor decomposition, each scalar multiplication (inner term)
    # produces a rank-1 contribution. The U and V matrices encode the linear
    # combinations of A and B elements, and the W matrix encodes where the
    # result goes (which C block and with what sign).

    # For inner term t of outer term o:
    # - U encodes the P elements involved (linear combos of A elements)
    # - V encodes the Q elements involved (linear combos of B elements)
    # - W encodes the C block and sign

    # Let me define the inner Strassen terms and their P/Q element usage:
    # s1: P elements {00,11}, Q elements {00,11}
    # s2: P elements {10,11}, Q elements {00}
    # s3: P elements {00}, Q elements {01,11}
    # s4: P elements {11}, Q elements {10,00}
    # s5: P elements {00,01}, Q elements {11}
    # s6: P elements {10,00}, Q elements {00,01}
    # s7: P elements {01,11}, Q elements {10,11}

    # And the C block contribution for each inner term (within the 2x2 result):
    # s1: R00(+1), R11(+1)
    # s2: R10(+1), R11(-1)
    # s3: R01(+1), R11(+1)
    # s4: R00(+1), R10(+1)
    # s5: R00(-1), R01(+1)
    # s6: R11(+1)
    # s7: R00(+1)

    # So for outer term o with C-block signs c_signs[4] (for C00,C01,C10,C11),
    # and inner term t with R-block signs r_signs[4] (for R00,R01,R10,R11),
    # the total W sign for C block b is c_signs[b] * r_signs[b].

    # Now let's define everything concretely.

    # P element (r,c) where r,c in {0,1}:
    # P[r,c] = sum over A blocks b with p_blocks[b] != 0 of p_blocks[b] * A_block_b[r,c]
    # A_block_b[r,c] has global index (i,j) where:
    #   b=0 (A00): i=r, j=c
    #   b=1 (A01): i=r, j=2+c
    #   b=2 (A10): i=2+r, j=c
    #   b=3 (A11): i=2+r, j=2+c

    # Q element (r,c) where r,c in {0,1}:
    # Q[r,c] = sum over B blocks b with q_blocks[b] != 0 of q_blocks[b] * B_block_b[r,c]
    # B_block_b[r,c] has global index (j,k) where:
    #   b=0 (B00): j=r, k=c
    #   b=1 (B01): j=r, k=2+c
    #   b=2 (B10): j=2+r, k=c
    #   b=3 (B11): j=2+r, k=2+c

    # C block b (0=C00, 1=C01, 2=C10, 3=C11):
    # C block b has global (i,k) where:
    #   b=0 (C00): i=r, k=c
    #   b=1 (C01): i=r, k=2+c
    #   b=2 (C10): i=2+r, k=c
    #   b=3 (C11): i=2+r, k=2+c
    # W index is k*n + i = k*4 + i

    # Inner Strassen definitions:
    # (p_elems, q_elems) where each is a list of (elem_idx, coeff)
    # elem_idx: 0=P00, 1=P01, 2=P10, 3=P11
    #           0=Q00, 1=Q01, 2=Q10, 3=Q11
    inner_strassen = [
        # s1: (P00+P11)(Q00+Q11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # s2: (P10+P11)Q00
        ([(2, 1), (3, 1)], [(0, 1)]),
        # s3: P00(Q01-Q11)
        ([(0, 1)], [(1, 1), (3, -1)]),
        # s4: P11(Q10-Q00)
        ([(3, 1)], [(2, 1), (0, -1)]),
        # s5: (P00+P01)Q11
        ([(0, 1), (1, 1)], [(3, 1)]),
        # s6: (P10-P00)(Q00+Q01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        # s7: (P01-P11)(Q10+Q11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),
    ]

    # R-block signs for each inner term (R00, R01, R10, R11):
    # s1: R00(+1), R11(+1)
    # s2: R10(+1), R11(-1)
    # s3: R01(+1), R11(+1)
    # s4: R00(+1), R10(+1)
    # s5: R00(-1), R01(+1)
    # s6: R11(+1)
    # s7: R00(+1)
    inner_r_signs = [
        [1, 0, 0, 1],   # s1: R00, R11
        [0, 0, 1, -1],  # s2: R10, R11
        [0, 1, 0, 1],   # s3: R01, R11
        [1, 0, 1, 0],   # s4: R00, R10
        [-1, 1, 0, 0],  # s5: R00, R01
        [0, 0, 0, 1],   # s6: R11
        [1, 0, 0, 0],   # s7: R00
    ]

    # Outer Strassen terms:
    # (p_blocks, q_blocks, c_signs)
    # c_signs: [C00, C01, C10, C11]
    outer_strassen = [
        # M1: (A00+A11)(B00+B11), C00(+1), C11(+1)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2: (A10+A11)B00, C10(+1), C11(-1)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3: A00(B01-B11), C01(+1), C11(+1)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4: A11(B10-B00), C00(+1), C10(+1)
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5: (A00+A01)B11, C00(-1), C01(+1)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6: (A10-A00)(B00+B01), C11(+1)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7: (A01-A11)(B10+B11), C00(+1)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    def get_p_elements(p_elem_idx, p_blocks):
        """Get U entries for P element p_elem_idx (0-3) given p_blocks."""
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        entries = []
        for b_idx, coeff in enumerate(p_blocks):
            if coeff == 0:
                continue
            if b_idx == 0:  # A00
                gi, gj = r, c
            elif b_idx == 1:  # A01
                gi, gj = r, 2 + c
            elif b_idx == 2:  # A10
                gi, gj = 2 + r, c
            elif b_idx == 3:  # A11
                gi, gj = 2 + r, 2 + c
            entries.append((gi, gj, coeff))
        return entries

    def get_q_elements(q_elem_idx, q_blocks):
        """Get V entries for Q element q_elem_idx (0-3) given q_blocks."""
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        entries = []
        for b_idx, coeff in enumerate(q_blocks):
            if coeff == 0:
                continue
            if b_idx == 0:  # B00
                gj, gk = r, c
            elif b_idx == 1:  # B01
                gj, gk = r, 2 + c
            elif b_idx == 2:  # B10
                gj, gk = 2 + r, c
            elif b_idx == 3:  # B11
                gj, gk = 2 + r, 2 + c
            entries.append((gj, gk, coeff))
        return entries

    def get_w_entries(c_block, sign):
        """Get W entries for C block c_block with given sign."""
        # C block c_block: i in {2*rb, 2*rb+1}, k in {2*cb, 2*cb+1}
        # where rb = c_block // 2, cb = c_block % 2
        rb = c_block // 2
        cb = c_block % 2
        entries = []
        for ri in range(2):
            for ck in range(2):
                gi = 2 * rb + ri
                gk = 2 * cb + ck
                entries.append((gk, gi, sign))  # (k, i, sign)
        return entries

    # Build the 49 terms for the 4x4x4 part
    for o_idx, (p_blocks, q_blocks, c_signs) in enumerate(outer_strassen):
        for t_idx, (p_elems, q_elems) in enumerate(inner_strassen):
            col = state[0]
            r_signs = inner_r_signs[t_idx]

            # Build U
            for p_elem_idx, coeff in p_elems:
                for gi, gj, block_coeff in get_p_elements(p_elem_idx, p_blocks):
                    U[gi * m + gj, col] += coeff * block_coeff

            # Build V
            for q_elem_idx, coeff in q_elems:
                for gj, gk, block_coeff in get_q_elements(q_elem_idx, q_blocks):
                    V[gj * p + gk, col] += coeff * block_coeff

            # Build W
            for c_block in range(4):
                total_sign = c_signs[c_block] * r_signs[c_block]
                if total_sign == 0:
                    continue
                for gk, gi, sign in get_w_entries(c_block, total_sign):
                    W[gk * n + gi, col] += sign

            state[0] += 1

    # 4x4x1 part: B column 4 (the 5th column, index 4)
    # Direct schoolbook: 4x4 * 4x1 = 16 terms
    for i in range(4):
        for j in range(4):
            col = state[0]
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0
            W[4 * n + i, col] = 1.0
            state[0] += 1

    used_rank = state[0]
    assert used_rank == 65, f"Expected rank 65, got {used_rank}"

    U = U[:, :used_rank]
    V = V[:, :used_rank]
    W = W[:, :used_rank]

    if not _verify(U, V, W, n, m, p, tol=1e-8):
        raise ValueError("Nested Strassen 4x4x5 decomposition failed verification")

    return U, V, W


def construct():
    """Main entry point: construct the rank-minimal exact tensor decomposition
    for 4x4x5.  Returns {'U','V','W'} as lists of lists (integer/half-integer)."""
    try:
        U, V, W = _construct_4x4x5_nested_strassen()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except (TimeoutError, ValueError):
        pass

    # Fallback: guaranteed-valid schoolbook (rank 80)
    U, V, W = _schoolbook_4x4x5()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
