import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = 6, 6, 6
    block_size = 3
    n_blocks = n // block_size  # 2

    rank = 23 * 8  # 23 per (ib, jb, kb) triple, 2^3 = 8 triples
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    def u_idx(ib, kb, r, c):
        return (ib * block_size + r) * m + (kb * block_size + c)

    def v_idx(kb, jb, r, c):
        return (kb * block_size + r) * p + (jb * block_size + c)

    def w_idx(ib, jb, r, c):
        return (jb * block_size + c) * n + (ib * block_size + r)

    # Laderman's 23 terms for 3x3 matrix multiplication.
    # Each term is (u_indices, v_indices, w_indices) where each is a list of (r, c) with coefficients.
    # For each term, the contribution to C is: sum of u_coeff * v_coeff * w_coeff at each position.
    # The convention: U[i, col] * V[j, col] * W[k, col] contributes to T[i, j, k].

    # I'll use the standard Laderman decomposition.
    # The 23 products M1..M23 and the C entries are:
    # C00 = M1 - M8 + M9 - M14 + M17 - M18 + M19
    # C01 = M4 + M8 + M14 - M17
    # C02 = M5 - M9 + M15 + M19 - M22
    # C10 = M6 - M10 + M15 - M20
    # C11 = M2 + M10 + M11 - M14 + M17 - M18 + M20 - M21
    # C12 = M7 - M11 + M16 - M18 + M21
    # C20 = M6 - M7 + M16 - M19
    # C21 = M12 + M11 - M16 + M22
    # C22 = M3 + M12 + M13 - M15 + M18 - M21 + M23

    # The 23 products:
    # M1 = A00*B00
    # M2 = A11*B11
    # M3 = A22*B22
    # M4 = (A01+A02)*B10
    # M5 = A02*(B20-B10)
    # M6 = (A10+A20)*B01
    # M7 = A20*(B02-B01)
    # M8 = (A00-A11)*(B01+B02)
    # M9 = (A00-A22)*(B10+B12)
    # M10 = (A10-A11)*(B02+B12)
    # M11 = (A20-A22)*(B01+B11)
    # M12 = (A01-A21)*(B12+B22)
    # M13 = (A12-A11)*(B20+B22)
    # M14 = (A00-A01)*(B20+B10+B11+B12+B21+B22)
    # M15 = (A11-A12)*(B00+B01+B02+B10+B12+B20)
    # M16 = (A22-A20)*(B00+B01+B02+B10+B11+B20+B21)
    # M17 = (A01-A11)*(B10+B20+B21+B22)
    # M18 = (A12-A22)*(B00+B01+B02+B12)
    # M19 = (A02-A20)*(B00+B10+B11+B21)
    # M20 = (A10-A00)*(B02+B11+B22)
    # M21 = (A21-A11)*(B00+B12+B22)
    # M22 = (A02-A22)*(B01+B10+B20)
    # M23 = (A10-A00)*(B01+B11+B21)

    # For each term, I need to determine the U, V, W indices and coefficients.
    # U indices: (row, col) in the A block, with coefficient
    # V indices: (row, col) in the B block, with coefficient
    # W indices: (row, col) in the C block, with coefficient

    # Let me define each term as (u_list, v_list, w_list) where each list is [(r, c, coeff), ...]

    laderman_terms = [
        # M1 = A00*B00
        ([(0, 0, 1)], [(0, 0, 1)], [(0, 0, 1)]),
        # M2 = A11*B11
        ([(1, 1, 1)], [(1, 1, 1)], [(1, 1, 1)]),
        # M3 = A22*B22
        ([(2, 2, 1)], [(2, 2, 1)], [(2, 2, 1)]),
        # M4 = (A01+A02)*B10
        ([(0, 1, 1), (0, 2, 1)], [(1, 0, 1)], [(0, 1, 1)]),
        # M5 = A02*(B20-B10)
        ([(0, 2, 1)], [(2, 0, 1), (1, 0, -1)], [(0, 2, 1)]),
        # M6 = (A10+A20)*B01
        ([(1, 0, 1), (2, 0, 1)], [(0, 1, 1)], [(1, 0, 1)]),
        # M7 = A20*(B02-B01)
        ([(2, 0, 1)], [(0, 2, 1), (0, 1, -1)], [(2, 0, 1)]),
        # M8 = (A00-A11)*(B01+B02)
        ([(0, 0, 1), (1, 1, -1)], [(0, 1, 1), (0, 2, 1)], [(0, 1, 1), (0, 2, 1)]),
        # M9 = (A00-A22)*(B10+B12)
        ([(0, 0, 1), (2, 2, -1)], [(1, 0, 1), (1, 2, 1)], [(0, 1, 1), (0, 2, 1)]),
        # M10 = (A10-A11)*(B02+B12)
        ([(1, 0, 1), (1, 1, -1)], [(0, 2, 1), (1, 2, 1)], [(1, 2, 1)]),
        # M11 = (A20-A22)*(B01+B11)
        ([(2, 0, 1), (2, 2, -1)], [(0, 1, 1), (1, 1, 1)], [(2, 1, 1)]),
        # M12 = (A01-A21)*(B12+B22)
        ([(0, 1, 1), (2, 1, -1)], [(1, 2, 1), (2, 2, 1)], [(1, 2, 1)]),
        # M13 = (A12-A11)*(B20+B22)
        ([(1, 2, 1), (1, 1, -1)], [(2, 0, 1), (2, 2, 1)], [(2, 0, 1)]),
        # M14 = (A00-A01)*(B20+B10+B11+B12+B21+B22)
        ([(0, 0, 1), (0, 1, -1)], [(2, 0, 1), (1, 0, 1), (1, 1, 1), (1, 2, 1), (2, 1, 1), (2, 2, 1)], [(0, 2, 1)]),
        # M15 = (A11-A12)*(B00+B01+B02+B10+B12+B20)
        ([(1, 1, 1), (1, 2, -1)], [(0, 0, 1), (0, 1, 1), (0, 2, 1), (1, 0, 1), (1, 2, 1), (2, 0, 1)], [(1, 2, 1)]),
        # M16 = (A22-A20)*(B00+B01+B02+B10+B11+B20+B21)
        ([(2, 2, 1), (2, 0, -1)], [(0, 0, 1), (0, 1, 1), (0, 2, 1), (1, 0, 1), (1, 1, 1), (2, 0, 1), (2, 1, 1)], [(2, 0, 1)]),
        # M17 = (A01-A11)*(B10+B20+B21+B22)
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (2, 0, 1), (2, 1, 1), (2, 2, 1)], [(0, 1, 1)]),
        # M18 = (A12-A22)*(B00+B01+B02+B12)
        ([(1, 2, 1), (2, 2, -1)], [(0, 0, 1), (0, 1, 1), (0, 2, 1), (1, 2, 1)], [(2, 2, 1)]),
        # M19 = (A02-A20)*(B00+B10+B11+B21)
        ([(0, 2, 1), (2, 0, -1)], [(0, 0, 1), (1, 0, 1), (1, 1, 1), (2, 1, 1)], [(2, 1, 1)]),
        # M20 = (A10-A00)*(B02+B11+B22)
        ([(1, 0, 1), (0, 0, -1)], [(0, 2, 1), (1, 1, 1), (2, 2, 1)], [(1, 0, 1)]),
        # M21 = (A21-A11)*(B00+B12+B22)
        ([(2, 1, 1), (1, 1, -1)], [(0, 0, 1), (1, 2, 1), (2, 2, 1)], [(2, 0, 1)]),
        # M22 = (A02-A22)*(B01+B10+B20)
        ([(0, 2, 1), (2, 2, -1)], [(0, 1, 1), (1, 0, 1), (2, 0, 1)], [(0, 2, 1)]),
        # M23 = (A10-A00)*(B01+B11+B21)
        ([(1, 0, 1), (0, 0, -1)], [(0, 1, 1), (1, 1, 1), (2, 1, 1)], [(1, 0, 1)]),
    ]

    # Now I need to determine the W coefficients for each term.
    # The W coefficient for a term at position (r, c) in the C block is the coefficient
    # of that term in the formula for C[r, c].
    # From the C formulas above, I can extract the coefficient of each M term in each C entry.

    # Let me create a dictionary: for each term index (0-22), which C entries it appears in and with what coefficient.
    # C00: M1:+1, M8:-1, M9:+1, M14:-1, M17:+1, M18:-1, M19:+1
    # C01: M4:+1, M8:+1, M14:+1, M17:-1
    # C02: M5:+1, M9:-1, M15:+1, M19:+1, M22:-1
    # C10: M6:+1, M10:-1, M15:+1, M20:-1
    # C11: M2:+1, M10:+1, M11:+1, M14:-1, M17:+1, M18:-1, M20:+1, M21:-1
    # C12: M7:+1, M11:-1, M16:+1, M18:-1, M21:+1
    # C20: M6:+1, M7:-1, M16:+1, M19:-1
    # C21: M12:+1, M11:+1, M16:-1, M22:+1
    # C22: M3:+1, M12:+1, M13:+1, M15:-1, M18:+1, M21:-1, M23:+1

    # For each term, the W entries are at the C positions where that term appears, with the corresponding coefficient.
    # But wait, the W matrix in the tensor decomposition has entries W[k, col] where k indexes the C output.
    # The C output index is (row, col) in the C block, which maps to w_idx(ib, jb, r, c).
    # So for each term, I need to add W[w_idx(ib, jb, r, c), col] += coeff for each (r, c, coeff) in the W list.

    # But the W list in my laderman_terms above only has one entry per term, which is wrong.
    # I need to fix this: the W list should contain all C positions where the term appears, with the correct coefficient.

    # Let me redefine the terms with proper W lists.

    # Actually, let me just build the W entries from the C formulas directly.
    # For term t (0-indexed), the W entries are:
    # For each C position (r, c) where term t appears with coefficient coeff, add W[w_idx(ib, jb, r, c), col] += coeff

    # Let me create a mapping from term index to list of (r, c, coeff) for W.
    w_coeffs = [
        # Term 0 (M1): C00:+1
        [(0, 0, 1)],
        # Term 1 (M2): C11:+1
        [(1, 1, 1)],
        # Term 2 (M3): C22:+1
        [(2, 2, 1)],
        # Term 3 (M4): C01:+1
        [(0, 1, 1)],
        # Term 4 (M5): C02:+1
        [(0, 2, 1)],
        # Term 5 (M6): C10:+1, C20:+1
        [(1, 0, 1), (2, 0, 1)],
        # Term 6 (M7): C12:+1, C20:-1
        [(1, 2, 1), (2, 0, -1)],
        # Term 7 (M8): C00:-1, C01:+1
        [(0, 0, -1), (0, 1, 1)],
        # Term 8 (M9): C00:+1, C02:-1
        [(0, 0, 1), (0, 2, -1)],
        # Term 9 (M10): C10:-1, C11:+1
        [(1, 0, -1), (1, 1, 1)],
        # Term 10 (M11): C11:+1, C12:-1, C21:+1
        [(1, 1, 1), (1, 2, -1), (2, 1, 1)],
        # Term 11 (M12): C21:+1, C22:+1
        [(2, 1, 1), (2, 2, 1)],
        # Term 12 (M13): C22:+1
        [(2, 2, 1)],
        # Term 13 (M14): C00:-1, C01:+1, C11:-1
        [(0, 0, -1), (0, 1, 1), (1, 1, -1)],
        # Term 14 (M15): C02:+1, C10:+1, C22:-1
        [(0, 2, 1), (1, 0, 1), (2, 2, -1)],
        # Term 15 (M16): C12:+1, C20:+1, C21:-1
        [(1, 2, 1), (2, 0, 1), (2, 1, -1)],
        # Term 16 (M17): C00:+1, C01:-1, C11:+1
        [(0, 0, 1), (0, 1, -1), (1, 1, 1)],
        # Term 17 (M18): C00:-1, C11:-1, C12:-1, C22:+1
        [(0, 0, -1), (1, 1, -1), (1, 2, -1), (2, 2, 1)],
        # Term 18 (M19): C00:+1, C02:+1, C20:-1
        [(0, 0, 1), (0, 2, 1), (2, 0, -1)],
        # Term 19 (M20): C10:-1, C11:+1
        [(1, 0, -1), (1, 1, 1)],
        # Term 20 (M21): C11:-1, C12:+1, C22:-1
        [(1, 1, -1), (1, 2, 1), (2, 2, -1)],
        # Term 21 (M22): C02:-1, C21:+1
        [(0, 2, -1), (2, 1, 1)],
        # Term 22 (M23): C22:+1
        [(2, 2, 1)],
    ]

    # Now build U, V, W
    col = 0
    for ib in range(n_blocks):
        for jb in range(n_blocks):
            for kb in range(n_blocks):
                for term_idx in range(23):
                    u_list, v_list, _ = laderman_terms[term_idx]
                    w_list = w_coeffs[term_idx]

                    for (r, c, coeff) in u_list:
                        U[u_idx(ib, kb, r, c), col] += coeff
                    for (r, c, coeff) in v_list:
                        V[v_idx(kb, jb, r, c), col] += coeff
                    for (r, c, coeff) in w_list:
                        W[w_idx(ib, jb, r, c), col] += coeff
                    col += 1

    # Verify the decomposition
    if time.time() < deadline - 10:
        T_computed = np.einsum('ir,jr,kr->ijk', U, V, W)

        T_true = np.zeros((n*m, m*p, p*n), dtype=int)
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    T_true[a, b, c] = 1

        if not np.array_equal(T_computed, T_true):
            # Fall back to Strassen 2x2 blocks
            rank2 = 7 * 27
            U2 = np.zeros((n * m, rank2), dtype=int)
            V2 = np.zeros((m * p, rank2), dtype=int)
            W2 = np.zeros((p * n, rank2), dtype=int)

            block_size2 = 2
            n_blocks2 = n // block_size2

            def u_idx2(ib, kb, r, c):
                return (ib * block_size2 + r) * m + (kb * block_size2 + c)

            def v_idx2(kb, jb, r, c):
                return (kb * block_size2 + r) * p + (jb * block_size2 + c)

            def w_idx2(ib, jb, r, c):
                return (jb * block_size2 + c) * n + (ib * block_size2 + r)

            col2 = 0
            for ib in range(n_blocks2):
                for jb in range(n_blocks2):
                    for kb in range(n_blocks2):
                        # M1 = (A00+A11)(B00+B11)
                        U2[u_idx2(ib, kb, 0, 0), col2] += 1
                        U2[u_idx2(ib, kb, 1, 1), col2] += 1
                        V2[v_idx2(kb, jb, 0, 0), col2] += 1
                        V2[v_idx2(kb, jb, 1, 1), col2] += 1
                        W2[w_idx2(ib, jb, 0, 0), col2] += 1
                        W2[w_idx2(ib, jb, 1, 1), col2] += 1
                        col2 += 1
                        # M2 = (A10+A11)B00
                        U2[u_idx2(ib, kb, 1, 0), col2] += 1
                        U2[u_idx2(ib, kb, 1, 1), col2] += 1
                        V2[v_idx2(kb, jb, 0, 0), col2] += 1
                        W2[w_idx2(ib, jb, 1, 0), col2] += 1
                        W2[w_idx2(ib, jb, 1, 1), col2] -= 1
                        col2 += 1
                        # M3 = A00(B11-B10)
                        U2[u_idx2(ib, kb, 0, 0), col2] += 1
                        V2[v_idx2(kb, jb, 0, 1), col2] += 1
                        V2[v_idx2(kb, jb, 1, 1), col2] -= 1
                        W2[w_idx2(ib, jb, 0, 1), col2] += 1
                        W2[w_idx2(ib, jb, 1, 1), col2] += 1
                        col2 += 1
                        # M4 = A11(B10-B00)
                        U2[u_idx2(ib, kb, 1, 1), col2] += 1
                        V2[v_idx2(kb, jb, 1, 0), col2] += 1
                        V2[v_idx2(kb, jb, 0, 0), col2] -= 1
                        W2[w_idx2(ib, jb, 0, 0), col2] += 1
                        W2[w_idx2(ib, jb, 1, 0), col2] += 1
                        col2 += 1
                        # M5 = (A00+A01)B11
                        U2[u_idx2(ib, kb, 0, 0), col2] += 1
                        U2[u_idx2(ib, kb, 0, 1), col2] += 1
                        V2[v_idx2(kb, jb, 1, 1), col2] += 1
                        W2[w_idx2(ib, jb, 0, 0), col2] -= 1
                        W2[w_idx2(ib, jb, 0, 1), col2] += 1
                        col2 += 1
                        # M6 = (A10-A00)(B00+B01)
                        U2[u_idx2(ib, kb, 1, 0), col2] += 1
                        U2[u_idx2(ib, kb, 0, 0), col2] -= 1
                        V2[v_idx2(kb, jb, 0, 0), col2] += 1
                        V2[v_idx2(kb, jb, 0, 1), col2] += 1
                        W2[w_idx2(ib, jb, 1, 1), col2] += 1
                        col2 += 1
                        # M7 = (A01-A11)(B10+B11)
                        U2[u_idx2(ib, kb, 0, 1), col2] += 1
                        U2[u_idx2(ib, kb, 1, 1), col2] -= 1
                        V2[v_idx2(kb, jb, 1, 0), col2] += 1
                        V2[v_idx2(kb, jb, 1, 1), col2] += 1
                        W2[w_idx2(ib, jb, 0, 0), col2] += 1
                        col2 += 1

            return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
