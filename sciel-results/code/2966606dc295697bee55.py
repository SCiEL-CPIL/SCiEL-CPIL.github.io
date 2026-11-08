"""Kronecker-embedded Strassen decomposition for 3x3 matrix multiplication.

Approach: treat the 3x3 matrices as 2x2 block matrices where the diagonal
blocks are 2x2 and the off-diagonal blocks are 1x1 scalars. Apply the rank-7
Strassen algorithm to the diagonal 2x2 block product (giving the 2x2 top-left
corner of C) and use direct multiplication for the remaining interactions.
Factors are shared across the block structure to minimise total rank.
"""

DIMENSIONS = [3, 3, 3]


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # ------------------------------------------------------------------
    # Index conventions
    #   A is flattened row-major:  A_idx(i, j) = i * m + j
    #   B is flattened row-major:  B_idx(j, k) = j * p + k
    #   C uses the OFFICIAL TRACE convention:  W_idx(i, k) = k * n + i
    #
    # Block structure (2x2 blocks of sizes 2,1):
    #   A = [[A11 (2x2), A12 (2x1)],      B = [[B11 (2x2), B12 (2x1)],
    #        [A21 (1x2), A22 (1x1)]]           [B21 (1x2), B22 (1x1)]]
    #   C = A*B = [[C11 (2x2), C12 (2x1)],
    #               [C21 (1x2), C22 (1x1)]]
    # ------------------------------------------------------------------

    def a_idx(i, j):
        return i * m + j

    def b_idx(j, k):
        return j * p + k

    def w_idx(i, k):
        # official trace convention W[k*n + i]
        return k * n + i

    # Rank-7 Strassen multipliers for the diagonal 2x2 blocks.
    # A11 = [[a00, a01], [a10, a11]]  (local coords)
    # B11 = [[b00, b01], [b10, b11]]  (local coords)
    # Each entry: (u_entries, v_entries, w_entries) where each *_entries is
    # a list of (index, coefficient) pairs. Indices refer to the local 2x2
    # block coordinates; we translate to global below.
    strassen_local = [
        # M1 = (a00 + a11)(b00 + b11)
        ([(0, 1.0), (3, 1.0)], [(0, 1.0), (3, 1.0)], [(0, 1.0), (3, 1.0)]),
        # M2 = (a10 + a11) b00
        ([(2, 1.0), (3, 1.0)], [(0, 1.0)], [(2, 1.0), (3, -1.0)]),
        # M3 = a00 (b01 - b11)
        ([(0, 1.0)], [(1, 1.0), (3, -1.0)], [(1, 1.0), (3, 1.0)]),
        # M4 = a11 (b10 - b00)
        ([(3, 1.0)], [(2, 1.0), (0, -1.0)], [(0, 1.0), (2, 1.0)]),
        # M5 = (a00 + a01) b11
        ([(0, 1.0), (1, 1.0)], [(3, 1.0)], [(0, -1.0), (1, 1.0)]),
        # M6 = (a10 - a00)(b00 + b01)
        ([(2, 1.0), (0, -1.0)], [(0, 1.0), (1, 1.0)], [(3, 1.0)]),
        # M7 = (a01 - a11)(b10 + b11)
        ([(1, 1.0), (3, -1.0)], [(2, 1.0), (3, 1.0)], [(0, 1.0)]),
    ]

    # Local 2x2 coordinate -> global A index. A11 occupies rows 0,1 cols 0,1.
    def a_local_to_global(li, lj):
        return a_idx(li, lj)

    # Local 2x2 coordinate -> global B index. B11 occupies rows 0,1 cols 0,1.
    def b_local_to_global(li, lj):
        return b_idx(li, lj)

    # Local 2x2 coordinate -> global W index. C11 occupies rows 0,1 cols 0,1.
    def w_local_to_global(li, lj):
        return w_idx(li, lj)

    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    def add_slice(r, u_entries, v_entries, w_entries):
        for idx, coeff in u_entries:
            U[idx][r] += coeff
        for idx, coeff in v_entries:
            V[idx][r] += coeff
        for idx, coeff in w_entries:
            W[idx][r] += coeff

    r = 0

    # ---- Group 1: Strassen on the diagonal 2x2 blocks (rank 7) ----
    for u_loc, v_loc, w_loc in strassen_local:
        u_g = [(a_local_to_global(i, c), cf) for (i, c), cf in
               [(e[0], e[1]) for e in u_loc]]
        # rebuild cleanly:
        u_g = [(a_local_to_global(i, j), cf) for (i, j), cf in
               [((e[0] // 2, e[0] % 2), e[1]) for e in u_loc]]
        v_g = [(b_local_to_global(i, j), cf) for (i, j), cf in
               [((e[0] // 2, e[0] % 2), e[1]) for e in v_loc]]
        w_g = [(w_local_to_global(i, j), cf) for (i, j), cf in
               [((e[0] // 2, e[0] % 2), e[1]) for e in w_loc]]
        add_slice(r, u_g, v_g, w_g)
        r += 1

    # ---- Group 2: A12 * B21  (2x1)(1x2) -> C12 (2x2)  [4 slices] ----
    # A12 = column 2 of A: a(0,2)=2, a(1,2)=5
    # B21 = row 2 of B: b(2,0)=6, b(2,1)=7
    # C12 = rows 0,1 col 2: C(0,2)=w(2,0)=6, C(1,2)=w(2,1)=7
    for (ai, bi, ci) in [(2, 6, 6), (2, 7, 7), (5, 6, 6 + 3), (5, 7, 7 + 3)]:
        # C(0,2) -> w_idx(0,2)=2*3+0=6 ; C(1,2)->w_idx(1,2)=2*3+1=7
        pass
    # explicit:
    add_slice(r, [(a_idx(0, 2), 1.0)], [(b_idx(2, 0), 1.0)], [(w_idx(0, 2), 1.0)]); r += 1
    add_slice(r, [(a_idx(0, 2), 1.0)], [(b_idx(2, 1), 1.0)], [(w_idx(1, 2), 1.0)]); r += 1
    add_slice(r, [(a_idx(1, 2), 1.0)], [(b_idx(2, 0), 1.0)], [(w_idx(0, 2), 1.0)]); r += 1
    add_slice(r, [(a_idx(1, 2), 1.0)], [(b_idx(2, 1), 1.0)], [(w_idx(1, 2), 1.0)]); r += 1

    # ---- Group 3: A11 * B12  (2x2)(2x1) -> C22 (2x1)  [4 slices] ----
    # B12 = column 2 of B rows 0,1: b(0,2)=2, b(1,2)=5
    # C22 = rows 0,1 col 2: w(0,2)=6, w(1,2)=7
    for (ai, bi, ci) in [
        (a_idx(0, 0), b_idx(0, 2), w_idx(0, 2)),
        (a_idx(0, 1), b_idx(1, 2), w_idx(0, 2)),
        (a_idx(1, 0), b_idx(0, 2), w_idx(1, 2)),
        (a_idx(1, 1), b_idx(1, 2), w_idx(1, 2)),
    ]:
        add_slice(r, [(ai, 1.0)], [(bi, 1.0)], [(ci, 1.0)])
        r += 1

    # ---- Group 4: A12 * B22  (2x1)(1x1) -> C22 add (2x1)  [2 slices] ----
    # A12: a(0,2)=2, a(1,2)=5 ; B22 = b(2,2)=8
    add_slice(r, [(a_idx(0, 2), 1.0)], [(b_idx(2, 2), 1.0)], [(w_idx(0, 2), 1.0)]); r += 1
    add_slice(r, [(a_idx(1, 2), 1.0)], [(b_idx(2, 2), 1.0)], [(w_idx(1, 2), 1.0)]); r += 1

    # ---- Group 5: A21 * B11  (1x2)(2x2) -> C21 (1x2)  [4 slices] ----
    # A21 = row 2 of A: a(2,0)=6, a(2,1)=7
    # B11: b(0,0)=0, b(1,0)=3, b(0,1)=1, b(1,1)=4
    # C21 = row 2 cols 0,1: w(0,2)=2, w(1,2)=5
    for (ai, bi, ci) in [
        (a_idx(2, 0), b_idx(0, 0), w_idx(0, 2)),
        (a_idx(2, 1), b_idx(1, 0), w_idx(0, 2)),
        (a_idx(2, 0), b_idx(0, 1), w_idx(1, 2)),
        (a_idx(2, 1), b_idx(1, 1), w_idx(1, 2)),
    ]:
        add_slice(r, [(ai, 1.0)], [(bi, 1.0)], [(ci, 1.0)])
        r += 1

    # ---- Group 6: A22 * B21  (1x1)(1x2) -> C21 add (1x2)  [2 slices] ----
    # A22 = a(2,2)=8 ; B21: b(2,0)=6, b(2,1)=7
    add_slice(r, [(a_idx(2, 2), 1.0)], [(b_idx(2, 0), 1.0)], [(w_idx(0, 2), 1.0)]); r += 1
    add_slice(r, [(a_idx(2, 2), 1.0)], [(b_idx(2, 1), 1.0)], [(w_idx(1, 2), 1.0)]); r += 1

    # ---- Group 7: A21 * B12  (1x2)(2x1) -> C22 add (1x1)  [2 slices] ----
    # A21: a(2,0)=6, a(2,1)=7 ; B12: b(0,2)=2, b(1,2)=5
    # C22 = w(2,2)=8
    add_slice(r, [(a_idx(2, 0), 1.0)], [(b_idx(0, 2), 1.0)], [(w_idx(2, 2), 1.0)]); r += 1
    add_slice(r, [(a_idx(2, 1), 1.0)], [(b_idx(1, 2), 1.0)], [(w_idx(2, 2), 1.0)]); r += 1

    # ---- Group 8: A22 * B22  (1x1)(1x1) -> C22 add (1x1)  [1 slice] ----
    add_slice(r, [(a_idx(2, 2), 1.0)], [(b_idx(2, 2), 1.0)], [(w_idx(2, 2), 1.0)]); r += 1

    # r should be 7+4+4+2+4+2+2+1 = 26
    assert r == rank, f"rank mismatch: {r} != {rank}"

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
