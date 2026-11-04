"""Rewritten candidate for matrix multiplication tensor rank minimisation (3x5x6)."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Kronecker products of known optimal tensors:
    #   2x2x2 via Strassen (rank 7)
    #   1x2x2 via schoolbook (rank 4)  [since no known rank-3 for 1x2x2 with integer coefficients]
    # 3x5x6 = (2+1) x (2+3) x (2+4) = (2+1)x2x(2+4) + (2+1)x3x(2+4)
    #         = 2x2x2 + 2x2x4 + 1x2x2 + 1x2x4 + 2x3x2 + 2x3x4 + 1x3x2 + 1x3x4
    # We split as: (2+1) x (2+3) x (2+4)
    # Ranks:
    #   2x2x2 : 7
    #   2x2x4 : 2 * (2x2x2) = 14  (split p into 2+2, each 2x2x2 Strassen)
    #   1x2x2 : 4  (schoolbook)
    #   1x2x4 : 2 * (1x2x2) = 8
    #   2x3x2 : 2 * (1x3x2) = 2 * 6 = 12  (schoolbook for 1x3x2 = 6, times 2 for the n-dim split)
    #   2x3x4 : 2 * (1x3x4) = 2 * 12 = 24
    #   1x3x2 : 6
    #   1x3x4 : 12
    # Total = 7 + 14 + 4 + 8 + 12 + 24 + 6 + 12 = 87

    total_rank = 87

    U = [[0] * total_rank for _ in range(n * m)]  # 15 x 87
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x 87
    W = [[0] * total_rank for _ in range(p * n)]  # 18 x 87

    col = 0

    # Helper: add a rank-r decomposition for an (an x bm x ap) sub-block
    # starting at (i0, j0, k0) in the global (n, m, p) space.
    # The sub-block's local indices (i, j, k) map to global (i0+i, j0+j, k0+k).
    # U_local has shape (an*bm, r), V_local (bm*ap, r), W_local (ap*an, r).
    def add_block(i0, j0, k0, an, bm, ap, U_local, V_local, W_local):
        nonlocal col
        r = U_local[0].__len__()
        assert col + r <= total_rank, f"col {col} + r {r} exceeds {total_rank}"
        for li in range(an * bm):
            i_local = li // bm
            j_local = li % bm
            gi = i0 + i_local
            gj = j0 + j_local
            g_ui = gi * m + gj
            for c in range(r):
                U[g_ui][col + c] += U_local[li][c]
        for lj in range(bm * ap):
            j_local = lj // ap
            k_local = lj % ap
            gj = j0 + j_local
            gk = k0 + k_local
            g_vj = gj * p + gk
            for c in range(r):
                V[g_vj][col + c] += V_local[lj][c]
        for lk in range(ap * an):
            k_local = lk // an
            i_local = lk % an
            gk = k0 + k_local
            gi = i0 + i_local
            g_wn = gk * n + gi
            for c in range(r):
                W[g_wn][col + c] += W_local[lk][c]
        col += r

    # --- Build small tensor decompositions ---

    def schoolbook(an, bm, ap):
        r = an * bm * ap
        U = [[0] * r for _ in range(an * bm)]
        V = [[0] * r for _ in range(bm * ap)]
        W = [[0] * r for _ in range(ap * an)]
        c = 0
        for i in range(an):
            for j in range(bm):
                for k in range(ap):
                    U[i * bm + j][c] = 1
                    V[j * ap + k][c] = 1
                    W[k * an + i][c] = 1
                    c += 1
        return U, V, W

    def strassen_222():
        # Rank-7 decomposition for 2x2x2 matrix multiplication.
        # U has shape (4, 7), V (4, 7), W (4, 7)
        # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
        # U row index = i*2 + j, V row index = j*2 + k, W row index = k*2 + i
        #
        # Strassen formulas:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22) B11
        # M3 = A11 (B12-B22)
        # M4 = A22 (B21-B11)
        # M5 = (A11+A12) B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # P11 = M1 + M4 - M5 + M7
        # P12 = M3 + M5
        # P21 = M2 + M4
        # P22 = M1 - M2 + M3 + M6

        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]

        # M1: A11+A22 times B11+B22
        # A rows: A11 is (i=0,j=0), A22 is (i=1,j=1)
        # B cols: B11 is (j=0,k=0), B22 is (j=1,k=1)
        # U: (0,0)->1, (1,1)->1 for M1
        # V: (0,0)->1, (1,1)->1 for M1
        # W: P11 += M1, P22 += M1
        # P11 is (i=0,k=0) => W row = k*2+i = 0*2+0 = 0
        # P22 is (i=1,k=1) => W row = k*2+i = 1*2+1 = 3
        for c, (ui, vi, wi) in enumerate([
            # (U_row, V_row, W_row) for each term; we'll build per-term
        ]):
            pass

        # Let's build it term by term more carefully.
        # Each "term" is a product of a linear combination of A entries and a linear combination of B entries.
        # We represent each term as a list of (u_coeff, u_row) for U, (v_coeff, v_row) for V, (w_coeff, w_row) for W.

        terms = []

        # M1 = (A11 + A22)(B11 + B22)
        # U: A11 at row (0,0)=0, A22 at row (1,1)=3
        # V: B11 at row (0,0)=0, B22 at row (1,1)=3
        # W contributions: P11 (row 0) += 1, P22 (row 3) += 1
        terms.append({
            'U': [(0, 1), (3, 1)],
            'V': [(0, 1), (3, 1)],
            'W': [(0, 1), (3, 1)]
        })

        # M2 = (A21 + A22)(B11)
        # U: A21 at row (1,0)=2, A22 at row (1,1)=3
        # V: B11 at row (0,0)=0
        # W: P21 (i=1,k=0) => W row = 0*2+1 = 1, += 1
        terms.append({
            'U': [(2, 1), (3, 1)],
            'V': [(0, 1)],
            'W': [(1, 1)]
        })

        # M3 = A11 (B12 - B22)
        # U: A11 at row (0,0)=0
        # V: B12 at row (0,1)=1, B22 at row (1,1)=3
        # W: P12 (i=0,k=1) => W row = 1*2+0 = 2, += 1
        terms.append({
            'U': [(0, 1)],
            'V': [(1, 1), (3, -1)],
            'W': [(2, 1)]
        })

        # M4 = A22 (B21 - B11)
        # U: A22 at row (1,1)=3
        # V: B21 at row (1,0)=2, B11 at row (0,0)=0
        # W: P11 (row 0) += 1, P21 (row 1) += 1
        terms.append({
            'U': [(3, 1)],
            'V': [(2, 1), (0, -1)],
            'W': [(0, 1), (1, 1)]
        })

        # M5 = (A11 + A12) B22
        # U: A11 at row (0,0)=0, A12 at row (0,1)=1
        # V: B22 at row (1,1)=3
        # W: P11 (row 0) -= 1, P12 (row 2) += 1
        terms.append({
            'U': [(0, 1), (1, 1)],
            'V': [(3, 1)],
            'W': [(0, -1), (2, 1)]
        })

        # M6 = (A21 - A11)(B11 + B12)
        # U: A21 at row (1,0)=2, A11 at row (0,0)=0
        # V: B11 at row (0,0)=0, B12 at row (0,1)=1
        # W: P22 (row 3) += 1
        terms.append({
            'U': [(2, 1), (0, -1)],
            'V': [(0, 1), (1, 1)],
            'W': [(3, 1)]
        })

        # M7 = (A12 - A22)(B21 + B22)
        # U: A12 at row (0,1)=1, A22 at row (1,1)=3
        # V: B21 at row (1,0)=2, B22 at row (1,1)=3
        # W: P11 (row 0) += 1
        terms.append({
            'U': [(1, 1), (3, -1)],
            'V': [(2, 1), (3, 1)],
            'W': [(0, 1)]
        })

        for t in terms:
            for (row, coeff) in t['U']:
                U[row][len(terms) - 1] += coeff
            for (row, coeff) in t['V']:
                V[row][len(terms) - 1] += coeff
            for (row, coeff) in t['W']:
                W[row][len(terms) - 1] += coeff

        return U, V, W

    # --- Now assemble the full 3x5x6 decomposition ---

    # Splits: n = 2+1, m = 2+3, p = 2+4

    # Block (i0,j0,k0, an,bm,ap):
    # (0,0,0, 2,2,2): Strassen, rank 7
    U_loc, V_loc, W_loc = strassen_222()
    add_block(0, 0, 0, 2, 2, 2, U_loc, V_loc, W_loc)

    # (0,0,2, 2,2,4): split p=4 into 2+2, two Strassen blocks
    U_loc, V_loc, W_loc = strassen_222()
    add_block(0, 0, 2, 2, 2, 2, U_loc, V_loc, W_loc)
    U_loc, V_loc, W_loc = strassen_222()
    add_block(0, 0, 4, 2, 2, 2, U_loc, V_loc, W_loc)

    # (0,2,0, 2,3,2): split i=2 into 1+1, j=3, k=2. Each 1x3x2 schoolbook (rank 6)
    U_loc, V_loc, W_loc = schoolbook(1, 3, 2)
    add_block(0, 2, 0, 1, 3, 2, U_loc, V_loc, W_loc)
    U_loc, V_loc, W_loc = schoolbook(1, 3, 2)
    add_block(1, 2, 0, 1, 3, 2, U_loc, V_loc, W_loc)

    # (0,2,2, 2,3,4): split i=2 into 1+1, p=4 into 2+2. Four 1x3x2 blocks
    U_loc, V_loc, W_loc = schoolbook(1, 3, 2)
    add_block(0, 2, 2, 1, 3, 2, U_loc, V_loc, W_loc)
    U_loc, V_loc, W_loc = schoolbook(1, 3, 2)
    add_block(0, 2, 4, 1, 3, 2, U_loc, V_loc, W_loc)
    U_loc, V_loc, W_loc = schoolbook(1, 3, 2)
    add_block(1, 2, 2, 1, 3, 2, U_loc, V_loc, W_loc)
    U_loc, V_loc, W_loc = schoolbook(1, 3, 2)
    add_block(1, 2, 4, 1, 3, 2, U_loc, V_loc, W_loc)

    # (2,0,0, 1,2,2): schoolbook 1x2x2, rank 4
    U_loc, V_loc, W_loc = schoolbook(1, 2, 2)
    add_block(2, 0, 0, 1, 2, 2, U_loc, V_loc, W_loc)

    # (2,0,2, 1,2,4): split p=4 into 2+2, two 1x2x2 blocks
    U_loc, V_loc, W_loc = schoolbook(1, 2, 2)
    add_block(2, 0, 2, 1, 2, 2, U_loc, V_loc, W_loc)
    U_loc, V_loc, W_loc = schoolbook(1, 2, 2)
    add_block(2, 0, 4, 1, 2, 2, U_loc, V_loc, W_loc)

    # (2,2,0, 1,3,2): schoolbook 1x3x2, rank 6
    U_loc, V_loc, W_loc = schoolbook(1, 3, 2)
    add_block(2, 2, 0, 1, 3, 2, U_loc, V_loc, W_loc)

    # (2,2,2, 1,3,4): split p=4 into 2+2, two 1x3x2 blocks
    U_loc, V_loc, W_loc = schoolbook(1, 3, 2)
    add_block(2, 2, 2, 1, 3, 2, U_loc, V_loc, W_loc)
    U_loc, V_loc, W_loc = schoolbook(1, 3, 2)
    add_block(2, 2, 4, 1, 3, 2, U_loc, V_loc, W_loc)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
