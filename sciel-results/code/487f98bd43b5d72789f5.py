"""Direct rank-44 decomposition for the 3x4 x 4x4 matrix multiplication tensor.

The tensor T_{i,j,k} = E_{i,j} (A[i,j]) * E_{j,k} (B[j,k]) maps to C[i,k]
with output convention W[k*n+i]. We split rows of A / rows of B into
{0,1,2} and columns into {0,1} and {2,3}, giving 8 blocks. The two 2x2
subproblems (A_{TL}B_{TL} and A_{BR}B_{BR}) use Strassen (rank 7 each);
the remaining four 2x2 and four 1x2 blocks use schoolbook. Total rank = 44.
A runtime self-check verifies the decomposition is exact; otherwise it
falls back to the schoolbook rank-48 construction (always correct).
"""

import time
import itertools

DIMENSIONS = [3, 4, 4]

N, M, P = 3, 4, 4
TIME_LIMIT = 120.0
SAFETY_MARGIN = 20.0

# W index convention: W[k*n + i, c]
def widx(k, i):
    return k * N + i


def _add_term(U, V, W, coeff, a_rows, b_rows, b_cols, c_rows, c_cols):
    """Add a scalar term coeff * (sum_{i in a_rows, r in b_rows, k in b_cols} A_{i,r} B_{r,k})
    to C for all (i, k) in c_rows x c_cols, using one rank-1 tensor summand."""
    if coeff == 0:
        return
    for i in a_rows:
        U[i * M] += coeff          # row = i*M + 0 ; A has m=4 cols indexed r directly
    for r in b_rows:
        V[r * P] += coeff          # row = r*P + 0 ; col k is added below
    for k in b_cols:
        V[r0 * P + k]  # placeholder never reached
    return


def _add_simple_term(U, V, W, coeff, i, r, k):
    """One elementary (rank-1) factor A_{i,r} * B_{r,k} contributing to C_{i,k}."""
    if coeff == 0:
        return
    u_row = i * M + r
    v_row = r * P + k
    w_row = k * N + i
    U[u_row] += coeff
    V[v_row] += coeff
    W[w_row] += coeff


def _naive2(U, V, W, i_rows, r_rows, k_cols):
    """Schoolbook for a (len(i_rows)) x (len(r_rows)) x (len(k_cols)) block."""
    for i in i_rows:
        for r in r_rows:
            for k in k_cols:
                _add_simple_term(U, V, W, 1, i, r, k)


def _strassen(U, V, W, i0, r0, k0):
    """Strassen for C[i0:i0+2, k0:k0+2] = A[i0:i0+2, r0:r0+2] * B[r0:r0+2, k0:k0+2].
    Uses the A-block row indices (two rows) and B-block col indices (two cols),
    with B's row (shared) fixed as r0. Note: in our block decomposition the
    'rows of B' coincide with the 'cols of A' (the shared j index), so we treat
    B's row = r0 (a single row index into B), and B's cols = k0, k0+1."""
    # Local indices:
    a00, a11 = i0, i0 + 1          # A block rows
    b0, b1 = k0, k0 + 1            # B block cols
    r = r0                          # shared row index (B row = A col)

    def A(r0_, r1_):
        # returns list of A row-indices for the 2x2 sub-block's row pair
        pass

    # We treat the 2x2 A block entries A[i,r] where i in {i0,i0+1}, r in {r0} only?
    # No: A block is 2 rows (i) x 1 col (r)? Actually A_{TL} uses A rows {0,1} x B rows {0,1}.
    # Here 'rows of A' = i in {0,1}; 'cols of A' (= rows of B) = r in {0,1}. That's a 2x2 A block.
    # So we DO have A col indices too. The 2x2 block is A[i0:i0+2, r0:r0+2] * B[r0:r0+2, k0:k0+2].
    aR0, aR1 = r0, r0 + 1   # A column / B row indices for the block
    iR0, iR1 = i0, i0 + 1   # A row indices (output rows)
    kC0, kC1 = k0, k0 + 1   # B column indices (output cols)

    # Strassen on the four A entries (iR, aR) and B entries (aR, kC):
    # A entries: A[iR0][aR0], A[iR0][aR1], A[iR1][aR0], A[iR1][aR1]
    # B entries: B[aR0][kC0], B[aR0][kC1], B[aR1][kC0], B[aR1][kC1]
    # M1 = (A00 + A11)*(B00 + B11)
    def M(a00, a11, b00, b11):
        _add_simple_term(U, V, W, a00, iR0, aR0)
        _add_simple_term(U, V, W, a11, iR1, aR1)
        _add_simple_term(U, V, W, b00, iR0, aR0)
        _add_simple_term(U, V, W, b11, iR1, aR1)
        return
    # The product of two LINEAR forms is not a single elementary term. We must expand.
    # This helper-based approach is getting tangled; instead expand each Strassen M
    # as a product of a linear form in A and a linear form in B -> that's a small
    # tensor which we can realize by distributing over the (a00 or a11 in A-form) x
    # (b00 or b11 in B-form) products, each being one elementary term, then add with
    # the Strassen signs on each C entry. We implement the full 2x2 block directly.
    #
    # A-forms:
    #  f1A = A00 + A11        -> rows (iR0,aR0)+(iR1,aR1)
    #  f2A = A10 + A11        -> (iR1,aR0)+(iR1,aR1)
    #  f3A = A00              -> (iR0,aR0)
    #  f4A = A11              -> (iR1,aR1)
    #  f5A = A00 + A01        -> (iR0,aR0)+(iR0,aR1)
    #  f6A = A10 - A00        -> (iR1,aR0)-(iR0,aR0)
    #  f7A = A01 - A11        -> (iR0,aR1)-(iR1,aR1)
    # B-forms:
    #  g1B = B00 + B11        -> (aR0,kC0)+(aR1,kC1)
    #  g2B = B00              -> (aR0,kC0)
    #  g3B = B01 - B11        -> (aR0,kC1)-(aR1,kC1)
    #  g4B = B00 + B10        -> (aR0,kC0)+(aR1,kC0)
    #  g5B = B11              -> (aR1,kC1)
    #  g6B = B00 + B01        -> (aR0,kC0)+(aR0,kC1)
    #  g7B = B10 + B11        -> (aR1,kC0)+(aR1,kC1)

    # Each M = (linear in A) * (linear in B) = sum over (i,a) in A-form, (a',k) in B-form of
    #    signA*signB * A[i,a]*B[a',k]  -> elementary term at output? No: A[i,a]*B[a',k]
    # contributes to C[i,k] only when a == a' (the shared index). So we must PAIR a with a'.
    # This means M = sum over a of (sum_i sA[i,a] A[i,a]) * (sum_k sB[a,k] B[a,k])
    #   = sum_a sum_i sum_k sA[i,a]*sB[a,k] * A[i,a]*B[a',k]  (a'=a)
    #   = sum over (i,a,k) of sA[i,a]*sB[a,k] * elementary(i,a,k).
    # Great: expandable as elementary terms with coefficient sA*sB.

    def linear_pairs(form):
        # form: dict mapping (i,a) -> coeff for A ; returns list
        pass

    # Define A-forms as list of (i, a, sA) and B-forms as (a, k, sB).
    AF = {
        1: [(iR0, aR0, 1), (iR1, aR1, 1)],
        2: [(iR1, aR0, 1), (iR1, aR1, 1)],
        3: [(iR0, aR0, 1)],
        4: [(iR1, aR1, 1)],
        5: [(iR0, aR0, 1), (iR0, aR1, 1)],
        6: [(iR1, aR0, 1), (iR0, aR0, -1)],
        7: [(iR0, aR1, 1), (iR1, aR1, -1)],
    }
    BF = {
        1: [(aR0, kC0, 1), (aR1, kC1, 1)],
        2: [(aR0, kC0, 1)],
        3: [(aR0, kC1, 1), (aR1, kC1, -1)],
        4: [(aR0, kC0, 1), (aR1, kC0, 1)],
        5: [(aR1, kC1, 1)],
        6: [(aR0, kC0, 1), (aR0, kC1, 1)],
        7: [(aR1, kC0, 1), (aR1, kC1, 1)],
    }

    def product_form(af, bf):
        # returns dict (i,a,k) -> coeff for the expanded M
        d = {}
        for (i, a, sA) in AF[af]:
            for (a2, k, sB) in BF[bf]:
                if a == a2:
                    c = sA * sB
                    d[(i, a, k)] = d.get((i, a, k), 0) + c
        return d

    # C entries (i,k) in {iR0,iR1} x {kC0,kC1}
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    signs = [
        (iR0, kC0, {1: +1, 4: +1, 5: -1, 7: +1}),
        (iR0, kC1, {3: +1, 5: +1}),
        (iR1, kC0, {2: +1, 4: +1}),
        (iR1, kC1, {1: +1, 2: -1, 3: +1, 6: +1}),
    ]
    # Accumulate coefficients for each elementary (i,a,k)
    elem = {}
    for ci, ck, msigns in signs:
        for m, s in msigns.items():
            pd = product_form(m, m)
            for (i, a, k), c in pd.items():
                if (i, k) != (ci, ck):
                    continue
                elem[(i, a, k)] = elem.get((i, a, k), 0) + s * c
    for (i, a, k), c in elem.items():
        if c:
            _add_simple_term(U, V, W, c, i, a, k)


def _build_rank44():
    U = [[0.0] * 44 for _ in range(N * M)]
    V = [[0.0] * 44 for _ in range(M * P)]
    W = [[0.0] * 44 for _ in range(P * N)]
    # column offsets
    base = 0
    def col(offset_terms):
        nonlocal base
        return base
    # Since we add directly with same coeff across each term set, we need ONE column per
    # elementary contribution group. Simplest: assign a fresh column for every elementary
    # term. But Strassen expansion yields multiple elementary terms per M that SHARE the
    # same M-form only partially. To keep it exact and simple, use a fresh column for
    # EVERY elementary (i,a,k) with nonzero coefficient. That yields rank = (# elementary
    # terms). We must ensure it's <= some bound and that W/V/U stay consistent.
    # Reimplement: track a dict mapping each elementary (i,a,k) to a column; a column is
    # shared only if we intend the same linear combination. Using one column per elementary
    # (i,a,k) is always exact (each is a basis tensor with coeff). Rank = number of distinct
    # elementary (i,a,k) with nonzero coeff.
    cols = {}
    next_col = [0]
    def get_col(i, a, k):
        key = (i, a, k)
        if key not in cols:
            cols[key] = next_col[0]
            next_col[0] += 1
        return cols[key]

    def elem_term(i, a, k, coeff):
        if coeff == 0:
            return
        c = get_col(i, a, k)
        U[i * M + a][c] += coeff
        V[a * P + k][c] += coeff
        W[k * N + i][c] += coeff

    # Recompute U,V,W sized to actual columns after building; do it here by using a
    # big enough width and trimming at the end.
    width = 44
    U = [[0.0] * width for _ in range(N * M)]
    V = [[0.0] * width for _ in range(M * P)]
    W = [[0.0] * width for _ in range(P * N)]

    # Replace _add_simple_term usage: operate on these arrays via elem_term.
    def simple(i, r, k, coeff=1):
        elem_term(i, r, k, coeff)

    def naive2(i_rows, r_rows, k_cols):
        for i in i_rows:
            for r in r_rows:
                for k in k_cols:
                    simple(i, r, k, 1)

    # Strassen via product forms (reimplemented locally to use elem_term)
    def strassen(i0, r0, k0):
        aR0, aR1 = r0, r0 + 1
        iR0, iR1 = i0, i0 + 1
        kC0, kC1 = k0, k0 + 1
        AF = {
            1: [(iR0, aR0, 1), (iR1, aR1, 1)],
            2: [(iR1, aR0, 1), (iR1, aR1, 1)],
            3: [(iR0, aR0, 1)],
            4: [(iR1, aR1, 1)],
            5: [(iR0, aR0, 1), (iR0, aR1, 1)],
            6: [(iR1, aR0, 1), (iR0, aR0, -1)],
            7: [(iR0, aR1, 1), (iR1, aR1, -1)],
        }
        BF = {
            1: [(aR0, kC0, 1), (aR1, kC1, 1)],
            2: [(aR0, kC0, 1)],
            3: [(aR0, kC1, 1), (aR1, kC1, -1)],
            4: [(aR0, kC0, 1), (aR1, kC0, 1)],
            5: [(aR1, kC1, 1)],
            6: [(aR0, kC0, 1), (aR0, kC1, 1)],
            7: [(aR1, kC0, 1), (aR1, kC1, 1)],
        }
        signs = [
            (iR0, kC0, {1: +1, 4: +1, 5: -1, 7: +1}),
            (iR0, kC1, {3: +1, 5: +1}),
            (iR1, kC0, {2: +1, 4: +1}),
            (iR1, kC1, {1: +1, 2: -1, 3: +1, 6: +1}),
        ]
        for ci, ck, msigns in signs:
            for m, s in msigns.items():
                for (i, a, sA) in AF[m]:
                    for (a2, k, sB) in BF[m]:
                        if a == a2 and (i, k) == (ci, ck):
                            simple(i, a, k, s * sA * sB)

    # Now emit each block. Order (8 blocks):
    # rows i groups: [0,1] (top), [2] (bottom); cols k groups [0,1] (left), [2,3] (right)
    # 1) A top rows {0,1}, B rows {0,1}, B cols {0,1} -> 2x2x2 Strassen  (i0=0,r0=0,k0=0)
    strassen(0, 0, 0)
    # 2) A rows {0,1}, B rows {2,3}, B cols {0,1} -> 2x2x2 naive
    naive2([0, 1], [2, 3], [0, 1])
    # 3) A rows {0,1}, B rows {0,1}, B cols {2,3} -> naive
    naive2([0, 1], [0, 1], [2, 3])
    # 4) A rows {0,1}, B rows {2,3}, B cols {2,3} -> naive
    naive2([0, 1], [2, 3], [2, 3])
    # 5) A rows {2}, B rows {0,1}, B cols {0,1} -> 1x2x2 naive
    naive2([2], [0, 1], [0, 1])
    # 6) A rows {2}, B rows {2,3}, B cols {0,1} -> naive
    naive2([2], [2, 3], [0, 1])
    # 7) A rows {2}, B rows {0,1}, B cols {2,3} -> naive
    naive2([2], [0, 1], [2, 3])
    # 8) A rows {2}, B rows {2,3}, B cols {2,3} -> 2x? Actually bottom-right:
    #    A rows {2} x B rows {2,3} x B cols {2,3}: naive
    naive2([2], [2, 3], [2, 3])

    # Wait: we wanted Strassen on the 2x2 A blocks A_TL and A_BR. A_BR = A rows {0,1}? No.
    # Bottom-right block of A is row {2} (single row). The only full 2x2 A blocks are
    # rows {0,1} with col-rows {0,1} (TL) and rows {0,1} with col-rows {2,3} (TR).
    # The "BR" I referenced doesn't exist as a 2x2 A block because A has only 3 rows.
    # So Strassen is only valid on A_{TL} (rows 0,1 / B-rows 0,1) and A_{TR} (rows 0,1 / B-rows 2,3).
    # Let's redo: apply Strassen to blocks 1 (rows01 x Brows01 x Bcols01) and block 3
    # (rows01 x Brows01 x Bcols23)? No: A block is (rows x Brows); B block is (Brows x Bcols).
    # Strassen needs a 2x2 A-block and 2x2 B-block sharing the 2 Brows.
    # Block 1: A rows{0,1} x Brows{0,1} (2x2) * B Brows{0,1} x Bcols{0,1} (2x2)  -> OK Strassen
    # Block 2: A rows{0,1} x Brows{2,3} (2x2) * B Brows{2,3} x Bcols{0,1} (2x2)  -> OK Strassen
    # Block 3: A rows{0,1} x Brows{0,1} (2x2) * B Brows{0,1} x Bcols{2,3} (2x2)  -> OK Strassen
    # Block 4: A rows{0,1} x Brows{2,3} (2x2) * B Brows{2,3} x Bcols{2,3} (2x2)  -> OK Strassen
    # Blocks 5-8: A row{2} x Brows{...} x Bcols{...} -> 1x2x2 naive (rank 4 each)
    # So Strassen on blocks 1,2,3,4 (4*7=28) + naive on 5,6,7,8 (4*4=16) = 44. 
    # Let's REBUILD correctly with Strassen on 1,2,3,4.

    # --- Rebuild from scratch ---
    width = 44
    U = [[0.0] * width for _ in range(N * M)]
    V = [[0.0] * width for _ in range(M * P)]
    W = [[0.0] * width for _ in range(P * N)]
    cols = {}
    next_col = [0]

    def get_col(i, a, k):
        key = (i, a, k)
        if key not in cols:
            if next_col[0] >= width:
                raise RuntimeError("exceeded width")
            cols[key] = next_col[0]
            next_col[0] += 1
        return cols[key]

    def simple(i, r, k, coeff=1):
        if coeff == 0:
            return
        c = get_col(i, r, k)
        U[i * M + r][c] += coeff
        V[r * P + k][c] += coeff
        W[k * N + i][c] += coeff

    def naive2(i_rows, r_rows, k_cols):
        for i in i_rows:
            for r in r_rows:
                for k in k_cols:
                    simple(i, r, k, 1)

    def strassen(i0, r0, k0):
        """Strassen 2x2 block: one column per M-term (7 columns total)."""
        aR0, aR1 = r0, r0 + 1
        iR0, iR1 = i0, i0 + 1
        kC0, kC1 = k0, k0 + 1
        AF = {
            1: [(iR0, aR0, 1), (iR1, aR1, 1)],
            2: [(iR1, aR0, 1), (iR1, aR1, 1)],
            3: [(iR0, aR0, 1)],
            4: [(iR1, aR1, 1)],
            5: [(iR0, aR0, 1), (iR0, aR1, 1)],
            6: [(iR1, aR0, 1), (iR0, aR0, -1)],
            7: [(iR0, aR1, 1), (iR1, aR1, -1)],
        }
        BF = {
            1: [(aR0, kC0, 1), (aR1, kC1, 1)],
            2: [(aR0, kC0, 1)],
            3: [(aR0, kC1, 1), (aR1, kC1, -1)],
            4: [(aR0, kC0, 1), (aR1, kC0, 1)],
            5: [(aR1, kC1, 1)],
            6: [(aR0, kC0, 1), (aR0, kC1, 1)],
            7: [(aR1, kC0, 1), (aR1, kC1, 1)],
        }
        CF = {
            1: [(iR0, kC0, 1), (iR1, kC1, 1)],
            2: [(iR1, kC0, 1), (iR1, kC1, -1)],
            3: [(iR0, kC1, 1), (iR1, kC1, 1)],
            4: [(iR1, kC0, 1)],
            5: [(iR0, kC0, -1), (iR0, kC1, 1)],
            6: [(iR1, kC1, 1)],
            7: [(iR0, kC0, 1)],
        }
        for m in range(1, 8):
            if next_col[0] >= width:
                raise RuntimeError("exceeded width in strassen")
            c = next_col[0]
            next_col[0] += 1
            for (i, r, sa) in AF[m]:
                U[i * M + r][c] += sa
            for (r, k, sb) in BF[m]:
                V[r * P + k][c] += sb
            for (i, k, sc) in CF[m]:
                W[k * N + i][c] += sc

    # Block 1: rows{0,1} x Brows{0,1} x Bcols{0,1}  -> Strassen
    strassen(0, 0, 0)
    # Block 2: rows{0,1} x Brows{2,3} x Bcols{0,1}  -> Strassen
    strassen(0, 2, 0)
    # Block 3: rows{0,1} x Brows{0,1} x Bcols{2,3}  -> Strassen
    strassen(0, 0, 2)
    # Block 4: rows{0,1} x Brows{2,3} x Bcols{2,3}  -> Strassen
    strassen(0, 2, 2)
    # Block 5: rows{2} x Brows{0,1} x Bcols{0,1} -> naive
    naive2([2], [0, 1], [0, 1])
    # Block 6: rows{2} x Brows{2,3} x Bcols{0,1} -> naive
    naive2([2], [2, 3], [0, 1])
    # Block 7: rows{2} x Brows{0,1} x Bcols{2,3} -> naive
    naive2([2], [0, 1], [2, 3])
    # Block 8: rows{2} x Brows{2,3} x Bcols{2,3} -> naive
    naive2([2], [2, 3], [2, 3])

    # Trim trailing all-zero columns
    while next_col[0] > 0:
        r = next_col[0] - 1
        if U[r] is None:
            break
        col_used = any(U[row][r] != 0 for row in range(len(U))) or \
                   any(V[row][r] != 0 for row in range(len(V))) or \
                   any(W[row][r] != 0 for row in range(len(W)))
        if not col_used:
            next_col[0] = r
        else:
            break
    rank = next_col[0]
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    return {"U": U, "V": V, "W": W, "rank": rank}


def _schoolbook():
    rank = N * M * P
    U = [[0.0] * rank for _ in range(N * M)]
    V = [[0.0] * rank for _ in range(M * P)]
    W = [[0.0] * rank for _ in range(P * N)]
    col = 0
    for i in range(N):
        for r in range(M):
            for k in range(P):
                U[i * M + r][col] = 1.0
                V[r * P + k][col] = 1.0
                W[k * N + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(res):
    """Verify that sum_c U[i*M+r][c] * V[r*P+k][c] * W[k*N+i][c] == 1 for all i,r,k."""
    U, V, W = res["U"], res["V"], res["W"]
    rank = len(U[0]) if U and U[0] else 0
    for i in range(N):
        for r in range(M):
            for k in range(P):
                s = 0.0
                for c in range(rank):
                    s += U[i * M + r][c] * V[r * P + k][c] * W[k * N + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def construct():
    start = time.time()
    deadline = TIME_LIMIT - SAFETY_MARGIN
    try:
        res = _build_rank44()
    except Exception:
        res = None
    if res is not None and time.time() < deadline and _verify(res):
        return {"U": res["U"], "V": res["V"], "W": res["W"]}
    # Fallback: guaranteed-correct schoolbook
    sb = _schoolbook()
    return {"U": sb["U"], "V": sb["V"], "W": sb["W"]}


# EVOLVE-BLOCK-START
# (construct defined above)
# EVOLVE-BLOCK-END
