"""Matrix multiplication 3x4x5: search for rank-reducing Strassen (2x2x2) placements."""
import time
import itertools

DIMENSIONS = [3, 4, 5]
DEADLINE = time.time() + 135.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START


def _validate(U, V, W, n, m, p):
    """Check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # tensor component: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _to_halfint(x):
    """Convert a float to nearest half-integer; return int twice the value."""
    h = round(x * 2)
    # represent as fraction: value = h/2. Store as h/2.0 or as rational.
    # We'll store entries as Python floats equal to h/2 (exact in binary for small h).
    return h / 2.0


def _strassen_222(A_rows, B_cols, C_rows, C_cols):
    """
    Return (u, v, w) coefficient maps for a 2x2x2 Strassen block.
    A_rows = [r1, r2] rows of A (i-index)
    B_cols = [c1, c2] cols of B (k-index)
    C_rows = [i1, i2] rows of C (i-index)
    C_cols = [j1, j2] cols of C (j-index)

    A submatrix: A[[r1,r2],[c1,c2]]  (i-rows x k-cols)
    B submatrix: B[[c1,c2],[j1,j2]]  (k-rows x j-cols)
    C submatrix: C[[i1,i2],[j1,j2]]  (i-rows x j-cols)

    The sub-product: C_sub = A_sub @ B_sub
    A_sub[i_idx, k_idx] = A[r_i, c_k]
    B_sub[k_idx, j_idx] = B[c_k, j_j]
    C_sub[i_idx, j_idx] = C[i_i, j_j]

    Strassen 2x2x2 (rank 7). We define the 7 products and the linear combos.
    Return lists of (coeff, u_key, v_key, w_key) for each of the 7 terms.
    u_key: (i_row_in_sub, j_col_in_sub) -> maps to A[r_i, c_j]
    Actually U index uses (i, j) where j is the column of A = the k-index in B.
    Let me define:
      U term uses A entry at global (i, k) -> U[i*m + k, r]
      V term uses B entry at global (k, j) -> V[k*p + j, r]
      W term uses C entry at global (i, j) -> W[j*n + i, r]
    For each of the 7 Strassen products, U and V are linear combos of A/B entries,
    W is the coefficient on the corresponding C entry.
    """
    # Strassen products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B21 - B22)
    # M4 = A22 (B12 - B11)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Sub-indexing: A_sub: row a in {0,1} -> A_rows[a], col b in {0,1} -> B_cols[b]
    #   A11 = A_sub[0,0] = A[A_rows[0], B_cols[0]]
    #   A12 = A_sub[0,1] = A[A_rows[0], B_cols[1]]
    #   A21 = A_sub[1,0] = A[A_rows[1], B_cols[0]]
    #   A22 = A_sub[1,1] = A[A_rows[1], B_cols[1]]
    #   B11 = B[B_cols[0], C_cols[0]]
    #   B12 = B[B_cols[0], C_cols[1]]
    #   B21 = B[B_cols[1], C_cols[0]]
    #   B22 = B[B_cols[1], C_cols[1]]
    #   C11 = C[C_rows[0], C_cols[0]]
    #   C12 = C[C_rows[0], C_cols[1]]
    #   C21 = C[C_rows[1], C_cols[0]]
    #   C22 = C[C_rows[1], C_cols[1]]
    A_rows, B_cols, C_rows, C_cols = A_rows, B_cols, C_rows, C_cols

    # Each term: (U_coeffs list of (coef, i, k), V_coeffs list of (coef, k, j),
    #             W_coeffs list of (coef, i, j))
    # Build helper to reference.
    def aref(ai, bk):  # A entry
        return (A_rows[ai], B_cols[bk])
    def bref(bk, cj):  # B entry
        return (B_cols[bk], C_cols[cj])
    def cref(ci, cj):  # C entry
        return (C_rows[ci], C_cols[cj])

    terms = []
    # M1 = (A11 + A22)(B11 + B22)
    terms.append((
        [ (1, aref(0,0)), (1, aref(1,1)) ],
        [ (1, bref(0,0)), (1, bref(1,1)) ],
        [ (1, cref(0,0)), (1, cref(1,1)) ],
    ))
    # M2 = (A21 + A22) B11
    terms.append((
        [ (1, aref(1,0)), (1, aref(1,1)) ],
        [ (1, bref(0,0)) ],
        [ (1, cref(1,0)) ],
    ))
    # M3 = A11 (B21 - B22)
    terms.append((
        [ (1, aref(0,0)) ],
        [ (1, bref(1,0)), (-1, bref(1,1)) ],
        [ (1, cref(0,1)) ],
    ))
    # M4 = A22 (B12 - B11)
    terms.append((
        [ (1, aref(1,1)) ],
        [ (1, bref(0,1)), (-1, bref(0,0)) ],
        [ (1, cref(0,0)), (1, cref(1,0)) ],
    ))
    # M5 = (A11 + A12) B22
    terms.append((
        [ (1, aref(0,0)), (1, aref(0,1)) ],
        [ (1, bref(1,1)) ],
        [ (-1, cref(0,0)), (1, cref(0,1)) ],
    ))
    # M6 = (A21 - A11)(B11 + B12)
    terms.append((
        [ (1, aref(1,0)), (-1, aref(0,0)) ],
        [ (1, bref(0,0)), (1, bref(0,1)) ],
        [ (1, cref(1,1)) ],
    ))
    # M7 = (A12 - A22)(B21 + B22)
    terms.append((
        [ (1, aref(0,1)), (-1, aref(1,1)) ],
        [ (1, bref(1,0)), (1, bref(1,1)) ],
        [ (1, cref(0,0)), (1, cref(1,1)) ],
    ))
    return terms


def construct():
    n, m, p = DIMENSIONS
    import numpy as np

    # Start from schoolbook rank n*m*p.
    # We'll build U, V, W as lists of rows; each column is a rank term.
    # Represent each term as a dict: U: {i*m+j: coef}, V: {j*p+k: coef}, W: {k*n+i: coef}
    # where coef can be float (half-integer).

    U_terms = []  # list of dict
    V_terms = []
    W_terms = []

    def add_term(u, v, w):
        U_terms.append(u)
        V_terms.append(v)
        W_terms.append(w)

    # We'll track which (i,j,k) tensor entries have been "covered" by Strassen blocks.
    # Actually simpler: build the full tensor T as a 3D array, initialize to 0,
    # add contributions from each term, then verify it equals 1 everywhere.
    # A schoolbook term at (i,j,k) contributes 1 to T[i,j,k].
    # A Strassen block contributes to a set of (i,j,k) entries (the 2x2x2 sub-box
    #   where i in C_rows, j in C_cols, k in B_cols) via the linear combos.
    #
    # Strategy: pick a set of non-overlapping (disjoint in the (i,j,k) box sense)
    #   2x2x2 placements, apply Strassen to each (replacing 8 schoolbook terms by 7),
    #   and fill the rest with schoolbook. Verify T == 1.

    # Enumerate candidate placements: choose 2 A-rows (i), 2 B-cols (k),
    #   and require C-rows == the 2 A-rows, C-cols == the 2 B-cols? No.
    #   The sub-product C_sub = A_sub @ B_sub where A_sub is (i x k) and B_sub is (k x j).
    #   For the sub-box to be a clean sub-matrix product, we need:
    #     C[i,j] gets contributions only from A[i,k]*B[k,j] for i in I, k in K, j in J.
    #   This is always true in matrix multiplication: C[i,j] = sum_k A[i,k]B[k,j].
    #   So for a sub-box with i in I, k in K, j in J, the contribution to C[i,j]
    #   from k in K is sum_{k in K} A[i,k] B[k,j]. The Strassen block computes exactly
    #   that partial sum for i in I, j in J. The remaining k not in K are handled
    #   by schoolbook terms. So the blocks must be disjoint in the K dimension
    #   (the summation index) to avoid double counting, but they can share I or J.
    #   Actually, two blocks with overlapping K would double count. So require
    #   disjoint K sets. I and J can overlap freely.

    # Let's find a set of 2x2x2 blocks with disjoint K (B-col) sets.
    # m=4 B-columns. We can partition the 4 B-cols into two pairs: {0,1},{2,3}.
    # Each pair can be used with any choice of 2 A-rows (from n=3: C(3,2)=3 choices)
    # and any choice of 2 C-cols (from p=5: C(5,2)=10 choices).
    # But we want to maximize the number of Strassen blocks while keeping K disjoint.
    # Each block uses 2 B-cols. With m=4, max 2 blocks (partitioning B-cols).
    # For each, choose 2 A-rows and 2 C-cols. To cover more, we could use more blocks
    #   but they must have disjoint K. So at most floor(m/2) = 2 blocks if each uses 2 cols.
    #   Alternatively, blocks could use different K-pairs.

    # Let's just try: partition B-cols into 2 pairs, and for each pair choose A-rows
    #   and C-cols. Try all combinations and verify; pick the one that's valid
    #   (which they all are by construction) and count rank.
    # Rank = 60 - 2*(#blocks). With 2 blocks: rank = 56.

    # But wait: a block only saves 1 rank if the 2x2x2 sub-box is fully covered
    #   by Strassen (replacing 8 schoolbook with 7). Yes, each block saves 1.

    # Let's also consider: could we do a 3x3x3 or other structure? For simplicity,
    #   stick with 2x2x2 Strassen blocks.

    # Choose partition of B-cols (size m=4) into pairs.
    # All perfect matchings of {0,1,2,3}: 
    matchings = [
        [(0,1),(2,3)],
        [(0,2),(1,3)],
        [(0,3),(1,2)],
    ]
    # For each matching, and for each pair, choose 2 A-rows (from 3) and 2 C-cols (from 5).
    a_row_choices = list(itertools.combinations(range(n), 2))  # 3 choices
    c_col_choices = list(itertools.combinations(range(p), 2))  # 10 choices

    best = None
    for matching in matchings:
        for pair_assign in itertools.product(a_row_choices, c_col_choices,
                                             a_row_choices, c_col_choices):
            (a1, c1), (a2, c2) = pair_assign
            blocks = []
            for (bk1, bk2), arows, ccols in zip(matching, [a1, a2], [c1, c2]):
                blocks.append((arows, (bk1, bk2), arows, ccols))
            # Build and verify.
            # Compute total tensor T as 3D array.
            T = np.zeros((n, m, p), dtype=float)
            U_acc = {}  # (i,j) -> coef
            V_acc = {}  # (j,k) -> coef
            W_acc = {}  # (k,i) -> coef
            rank_terms = []
            # For each block, add Strassen terms.
            for arows, bcols, crows, ccols in blocks:
                terms = _strassen_222(arows, bcols, crows, ccols)
                for u, v, w in terms:
                    # Accumulate.
                    for coef, (i, k) in u:
                        key = (i, k)
                        U_acc[key] = U_acc.get(key, 0.0) + coef
                    for coef, (k, j) in v:
                        key = (k, j)
                        V_acc[key] = V_acc.get(key, 0.0) + coef
                    for coef, (i, j) in w:
                        key = (i, j)
                        W_acc[key] = W_acc.get(key, 0.0) + coef
                    # Verify tensor contribution: this term contributes
                    #   to T[i,j,k] for i in arows, k in bcols, j in ccols via
                    #   sum over the linear combos. Let's compute the contribution
                    #   to T for this term.
                    # The term is: (sum_u coef_u * A[i_u, k_u]) * (sum_v coef_v * B[k_v, j_v])
                    #   times W part. But W is a linear combo of C entries.
                    #   Actually the tensor identity: the product of the U-linear-combo
                    #   and V-linear-combo equals a linear combo of C[i,j] for the block.
                    #   We already encoded W as the C-coeffs. So the contribution to T[i,j,k]
                    #   is: for each (i,j) with W-coef w_ij, and the U,V combos give the
                    #   partial sum sum_{k in bcols} A[i,k]B[k,j] = C[i,j] (partial).
                    #   So T[i,j,k] for k in bcols gets w_ij * (coefficient of A[i,k]B[k,j] in the product).
                    # This is getting complex. Let's just verify by direct evaluation.
            # Instead of manual accumulation, let's verify by constructing the full
            #   U, V, W matrices (including schoolbook for the rest) and checking T==1.
            # Determine which (i,j,k) are covered by Strassen blocks (k in some bcols,
            #   i in some arows, j in some ccols). For covered entries, the Strassen
            #   terms handle them; for uncovered, use schoolbook.
            # But the Strassen terms' U,V,W are linear combos that span the block.
            #   The cleanest: build U, V, W as matrices where each column is one "term"
            #   (either a schoolbook basis term or a Strassen term), and verify T==1.

            # Let me rebuild: collect all terms (schoolbook + Strassen) as columns.
            # Mark which (i,j,k) are "claimed" by Strassen blocks to avoid double count.
            claimed = set()
            for arows, bcols, crows, ccols in blocks:
                for i in arows:
                    for k in bcols:
                        for j in ccols:
                            claimed.add((i, j, k))

            columns = []  # each column: (U_dict, V_dict, W_dict)
            # Add Strassen columns.
            for arows, bcols, crows, ccols in blocks:
                terms = _strassen_222(arows, bcols, crows, ccols)
                for u, v, w in terms:
                    Ud = {}
                    for coef, (i, k) in u:
                        Ud[(i, k)] = Ud.get((i, k), 0.0) + coef
                    Vd = {}
                    for coef, (k, j) in v:
                        Vd[(k, j)] = Vd.get((k, j), 0.0) + coef
                    Wd = {}
                    for coef, (i, j) in w:
                        Wd[(i, j)] = Wd.get((i, j), 0.0) + coef
                    columns.append((Ud, Vd, Wd))
            # Add schoolbook columns for unclaimed (i,j,k).
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i, j, k) not in claimed:
                            Ud = {(i, k): 1.0}
                            Vd = {(k, j): 1.0}
                            Wd = {(i, j): 1.0}
                            columns.append((Ud, Vd, Wd))
            rank = len(columns)
            if best is None or rank < best[0]:
                best = (rank, columns, blocks)

    if best is None:
        # Fallback: pure schoolbook.
        columns = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    columns.append(({(i,k):1.0}, {(k,j):1.0}, {(i,j):1.0}))
        rank = n * m * p
    else:
        rank, columns, blocks = best

    # Build U, V, W as 2D lists (rows x rank).
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    for r, (Ud, Vd, Wd) in enumerate(columns):
        for (i, k), coef in Ud.items():
            U[i * m + k][r] = coef
        for (k, j), coef in Vd.items():
            V[k * p + j][r] = coef
        for (i, j), coef in Wd.items():
            W[j * n + i][r] = coef

    # Convert to half-integers (all coefs here are integers already, but round).
    U = [[_to_halfint(x) for x in row] for row in U]
    V = [[_to_halfint(x) for x in row] for row in V]
    W = [[_to_halfint(x) for x in row] for row in W]

    # Verify.
    if not _validate(U, V, W, n, m, p):
        # Fallback to schoolbook.
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
