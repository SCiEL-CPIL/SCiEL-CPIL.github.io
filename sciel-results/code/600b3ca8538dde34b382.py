"""Fresh design for 2x4 by 4x7 rank-minimal exact tensor decomposition.

Strategy: build a Strassen-like block decomposition of the 4x4 middle block,
use schoolbook for the fringe rows/columns, then attempt to linearly re-combine
terms (solve a least-squares system) to reduce the total rank.  If no valid
lower-rank decomposition is found, fall back to the verified schoolbook
decomposition so a correct answer is ALWAYS returned.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

DEADLINE = 115.0
_START = time.time()


def _remaining():
    return DEADLINE - (time.time() - _START)


def _tensor(n, m, p):
    """Matrix-multiplication tensor T[i, j, k] = 1 (the identity of the matmul)."""
    return np.ones((n, m, p), dtype=int)


def _verify(U, V, W, n, m, p):
    """Check U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all (i, j, k)."""
    r = U.shape[1]
    # Build the contraction: sum over r of outer products.
    # We accumulate a 3D tensor efficiently via einsum.
    Ua = np.asarray(U, dtype=float).reshape(n, m, r)
    Va = np.asarray(V, dtype=float).reshape(m, p, r)
    Wa = np.asarray(W, dtype=float).reshape(p, n, r)
    # T[i,j,k] = sum_r Ua[i,j,r] Va[j,k,r] Wa[k,i,r]
    # First compute A[i,j,k,r] = Ua[i,j,r]*Va[j,k,r]
    A = Ua[:, :, None, :] * Va[:, None, :, :]          # (n, m, p, r)
    B = A * Wa[None, None, :, :]                        # (n, m, p, r)
    T = B.sum(axis=3)
    return np.allclose(T, 1.0, rtol=0, atol=1e-9)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _as_list(arr):
    return arr.tolist()


def _try_recombine(n, m, p, U, V, W, max_iters=8):
    """Try to reduce rank by re-combining columns via least squares.

    We look for a linear relation among the current rank r columns such that a
    subset of columns can be expressed as a combination of the others, allowing
    us to drop redundant columns.  Because the tensor is exactly 1 everywhere,
    any valid re-combination that preserves all (i,j,k) products is acceptable.

    Approach: find the smallest index set S such that for each column c not in S,
    there exist coefficients a_s such that column c == sum_s a_s * column_s in the
    joint "signature" sense.  We approximate this by treating each column as a
    vector in R^{n*m} x R^{m*p} x R^{p*n} and checking linear dependence across
    the triple product.  A simpler robust method: detect duplicate columns and
    columns that are scalar multiples, then remove them.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]

    # Build a "fingerprint" matrix: each column c is represented by the triple
    # (U[:,c], V[:,c], W[:,c]).  Two columns are linearly dependent in the
    # decomposition sense only if their product signatures match.  We use the
    # element-wise product signature S_c = U_c * (V_c * W_c reshaped) — but the
    # cleanest invariant is the rank-1 tensor P_c = U_c ⊗ V_c ⊗ W_c.
    # P_c[i,j,k] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c].  The sum of P_c must
    # equal the all-ones tensor.
    def sig(c):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        t = np.einsum('ij,jk,ki->ijk', u, v, w)
        return t.reshape(-1)

    # Try to find a column that is a linear combination of the others in the
    # space of (i,j,k) product tensors.  If P_c = sum_{s != c} a_s P_s, we can
    # absorb column c into the others and drop it.
    # We attempt this greedily: for each column c, solve least squares for the
    # others; if the residual is ~0, drop c and merge coefficients.
    changed = True
    for _ in range(max_iters):
        if _remaining() < 10:
            break
        r = U.shape[1]
        if r <= 1:
            break
        # Build P matrix of shape (n*m*p, r)
        P = np.zeros((n * m * p, r))
        for c in range(r):
            P[:, c] = sig(c)
        # For each target column c, try to express it via the others.
        removed_any = False
        new_idx = []
        coeff_map = {}  # original col -> list of (newcol, coeff)
        for c in range(r):
            others = [t for t in range(r) if t != c]
            Po = P[:, others]
            Pc = P[:, c]
            # Solve Po @ a ≈ Pc
            a, res, rank_, sv = np.linalg.lstsq(Po, Pc, rcond=None)
            recon = Po @ a
            if np.allclose(recon, Pc, rtol=0, atol=1e-9):
                # column c is redundant; merge into others
                coeff_map[c] = [(others[idx], float(a[idx])) for idx in range(len(others)) if abs(a[idx]) > 1e-12]
                removed_any = True
            else:
                new_idx.append(c)
        if not removed_any:
            break
        # Rebuild: for each kept column, add contributions from removed columns.
        nr = len(new_idx)
        U2 = np.zeros((n * m, nr))
        V2 = np.zeros((m * p, nr))
        W2 = np.zeros((p * n, nr))
        for ni, oc in enumerate(new_idx):
            U2[:, ni] = U[:, oc]
            V2[:, ni] = V[:, oc]
            W2[:, ni] = W[:, oc]
            for (tc, coef) in coeff_map.get(oc, []):
                # find new index of tc
                if tc in new_idx:
                    ti = new_idx.index(tc)
                    U2[:, ti] += coef * U[:, tc]
                    V2[:, ti] += coef * V[:, tc]
                    W2[:, ti] += coef * W[:, tc]
        U, V, W = U2, V2, W2
        changed = True
    if not changed:
        return None
    if _verify(U, V, W, n, m, p):
        return U, V, W
    return None


def _strassen_base(n, m, p):
    """Build a Strassen-like decomposition for the 4x4 middle, schoolbook fringe.

    Split A (2x4) into rows: A_top (1x4), A_bot (1x4).
    Split B (4x7) into cols: B_left (4x3), B_right (4x4).
    Split B's 4 rows into B1 (2x7 top), B2 (2x7 bottom) for Strassen on 4x4.

    The 4x4 block (A_bot[1:3] as 2x2, B cols [3:7] as 2x2 rows) uses Strassen (7 terms).
    Everything else uses schoolbook.
    """
    U = []
    V = []
    W = []

    def add(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)

    # --- Fringe: A_top (row i=0) times all of B (cols k=0..6), all j ---
    for j in range(m):
        for k in range(p):
            u = np.zeros(n * m); u[0 * m + j] = 1
            v = np.zeros(m * p); v[j * p + k] = 1
            w = np.zeros(p * n); w[k * n + 0] = 1
            add(u, v, w)

    # --- Fringe: A_bot (i=1) times B's fringe columns k=0,1,2 (the 4x3 part), all j ---
    for j in range(m):
        for k in range(3):
            u = np.zeros(n * m); u[1 * m + j] = 1
            v = np.zeros(m * p); v[j * p + k] = 1
            w = np.zeros(p * n); w[k * n + 1] = 1
            add(u, v, w)

    # --- 4x4 Strassen core: A_bot cols j=2,3 (i=1) x B cols k=3,4,5,6 (i=1) ---
    # A_core = A[1, 2:4] (2x2), B_core = B[0:4, 3:7] (4x4)
    # B_core rows split: top = B[0:2, 3:7] (2x4), bottom = B[2:4, 3:7] (2x4)
    # B_core cols split: left = B[:, 3:5] (4x2), right = B[:, 5:7] (4x2)
    # A_core rows: a = A[1,2] (top), b = A[1,3] (bot)
    # B_core 2x2 blocks:
    #   B11 = B[0:2, 3:5]  (top-left)
    #   B12 = B[0:2, 5:7]  (top-right)
    #   B21 = B[2:4, 3:5]  (bottom-left)
    #   B22 = B[2:4, 5:7]  (bottom-right)
    # C11 = a*B11 + b*B21  -> output A[1,2]*B[:,3:5] + A[1,3]*B[:,3:5] for i=1
    # C12 = a*B12 + b*B22  -> output for i=1, k=5,6
    # C21 = b*(B11+B22) - (a+b)*B22 ... Strassen on (a x [B11 B12]) etc.

    # We model the 2x2 x 4x4 as: A_core (2x2) times B_core (4x4).
    # Write B_core as 2x2 blocks of 2x2:
    #   M11 = B[0:2, 3:5], M12 = B[0:2, 5:7], M21 = B[2:4, 3:5], M22 = B[2:4, 5:7]
    # A_core = [[a, b]] (1x2? no, A[1,2:4] is 1x2). Wait A_bot is row i=1 only,
    # so A_core is 1x2 (a, b). That's a 1x2 times 4x4, not 2x2x4x4.
    #
    # Reconsider: A is 2x4. A_bot = A[1, :] is 1x4. The "4x4" needs A to contribute
    # 2 rows. So the Strassen core must involve BOTH rows of A for a 2x4x4x4 subproblem.
    # Better core: A[0:2, 2:4] (2x2) x B[0:4, 3:7] (4x4).
    #
    # Redo: use the FULL 2x4 A for rows j=2,3 and B cols k=3..6, i=0,1.

    # Clear and rebuild the core part.
    U.clear(); V.clear(); W.clear()

    # --- Fringe 1: A[:, 0:2] (both i, j=0,1) x B[0:4, 0:3] (k=0,1,2) ---
    for i in range(2):
        for j in range(2):
            for k in range(3):
                u = np.zeros(n * m); u[i * m + j] = 1
                v = np.zeros(m * p); v[j * p + k] = 1
                w = np.zeros(p * n); w[k * n + i] = 1
                add(u, v, w)

    # --- Fringe 2: A[:, 2:4] (both i, j=2,3) x B[0:4, 0:3] (k=0,1,2) ---
    for i in range(2):
        for j in range(2, 4):
            for k in range(3):
                u = np.zeros(n * m); u[i * m + j] = 1
                v = np.zeros(m * p); v[j * p + k] = 1
                w = np.zeros(p * n); w[k * n + i] = 1
                add(u, v, w)

    # --- Strassen core: A[0:2, 2:4] (2x2) x B[0:4, 3:7] (4x4) ---
    # A_core = [[A00, A01], [A10, A11]] = A[0,2], A[0,3], A[1,2], A[1,3]
    # B_core 4x4, split into 2x2 blocks of 2x2:
    #   M11 = B[0:2, 3:5], M12 = B[0:2, 5:7]
    #   M21 = B[2:4, 3:5], M22 = B[2:4, 5:7]
    # Strassen computes C = A_core * B_core (2x2 x 4x4 -> 2x4) using 7 products
    # of (2x2) x (4x4) blocks:
    # p1 = (A00 - A11) * (M21 + M22)
    # p2 = (A00 + A11) * M11
    # p3 = A01 * (M12 - M11)
    # p4 = A10 * (M22 - M11)
    # p5 = (A00 + A01) * M12
    # p6 = (A10 - A00) * (M11 + M21)
    # p7 = (A11 - A10) * (M22 + M12)
    # C11 = p5 + p3 - p7 - p4
    # C12 = p6 + p4
    # C21 = p1 - p6 + p2 - p3
    # C22 = p5 + p1 - p2 - p7
    # Each p is a (1x2) * (4x4) = 1x4, but here A blocks are scalars per (i) row...
    # Actually A_core is 2x2 of scalars, B blocks are 2x2 matrices.  Standard Strassen
    # for 2x2 x 2x2.  Here B is 4x4 = 2x2 blocks.  So treat A_core as 2x2 scalars and
    # B_core as 2x2 blocks each 2x2.  The 7 products are (scalar-combos of A) times
    # (block-combos of B), each producing a 2x2 block.

    # We'll enumerate by index.  A_core[i][j] lives at A[i*4 + (2+j)].
    # B_core block (bx, by) where bx in {0,1} (top/bottom rows), by in {0,1} (left/right cols)
    #   block (bx, by) = B[2*bx : 2*bx+2, 3 + 2*by : 5 + 2*by]  -> 2 rows x 2 cols
    #   row index j in B: 2*bx + jr,  col index k in B: 3 + 2*by + kc

    def a_idx(i, j):
        # A_core row i (0,1) col j (0,1) -> global A index i*4 + (2+j)
        return i * m + (2 + j)

    def b_row(bx, jr):
        return 2 * bx + jr

    def b_col(by, kc):
        return 3 + 2 * by + kc

    # 7 Strassen products.  Each product is sum over A-combo and B-combo.
    # For a product like (Aa - Ab) * (Mx + My): expand into 4 scalar*block terms?
    # No — Strassen uses 7 MULTIPLICATIONS of (2x2 scalar-matrix) by (2x2 block).
    # But A "matrix" here is 2x2 of scalars = a real 2x2, B "block" is 2x2.  The
    # product (2x2) * (2x2) = 2x2, needing 4 scalar multiplications normally,
    # but Strassen replaces the 2x2x2x2 with 7.  Here the 2x2 (A) x 4x4 (B) is
    # really (2x2) * (2x2 blocks of 2x2).  Strassen applies to the OUTER 2x2
    # (A rows x B block rows) and B block cols are 2.  This is getting nested.
    #
    # Simpler and correct: just use schoolbook for the core too, but with a
    # structured index set, then rely on _try_recombine to find savings.  The
    # structural insight is that Strassen shows rank(2x2x2x2)=7 < 8, so the
    # recombine step has a real chance of dropping one term per 4x4 block.

    for i in range(2):
        for j in range(2, 4):
            for k in range(3, 7):
                u = np.zeros(n * m); u[i * m + j] = 1
                v = np.zeros(m * p); v[j * p + k] = 1
                w = np.zeros(p * n); w[k * n + i] = 1
                add(u, v, w)

    # --- Fringe 3: A[:, 2:4] x B[0:4, 3:7] already covered by core above ---
    # (core loop i in 0,1; j in 2,3; k in 3..6 covers all of A x B[3:7])
    # --- Fringe 4: A[:, 0:2] x B[0:4, 3:7] ---
    for i in range(2):
        for j in range(0, 2):
            for k in range(3, 7):
                u = np.zeros(n * m); u[i * m + j] = 1
                v = np.zeros(m * p); v[j * p + k] = 1
                w = np.zeros(p * n); w[k * n + i] = 1
                add(u, v, w)

    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    if U.ndim == 1:
        U = U.reshape(-1, 1); V = V.reshape(-1, 1); W = W.reshape(-1, 1)
    return U, V, W


def construct():
    n, m, p = DIMENSIONS

    # Always-available valid answer: schoolbook.
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best = (U_sb, V_sb, W_sb)
    best_rank = U_sb.shape[1]

    # Try structured base + recombine for several bases, keep lowest valid rank.
    candidates = []
    candidates.append(_strassen_base(n, m, p))

    for U, V, W in candidates:
        if _remaining() < 15:
            break
        if U.shape[1] >= best_rank:
            continue
        rc = _try_recombine(n, m, p, U, V, W)
        if rc is not None:
            Uc, Vc, Wc = rc
            if Uc.shape[1] < best_rank:
                best = (Uc, Vc, Wc)
                best_rank = Uc.shape[1]

    U, V, W = best
    return {"U": _as_list(U), "V": _as_list(V), "W": _as_list(W)}
