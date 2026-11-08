"""Fresh design: verified Strassen block decomposition + local rank refinement."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 4, 7]

# ---- deadline guard -------------------------------------------------------
_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 12.0          # safety margin before the hard kill

def _time_left():
    return _LIMIT - _MARGIN - (time.time() - _T0)

# ---------------------------------------------------------------------------
# 1) Build the exact matrix-multiplication tensor T[i,j,k] = 1
#    with index mapping  U-index = i*m+j, V-index = j*p+k, W-index = k*n+i
# ---------------------------------------------------------------------------
def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify(U, V, W, T, tol=1e-6):
    """Return True iff the decomposition reproduces T exactly (within tol)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    # reconstruct tensor via einsum
    R = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.allclose(R, T, atol=tol)

# ---------------------------------------------------------------------------
# 2) Strassen block decomposition
#    A (n x m) = [[A00, A01],[A10, A11]]   (2 x 2 blocks, each (n/2)x(m/2))
#    B (m x p) = [[B00, B01],[B10, B11]]
#    C (n x p) = [[C00, C01],[C10, C11]]
#    C00 = M1 - M2 + M3 + M4   C01 = M5 + M1 - M4 - M7
#    C10 = M2 + M6             C11 = M1 - M3 + M5 + M7
#    M1=(A00+A11)(B00+B11)  M2=(A10+A11)B00  M3=A00(B01-B11)
#    M4=A11(B10-B00)        M5=(A00+A01)B11 M6=(A10-A00)(B00+B01)
#    M7=(A01-A11)(B10+B11)
# ---------------------------------------------------------------------------
def _strassen_factors(n, m, p):
    n2, m2 = n // 2, m // 2
    # block row/col index helpers
    def uidx(bi, bj, i, j):      # bi in {0,1} block-row, bj in {0,1} block-col
        return (bi * n2 + i) * m + (bj * m2 + j)
    def vidx(bj, bk, j, k):      # bj block-col of B, bk block-row of B
        return (bj * m2 + j) * p + (bk * (p // 2) + k)
    def widx(bk, bi, k, i):      # bk block-row of C, bi block-col of C
        return (bk * (p // 2) + k) * n + (bi * n2 + i)

    nA = n2 * m2
    nB = m2 * (p // 2)
    nC = (p // 2) * n2
    rank = 7 * nA   # M1,M2,M3,M4,M5,M6,M7 each over nA positions; M6,M7 use A10/A01 (n2 x m2)
    # We will build per-position terms. Each M term is a sum over its A-block and B-block
    # positions; we allocate one rank column per (M, a-position, b-position).
    U = {}
    V = {}
    W = {}
    col = 0
    def add_term(uidx, vidx, widx, a_val, b_val, c_val):
        nonlocal col
        U.setdefault(uidx, []).append(a_val)
        V.setdefault(vidx, []).append(b_val)
        W.setdefault(widx, []).append(c_val)
        col += 1

    # For each M we iterate over the relevant A and B block positions.
    # M1: A00+A11 (block (0,0)+(1,1))  times B00+B11 (block (0,0)+(1,1))
    # We expand (A00+A11)*(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # Each cross term is a separate bilinear form (product of a single A pos and single B pos).
    def expand(A_blocks, B_blocks, c_sign):
        # A_blocks: list of (bi,bj) ; B_blocks: list of (bk,bk2)
        for (abi, abj) in A_blocks:
            for i in range(n2):
                for j in range(m2):
                    for (bbk, bbl) in B_blocks:
                        for k in range(p // 2):
                            add_term(uidx(abi, abj, i, j),
                                     vidx(bbl, bbk, j, k),
                                     widx(bbk, abi, k, i),
                                     1, 1, c_sign)

    # M1 = (A00+A11)(B00+B11): contributes +1 to C00 and +1 to C01 and -1 to C11
    # We need to track per-output-block contributions. Instead of expand() we build
    # a list of (A_blocks, B_blocks) for each M, then for each M add its contribution
    # to each output block with the right sign.
    M = {
        1: ([(0, 0), (1, 1)], [(0, 0), (1, 1)]),
        2: ([(1, 0), (1, 1)], [(0, 0)]),
        3: ([(0, 0)], [(0, 1), (1, 1)]),
        4: ([(1, 1)], [(1, 0), (0, 0)]),
        5: ([(0, 0), (0, 1)], [(1, 1)]),
        6: ([(1, 0), (0, 0)], [(0, 0), (0, 1)]),
        7: ([(0, 1), (1, 1)], [(1, 0), (1, 1)]),
    }
    # output-block signs: C00, C01, C10, C11
    signs = {
        1: (1, 1, 0, -1),
        2: (-1, 0, 1, 0),
        3: (1, 0, 0, 0),
        4: (1, 0, 0, 0),
        5: (0, 1, 0, 1),
        6: (0, 0, 1, 0),
        7: (0, -1, 0, 1),
    }
    # output block (bi,bk) -> widx uses (bk, bi)
    out_blocks = [(0, 0), (0, 1), (1, 0), (1, 1)]  # (bi, bk)

    # Build as lists of (a_pos, b_pos, c_sign) triples; a_pos=(bi,bj,i,j), b_pos=(bk,bk2,j,k)
    # We must be careful: for M with multiple A blocks and multiple B blocks, the product
    # (sum A)*(sum B) expands to all cross terms, each a bilinear form.
    for midx, (Ab, Bb) in M.items():
        s00, s01, s10, s11 = signs[midx]
        for (abi, abj) in Ab:
            for i in range(n2):
                for j in range(m2):
                    for (bbk, bbl) in Bb:
                        for k in range(p // 2):
                            # A position
                            ai = uidx(abi, abj, i, j)
                            # B position: B block (bbl, bbk) at (j, k)
                            bi = vidx(bbl, bbk, j, k)
                            # C contributions
                            for (ob_i, ob_k, sgn) in ((0,0,s00),(0,1,s01),(1,0,s10),(1,1,s11)):
                                if sgn == 0:
                                    continue
                                ci = widx(ob_k, ob_i, k, i)
                                U.setdefault(ai, []).append(1)
                                V.setdefault(bi, []).append(1)
                                W.setdefault(ci, []).append(sgn)
                                col += 1

    # Convert dict-of-lists to dense matrices of width col
    Umat = np.zeros((n * m, col), dtype=int)
    Vmat = np.zeros((m * p, col), dtype=int)
    Wmat = np.zeros((p * n, col), dtype=int)
    for idx, vals in U.items():
        Umat[idx, :len(vals)] = vals
    for idx, vals in V.items():
        Vmat[idx, :len(vals)] = vals
    for idx, vals in W.items():
        Wmat[idx, :len(vals)] = vals
    return Umat, Vmat, Wmat

# ---------------------------------------------------------------------------
# 3) Local refinement: try to drop redundant rank columns
#    A column c is redundant if there exist integer/half-integer combos that
#    reproduce its effect. Simplest safe test: if a column is exactly a linear
#    combination (over the reconstruction) of others, drop it. We use a
#    conservative check: attempt to express column c as an integer combination
#    of a small subset of other columns by solving a linear system.
# ---------------------------------------------------------------------------
def _try_drop_column(U, V, W, c, T, n, m, p):
    """Return True if column c can be removed while preserving the decomposition."""
    r = U.shape[1]
    if r <= 1:
        return False
    # The tensor contribution of column c is U[:,c] outer V[:,c] outer W[:,c].
    # We want to find coefficients alpha (length r-1, excluding c) s.t.
    #   sum_{d != c} alpha_d * U[:,d] o V[:,d] o W[:,d] == U[:,c] o V[:,c] o W[:,c]
    # This is a linear system in alpha. Build via flattening.
    Uf = U[:, c]  # shape (nm,)
    Vf = V[:, c]
    Wf = W[:, c]
    target = np.einsum('a,b,d->abd', Uf, Vf, Wf)
    target_flat = target.ravel()
    # Build matrix A where column d (d != c) is the flattened tensor of column d
    others = [d for d in range(r) if d != c]
    A = np.zeros((target.size, len(others)), dtype=float)
    for j, d in enumerate(others):
        A[:, j] = np.einsum('a,b,e->abe', U[:, d], V[:, d], W[:, d]).ravel()
    # Solve least squares; check residual
    try:
        sol, res, rankA, sv = np.linalg.lstsq(A, target_flat, rcond=None)
    except np.linalg.LinAlgError:
        return False
    recon = A @ sol
    if not np.allclose(recon, target_flat, atol=1e-6):
        return False
    # Verify full reconstruction after removal
    newU = np.delete(U, c, axis=1)
    newV = np.delete(V, c, axis=1)
    newW = np.delete(W, c, axis=1)
    # adjust: we removed column c but kept others; the alpha combination must be applied.
    # Actually dropping column c and keeping others unchanged only works if column c
    # equals the sum of the *existing* other columns (alpha already encoded). But we
    # did not modify the other columns, so we must verify the decomposition still holds.
    # The correct operation: new columns = others, but we need the alpha to be baked in.
    # Since we cannot change entries, the only safe drop is when alpha_d is 0 for all d
    # except those where we can merge. This is complex; we only accept a drop if the
    # residual is zero AND the combination coefficients are such that we can represent
    # them by modifying kept columns. To stay safe, we only drop when column c is exactly
    # equal (coefficient 1) to a single other column and 0 elsewhere.
    if np.count_nonzero(np.abs(sol) > 1e-9) != 1:
        return False
    d = others[int(np.argmax(np.abs(sol)))]
    if not np.allclose(sol, np.where(np.arange(len(others)) == np.argmax(np.abs(sol)), sol[np.argmax(np.abs(sol))], 0), atol=1e-9):
        pass
    # Verify by full reconstruction with column c removed (others unchanged)
    R = np.einsum('ac,bc,dc->abd', newU, newV, newW)
    return np.allclose(R, T, atol=1e-6)

def _refine(U, V, W, T, n, m, p, max_iters=200):
    """Iteratively try to drop redundant columns."""
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    improved = True
    iters = 0
    while improved and iters < max_iters and _time_left() > 5.0:
        improved = False
        iters += 1
        r = U.shape[1]
        for c in range(r):
            if _time_left() < 5.0:
                break
            if _try_drop_column(U, V, W, c, T, n, m, p):
                U = np.delete(U, c, axis=1)
                V = np.delete(V, c, axis=1)
                W = np.delete(W, c, axis=1)
                improved = True
                break
    return U, V, W

# ---------------------------------------------------------------------------
# 4) Integer/half-integer normalization
# ---------------------------------------------------------------------------
def _to_rational(U, V, W):
    """Return integer (or half-integer) versions. Strassen already uses ints."""
    return U.astype(int), V.astype(int), W.astype(int)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)

    # Baseline: schoolbook (always valid fallback)
    rank = n * m * p
    U0 = [[0] * rank for _ in range(n * m)]
    V0 = [[0] * rank for _ in range(m * p)]
    W0 = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j][col] = 1
                V0[j * p + k][col] = 1
                W0[k * n + i][col] = 1
                col += 1
    best = (rank, U0, V0, W0)

    # Strassen
    try:
        if _time_left() > 5.0:
            Us, Vs, Ws = _strassen_factors(n, m, p)
            if _verify(Us, Vs, Ws, T):
                r_s = Us.shape[1]
                if r_s < best[0]:
                    best = (r_s, Us, Vs, Ws)
    except Exception:
        pass

    # Local refinement on the best candidate
    try:
        if _time_left() > 10.0:
            r_b, Ub, Vb, Wb = best
            Ur, Vr, Wr = _refine(np.asarray(Ub, float), np.asarray(Vb, float),
                                 np.asarray(Wb, float), T, n, m, p)
            if _verify(Ur, Vr, Wr, T):
                r_r = Ur.shape[1]
                if r_r < r_b:
                    best = (r_r, Ur, Vr, Wr)
    except Exception:
        pass

    r, U, V, W = best
    U, V, W = _to_rational(np.asarray(U), np.asarray(V), np.asarray(W))
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
