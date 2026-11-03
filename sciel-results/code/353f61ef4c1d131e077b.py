"""Matrix multiplication 4,4,5 tensor decomposition.
Kronecker-Strassen rank-65 baseline + block 2x2x3 (rank 72) alternative,
then ALS refinement with half-integer rounding and term-merging to shave rank."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_terms():
    """7 Strassen 2x2x2 terms as sparse (index, value) lists for U, V, W.
    U idx: i*2+j ; V idx: j*2+k ; W idx: k*2+i
    """
    return [
        # M1 = (a+d)(e+h)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (c+d)e
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        # M3 = a(f-h)
        ([(0, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)]),
        # M4 = d(g-e)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        # M5 = (a+b)h
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)]),
        # M6 = (c-a)(e+f)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (b-d)(g+h)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]


def _verify_decomposition(U, V, W, n=4, m=4, p=5, tol=1e-7):
    """Check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U[i * m + j, :] * V[j * p + k, :], W[k * n + i, :])
                if abs(total - 1.0) > tol:
                    return False
    return True


def _verify_half_integer(U, V, W, tol=1e-9):
    """All entries must be multiples of 0.5."""
    for M in (U, V, W):
        M = np.asarray(M, dtype=np.float64)
        if np.any(np.abs(M * 2.0 - np.round(M * 2.0)) > tol):
            return False
    return True


def _build_schoolbook(n=4, m=4, p=5):
    """Rank n*m*p = 80 schoolbook."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _kron_strassen(n=4, m=4, p=5):
    """Rank-65: Kronecker of two 2x2x2 Strassen (4x4x4) + naive 5th column."""
    U_cols, V_cols, W_cols = [], [], []
    st = _strassen_2x2x2_terms()
    for (u1, v1, w1) in st:
        for (u2, v2, w2) in st:
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            for (idx1, val1) in u1:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2:
                    i2, j2 = idx2 // 2, idx2 % 2
                    u[(i1 * 2 + i2) * m + (j1 * 2 + j2)] = val1 * val2
            for (idx1, val1) in v1:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2:
                    j2, k2 = idx2 // 2, idx2 % 2
                    v[(j1 * 2 + j2) * p + (k1 * 2 + k2)] = val1 * val2
            for (idx1, val1) in w1:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2:
                    k2, i2 = idx2 // 2, idx2 % 2
                    w[(k1 * 2 + k2) * n + (i1 * 2 + i2)] = val1 * val2
            U_cols.append(u); V_cols.append(v); W_cols.append(w)
    # naive 5th column (k=4)
    for i in range(n):
        for j in range(m):
            u = np.zeros(n * m); u[i * m + j] = 1
            v = np.zeros(m * p); v[j * p + 4] = 1
            w = np.zeros(p * n); w[4 * n + i] = 1
            U_cols.append(u); V_cols.append(v); W_cols.append(w)
    return np.array(U_cols).T, np.array(V_cols).T, np.array(W_cols).T


def _block_strassen(n=4, m=4, p=5):
    """Rank-72: 2x2 blocks. C00,C10 use 2x2x2 Strassen (28 terms),
    C01,C11 use 2x2x3 (11 terms each, 44 total)."""
    U_cols, V_cols, W_cols = [], [], []
    st = _strassen_2x2x2_terms()

    def add(u, v, w):
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

    def s22(ai, aj, bj, bk):
        for (us, vs, ws) in st:
            u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
            for idx, val in us:
                ui, uj = idx // 2, idx % 2
                u[(ai + ui) * m + (aj + uj)] = val
            for idx, val in vs:
                vj, vk = idx // 2, idx % 2
                v[(bj + vj) * p + (bk + vk)] = val
            for idx, val in ws:
                wk, wi = idx // 2, idx % 2
                w[(bk + wk) * n + (ai + wi)] = val
            add(u, v, w)

    def s22x3(ai, aj, bj, bk):
        s22(ai, aj, bj, bk)
        for di in range(2):
            for dj in range(2):
                u = np.zeros(n * m); u[(ai + di) * m + (aj + dj)] = 1
                v = np.zeros(m * p); v[(bj + dj) * p + (bk + 2)] = 1
                w = np.zeros(p * n); w[(bk + 2) * n + (ai + di)] = 1
                add(u, v, w)

    # C00 = A00*B00 + A01*B10
    s22(0, 0, 0, 0); s22(0, 2, 2, 0)
    # C10 = A10*B00 + A11*B10
    s22(2, 0, 0, 0); s22(2, 2, 2, 0)
    # C01 (2x3) = A00*B01 + A01*B11
    s22x3(0, 0, 0, 2); s22x3(0, 2, 2, 2)
    # C11 (2x3) = A10*B01 + A11*B11
    s22x3(2, 0, 0, 2); s22x3(2, 2, 2, 2)

    return np.array(U_cols).T, np.array(V_cols).T, np.array(W_cols).T


def _snap_half_integer(M):
    return np.round(np.asarray(M, dtype=np.float64) * 2.0) / 2.0


def _als_refine(U, V, W, n=4, m=4, p=5, max_iters=300, tol=1e-9, deadline=None):
    """Alternating least squares: fix two factors, solve for the third via
    linear least squares, then snap to half-integers and verify.
    Returns (U,V,W) if a valid half-integer decomposition is found, else None.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]

    # Precompute index maps
    uidx = [(i * m + j) for i in range(n) for j in range(m)]
    vidx = [(j * p + k) for j in range(m) for k in range(p)]
    widx = [(k * n + i) for k in range(p) for i in range(n)]

    def try_snap_and_verify(Uc, Vc, Wc):
        Us, Vs, Ws = _snap_half_integer(Uc), _snap_half_integer(Vc), _snap_half_integer(Wc)
        if _verify_decomposition(Us, Vs, Ws, n, m, p, tol=1e-6):
            return Us, Vs, Ws
        # also try raw (non-snapped) in case entries are already near-exact
        if _verify_decomposition(Uc, Vc, Wc, n, m, p, tol=1e-6):
            return Uc, Vc, Wc
        return None

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        # Solve for W: for each (k,i), W[widx] . (U[uidx] * V[vidx]) = 1
        # Build system: rows = (k,i) pairs, cols = rank
        A = np.zeros((p * n, rank))
        b = np.ones(p * n)
        for k in range(p):
            for i in range(n):
                wrow = k * n + i
                for r in range(rank):
                    A[wrow, r] = U[i * m + 0, r]  # placeholder, replaced below
        # Build properly:
        A = np.zeros((p * n, rank))
        b = np.ones(p * n)
        for k in range(p):
            for i in range(n):
                wrow = k * n + i
                for r in range(rank):
                    # sum over j of U[i*m+j,r]*V[j*p+k,r]
                    s = 0.0
                    for j in range(m):
                        s += U[i * m + j, r] * V[j * p + k, r]
                    A[wrow, r] = s
        try:
            Wsol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            continue
        Wc = Wsol.T
        res = try_snap_and_verify(U, V, Wc)
        if res is not None:
            return res

        # Solve for U
        A = np.zeros((n * m, rank))
        b = np.ones(n * m)
        for i in range(n):
            for j in range(m):
                urow = i * m + j
                for r in range(rank):
                    s = 0.0
                    for k in range(p):
                        s += V[j * p + k, r] * W[k * n + i, r]
                    A[urow, r] = s
        try:
            Usol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            continue
        Uc = Usol.T
        res = try_snap_and_verify(Uc, V, W)
        if res is not None:
            return res

        # Solve for V
        A = np.zeros((m * p, rank))
        b = np.ones(m * p)
        for j in range(m):
            for k in range(p):
                vrow = j * p + k
                for r in range(rank):
                    s = 0.0
                    for i in range(n):
                        s += U[i * m + j, r] * W[k * n + i, r]
                    A[vrow, r] = s
        try:
            Vsol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            continue
        Vc = Vsol.T
        res = try_snap_and_verify(U, Vc, W)
        if res is not None:
            return res

        U, V, W = Uc, Vc, Wc

    return None


def _merge_zero_terms(U, V, W, tol=1e-9):
    """Remove columns that are zero in all three factors."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    keep = []
    for r in range(U.shape[1]):
        if (np.linalg.norm(U[:, r]) < tol and
                np.linalg.norm(V[:, r]) < tol and
                np.linalg.norm(W[:, r]) < tol):
            continue
        keep.append(r)
    return U[:, keep], V[:, keep], W[:, keep]


def _try_reduced_rank_als(n, m, p, target_rank, seed, max_iters=150, deadline=None):
    """Random init + ALS to try to hit target_rank."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, target_rank) * 0.5
    V = rng.randn(m * p, target_rank) * 0.5
    W = rng.randn(p * n, target_rank) * 0.5
    res = _als_refine(U, V, W, n, m, p, max_iters=max_iters, deadline=deadline)
    return res


def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    n, m, p = DIMENSIONS

    candidates = []

    # 1. Kronecker-Strassen rank-65 baseline (guaranteed)
    try:
        U, V, W = _kron_strassen(n, m, p)
        if _verify_decomposition(U, V, W, n, m, p):
            candidates.append((U.shape[1], U, V, W))
    except Exception:
        pass

    # 2. Block 2x2x3 rank-72 alternative
    try:
        U, V, W = _block_strassen(n, m, p)
        if _verify_decomposition(U, V, W, n, m, p):
            candidates.append((U.shape[1], U, V, W))
    except Exception:
        pass

    # 3. Schoolbook rank-80 fallback
    U, V, W = _build_schoolbook(n, m, p)
    candidates.append((U.shape[1], U, V, W))

    candidates.sort(key=lambda x: x[0])
    best_rank, best_U, best_V, best_W = candidates[0]

    # 4. ALS refinement on the best candidate to shave rank via snapping
    if time.time() < deadline - 20:
        try:
            refined = _als_refine(best_U, best_V, best_W, n, m, p,
                                  max_iters=200, deadline=deadline - 15)
            if refined is not None:
                rU, rV, rW = refined
                rU, rV, rW = _merge_zero_terms(rU, rV, rW)
                if rU.shape[1] < best_rank and _verify_decomposition(rU, rV, rW, n, m, p):
                    best_rank, best_U, best_V, best_W = rU.shape[1], rU, rV, rW
        except Exception:
            pass

    # 5. Try reduced-rank random ALS a few times if time remains
    if time.time() < deadline - 30:
        for target_rank in range(best_rank - 1, max(best_rank - 6, 40), -1):
            if time.time() > deadline - 20:
                break
            found = False
            for seed in [target_rank, target_rank + 7, target_rank + 13]:
                if time.time() > deadline - 15:
                    break
                try:
                    res = _try_reduced_rank_als(n, m, p, target_rank, seed,
                                                max_iters=120, deadline=deadline - 10)
                    if res is not None:
                        rU, rV, rW = res
                        rU, rV, rW = _merge_zero_terms(rU, rV, rW)
                        if rU.shape[1] <= target_rank and _verify_decomposition(rU, rV, rW, n, m, p):
                            best_rank, best_U, best_V, best_W = rU.shape[1], rU, rV, rW
                            found = True
                            break
                except Exception:
                    pass
            if found:
                break

    U, V, W = best_U, best_V, best_W
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
