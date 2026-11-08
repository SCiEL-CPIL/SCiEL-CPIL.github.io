import time
import numpy as np

DIMENSIONS = [4, 4, 4]
N, M, P = 4, 4, 4
TOL = 1e-6


def _validate_fast(U, V, W, n=N, m=M, p=P):
    """Vectorized check: sum_r W[k*n+i,r]*U[i*m+j,r]*V[j*p+k,r] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    # A[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    A = np.einsum('imr,jpr,knr->ijk', U, V, W)
    return bool(np.allclose(A, 1.0, atol=TOL))


def _validate_slow(U, V, W, n=N, m=M, p=P):
    """Fallback scalar validator (matches the original convention exactly)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    for ip in range(n):
        for kp in range(p):
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            expected = 1.0 if (ip == i and kp == k and jp == j) else 0.0
                            s = float(np.dot(W[kp * n + ip], U[i * m + j] * V[jp * p + k]))
                            if abs(s - expected) > 1e-6:
                                return False
    return True


def _strassen2():
    # 2x2 matmul rank-7 (Strassen). Indices: U[i*2+j], V[j*2+k], W[k*2+i].
    u2 = np.array([[1, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 1, 1, 1],
                   [1, 0, 0, 0, 1, 0, 0]], dtype=float)
    v2 = np.array([[1, 0, 1, 0, 0, 1, 0],
                   [0, 0, 0, 1, 1, 0, 0],
                   [1, 0, 0, 0, 0, 0, 1],
                   [0, 0, 1, 0, 1, 0, 0]], dtype=float)
    w2 = np.array([[1, 0, 0, 0, 0, 1, 1],
                   [1, 0, 0, 0, 0, 0, 1],
                   [1, 0, 1, 0, 1, 0, 0],
                   [0, 1, 1, 1, 0, 0, 0]], dtype=float)
    return u2, v2, w2


def _schoolbook(n=N, m=M, p=P):
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
    return {"U": U, "V": V, "W": W}


def _kronecker_strassen4():
    """4x4 = 2x2 Kronecker of 2x2 Strassen. Rank 49.
    Uses (M⊗N)(A⊗B) = (MA)⊗(NB) with M4[iH*2+iL, jH*2+jL] = Mh[iH,jH]*Ml[iL,jL]."""
    u2, v2, w2 = _strassen2()
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    for iH in range(2):
        for iL in range(2):
            for jH in range(2):
                for jL in range(2):
                    rowU = 8 * iH + 4 * iL + 2 * jH + jL
                    for r1 in range(7):
                        for r2 in range(7):
                            R = r1 * 7 + r2
                            U4[rowU, R] = u2[iH * 2 + jH, r1] * u2[iL * 2 + jL, r2]
    for jH in range(2):
        for jL in range(2):
            for kH in range(2):
                for kL in range(2):
                    rowV = 8 * jH + 4 * jL + 2 * kH + kL
                    for r1 in range(7):
                        for r2 in range(7):
                            R = r1 * 7 + r2
                            V4[rowV, R] = v2[jH * 2 + kH, r1] * v2[jL * 2 + kL, r2]
    for kH in range(2):
        for kL in range(2):
            for iH in range(2):
                for iL in range(2):
                    rowW = 8 * kH + 4 * kL + 2 * iH + iL
                    for r1 in range(7):
                        for r2 in range(7):
                            R = r1 * 7 + r2
                            W4[rowW, R] = w2[kH * 2 + iH, r1] * w2[kL * 2 + iL, r2]
    return {"U": U4, "V": V4, "W": W4}


def _try_reduce_rank(U, V, W, target_rank, max_trials=2000, seed=0):
    """Try to find a rank-`target_rank` sub-decomposition by adding a random combination
    of `target_rank` existing columns to each of the other columns (Gaussian elimination)."""
    rng = np.random.default_rng(seed)
    n = U.shape[1]
    for _ in range(max_trials):
        idx = rng.choice(n, size=target_rank, replace=False)
        keep = set(idx.tolist())
        # For each dropped column d, express as combination of kept columns using random weights
        # Actually: new column d' = col_d + sum_{c in keep} w_c * col_c, then zero out d.
        # This is a valid rank-preserving column operation only if we keep all columns;
        # to REDUCE rank we need the tensor to lie in a subspace. Instead we test:
        # project: set dropped columns to 0 and see if still valid (only works if redundant).
        U2 = U.copy(); V2 = V.copy(); W2 = W.copy()
        for d in range(n):
            if d in keep:
                continue
            # zero out column d (only valid if it was a linear combination that contributes 0 net)
            U2[:, d] = 0; V2[:, d] = 0; W2[:, d] = 0
        if _validate_fast(U2, V2, W2):
            cols = [c for c in range(n) if c in keep]
            return {"U": U2[:, cols], "V": V2[:, cols], "W": W2[:, cols]}
    return None


def _random_block_search(deadline, best_rank):
    """Combinatorial search: 4x4 as 2x2 blocks of 2x2. Try all 3^4 assignments of each
    block to {direct, strassen} and random Strassen-like 3-term patterns, keep best valid."""
    rng = np.random.default_rng(12345)
    best = None
    u2, v2, w2 = _strassen2()

    def build_and_check(U, V, W):
        if not _validate_fast(U, V, W):
            return None
        # prune zero columns
        nz = [c for c in range(U.shape[1]) if (U[:, c] != 0).any() and (V[:, c] != 0).any() and (W[:, c] != 0).any()]
        if len(nz) < U.shape[1]:
            U2, V2, W2 = U[:, nz], V[:, nz], W[:, nz]
            if _validate_fast(U2, V2, W2):
                return {"U": U2, "V": V2, "W": W2}
        return {"U": U, "V": V, "W": W}

    # Direct 2x2 rank-4 blocks
    def direct2():
        u = np.zeros((4, 4)); v = np.zeros((4, 4)); w = np.zeros((4, 4))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u[i*2+j, i*2+j*2+k - (j*2+k) + 0] = 0  # placeholder
        # build properly: rank 4, index r = i*4 + j*2 + k? simpler:
        u = np.zeros((4, 4)); v = np.zeros((4, 4)); w = np.zeros((4, 4))
        r = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u[i*2+j, r] = 1; v[j*2+k, r] = 1; w[k*2+i, r] = 1; r += 1
        return u, v, w

    # We'll just attempt Kronecker variants with random column merging and return if better
    base = _kronecker_strassen4()
    U, V, W = base["U"], base["V"], base["W"]
    for trial in range(40):
        if time.time() > deadline:
            break
        cand = _try_reduce_rank(U, V, W, target_rank=min(48, best_rank - 1), max_trials=300, seed=trial)
        if cand is not None:
            r = cand["U"].shape[1]
            if best is None or r < best["U"].shape[1]:
                best = cand
    return best


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    DEADLINE = start + 140  # safety margin before 150s

    def remaining():
        return DEADLINE - time.time()

    # Fallbacks in increasing rank
    candidates = []

    # 1) Kronecker of Strassen -> rank 49 (fast, proven)
    try:
        k = _kronecker_strassen4()
        if _validate_fast(k["U"], k["V"], k["W"]):
            candidates.append((k["U"].shape[1], k))
    except Exception as e:
        print("kron err", e)

    # 2) Schoolbook -> rank 64 (always valid)
    sb = _schoolbook(n, m, p)
    if _validate_fast(sb["U"], sb["V"], sb["W"]):
        candidates.append((sb["U"].shape[1], sb))

    if not candidates:
        # last resort: return schoolbook regardless
        U, V, W = sb["U"], sb["V"], sb["W"]
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    candidates.sort(key=lambda x: x[0])
    best_rank, best = candidates[0]
    print("initial best rank:", best_rank)

    # 3) Attempt rank reduction below best_rank while time remains
    if remaining() > 25:
        U, V, W = best["U"], best["V"], best["W"]
        for target in range(best_rank - 1, max(best_rank - 6, 40), -1):
            if remaining() < 12:
                break
            cand = _try_reduce_rank(U, V, W, target, max_trials=1500, seed=target)
            if cand is not None:
                r = cand["U"].shape[1]
                if r < best_rank:
                    best_rank, best = r, cand
                    print("reduced to rank", r)
                    break

    if remaining() > 15:
        rb = _random_block_search(DEADLINE - 8, best_rank)
        if rb is not None and rb["U"].shape[1] < best_rank:
            best_rank, best = rb["U"].shape[1], rb
            print("block search rank", best_rank)

    # Final safety
    if time.time() > DEADLINE - 3:
        pass

    U, V, W = best["U"], best["V"], best["W"]
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
