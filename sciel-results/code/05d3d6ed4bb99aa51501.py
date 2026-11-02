"""Program for matrix multiplication 3x4 * 4x6 with rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]
n, m, p = DIMENSIONS
DEADLINE = 110.0


def _build_rank60():
    """Build a rank-60 decomposition for 3x4 x 4x6.

    Split the 6 output columns into two blocks of 3.
    For each block: 27 terms for 3x3x3 schoolbook + 3 terms for 3x1x3.
    Total = 2 * (27 + 3) = 60.
    """
    R = 60
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]

    col = 0
    for block in range(2):
        k_base = block * 3
        for i in range(3):
            for j in range(3):
                for k_loc in range(3):
                    k = k_base + k_loc
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        for r in range(3):
            i = r
            k = k_base + r
            U[i * m + 3][col] = 1.0
            V[3 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1

    return {"U": U, "V": V, "W": W}


def _build_schoolbook():
    """Build the schoolbook rank-72 decomposition."""
    R = n * m * p
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1

    return {"U": U, "V": V, "W": W}


def _fast_verify(U, V, W, tol=1e-6):
    """Fast verification exploiting tensor sparsity.

    Only 72 nonzero entries (i,j,k) with value 1, and 360 zero entries.
    We check all 432 entries but with vectorized numpy.
    """
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    rank = U_np.shape[1]

    # Precompute V*W element-wise per (b,c) pair is expensive; instead do dot per entry
    # but vectorize over the 432 entries.
    # Build index arrays
    A_idx = np.array([i * m + j for i in range(n) for j in range(m) for k in range(p)])
    B_idx = np.array([j * p + k for i in range(n) for j in range(m) for k in range(p)])
    C_idx = np.array([k * n + i for i in range(n) for j in range(m) for k in range(p)])
    # target: 1 for consistent (i=j', k=k', i=i'), 0 otherwise
    # A_idx = i*m+j, B_idx = j'*p+k, C_idx = k'*n+i'
    # consistent iff j == j' and i == i' and k == k'
    i_arr = A_idx // m
    j_arr = A_idx % m
    jp_arr = B_idx // p
    k_arr = B_idx % p
    kn_arr = C_idx // n
    i2_arr = C_idx % n
    consistent = (j_arr == jp_arr) & (i_arr == i2_arr) & (k_arr == kn_arr)
    target = consistent.astype(np.float64)

    # Compute sum_r U[A,r] * V[B,r] * W[C,r] for all 432 entries
    U_sel = U_np[A_idx, :]  # (432, rank)
    V_sel = V_np[B_idx, :]  # (432, rank)
    W_sel = W_np[C_idx, :]  # (432, rank)
    vals = np.sum(U_sel * V_sel * W_sel, axis=1)  # (432,)
    if not np.all(np.abs(vals - target) <= tol):
        return False
    return True


def _round_to_half(U, V, W):
    """Round entries to nearest half-integer and verify."""
    U_r = np.round(np.array(U, dtype=np.float64) * 2.0) / 2.0
    V_r = np.round(np.array(V, dtype=np.float64) * 2.0) / 2.0
    W_r = np.round(np.array(W, dtype=np.float64) * 2.0) / 2.0
    if _fast_verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), tol=1e-6):
        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    return None


def _als(U, V, W, max_iter=500, start_time=0.0, deadline=110.0):
    """Alternating least squares. Returns (U, V, W) as numpy arrays."""
    def check_time():
        return time.time() - start_time < deadline

    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]

    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    for it in range(max_iter):
        if not check_time():
            break

        # Update U
        for a in range(n * m):
            i = a // m
            j = a % m
            M = np.zeros((p, rank))
            for k in range(p):
                b = j * p + k
                c = k * n + i
                M[k, :] = V[b, :] * W[c, :]
            try:
                U[a, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
            except Exception:
                pass

        # Update V
        for b in range(m * p):
            j = b // p
            k = b % p
            M = np.zeros((n, rank))
            for i in range(n):
                a = i * m + j
                c = k * n + i
                M[i, :] = U[a, :] * W[c, :]
            try:
                V[b, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
            except Exception:
                pass

        # Update W
        for c in range(p * n):
            k = c // n
            i = c % n
            M = np.zeros((m, rank))
            for j in range(m):
                a = i * m + j
                b = j * p + k
                M[j, :] = U[a, :] * V[b, :]
            try:
                W[c, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
            except Exception:
                pass

        # Loss
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = float(np.dot(U[a, :], V[b, :] * W[c, :]))
                    loss += (s - 1.0) ** 2
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        if loss < 1e-8:
            break

    return best_U, best_V, best_W


def _try_rank(rank, max_iter, seed, start_time, deadline):
    """Try to find a rank-r decomposition. Returns dict or None."""
    if time.time() - start_time >= deadline:
        return None
    np.random.seed(seed)
    U0 = np.random.randn(n * m, rank) * 0.5
    V0 = np.random.randn(m * p, rank) * 0.5
    W0 = np.random.randn(p * n, rank) * 0.5

    U, V, W = _als(U0, V0, W0, max_iter=max_iter, start_time=start_time, deadline=deadline)

    # Check direct
    if _fast_verify(U.tolist(), V.tolist(), W.tolist(), tol=1e-6):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try rounding to half-integers
    rounded = _round_to_half(U.tolist(), V.tolist(), W.tolist())
    if rounded is not None:
        return rounded

    # Try a short polish on the rounded values
    U2 = np.round(U * 2.0) / 2.0
    V2 = np.round(V * 2.0) / 2.0
    W2 = np.round(W * 2.0) / 2.0
    if time.time() - start_time < deadline:
        U3, V3, W3 = _als(U2, V2, W2, max_iter=min(200, max_iter // 3),
                          start_time=start_time, deadline=deadline)
        if _fast_verify(U3.tolist(), V3.tolist(), W3.tolist(), tol=1e-6):
            return {"U": U3.tolist(), "V": V3.tolist(), "W": W3.tolist()}
        rounded2 = _round_to_half(U3.tolist(), V3.tolist(), W3.tolist())
        if rounded2 is not None:
            return rounded2

    return None


def construct():
    start_time = time.time()

    def check_time():
        return time.time() - start_time < DEADLINE

    # Primary: rank-60 solution
    best = _build_rank60()
    if _fast_verify(best["U"], best["V"], best["W"], tol=1e-6):
        best_rank = 60
    else:
        best = _build_schoolbook()
        best_rank = 72

    # Search for lower rank: 45 to best_rank-1
    # Allocate time: ~2s per (rank, seed) attempt, 8 seeds per rank
    for r in range(45, best_rank):
        if not check_time():
            break
        for seed in range(8):
            if not check_time():
                break
            res = _try_rank(r, max_iter=300, seed=seed,
                            start_time=start_time, deadline=DEADLINE)
            if res is not None:
                if _fast_verify(res["U"], res["V"], res["W"], tol=1e-6):
                    best_rank = r
                    best = res
                    break

    return best
