import numpy as np
import time

DIMENSIONS = [3, 3, 5]


def _build_target_tensor(n, m, p):
    """T[a,b,c]=1 where a=i*m+j, b=j*p+k, c=k*n+i."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _verify(U, V, W, T, tol=1e-7):
    """Exact-ish check: max |sum_c U[a,c]V[b,c]W[c,c'] - T[a,b,c']| < tol."""
    P = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.max(np.abs(P - T)) < tol


def _round_halfint(M):
    """Round each entry to nearest half-integer."""
    return np.round(2.0 * M) / 2.0


def _schoolbook(n, m, p):
    r = n * m * p
    U = np.zeros((n * m, r), dtype=np.float64)
    V = np.zeros((m * p, r), dtype=np.float64)
    W = np.zeros((p * n, r), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, r


def _to_lists(U, V, W):
    """Convert to integer/half-integer lists for the interface."""
    Ur = _round_halfint(U)
    Vr = _round_halfint(V)
    Wr = _round_halfint(W)
    # Represent half-integers as floats; evaluator accepts integer/half-integer entries.
    U_l = [[float(x) for x in row] for row in Ur]
    V_l = [[float(x) for x in row] for row in Vr]
    W_l = [[float(x) for x in row] for row in Wr]
    return U_l, V_l, W_l


def _polish(U, V, W, T, n, m, p, deadline, max_iter=400, step=1.0):
    """Local polish: gradient-ish descent on the bilinear residual, with
    exact-verification gating. Returns improved (U,V,W) if a verified
    exact decomposition is found, else None."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    r = U.shape[1]
    best_err = np.inf
    best = (U.copy(), V.copy(), W.copy())
    for it in range(max_iter):
        if time.time() > deadline:
            break
        # Compute residual R[a,b,c] = sum_d U[a,d]V[b,d]W[c,d] - T[a,b,c]
        R = np.einsum('ad,bd,cd->abc', U, V, W) - T
        err = np.max(np.abs(R))
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        if err < 1e-9:
            break
        # Gradient w.r.t. U: gradU[a,d] = sum_{b,c} R[a,b,c] V[b,d] W[c,d]
        gradU = np.einsum('abc,bd,cd->ad', R, V, W)
        gradV = np.einsum('abc,ad,cd->bd', R, U, W)
        gradW = np.einsum('abc,ad,bd->cd', R, U, V)
        # Adaptive step via line search (backtracking)
        lr = step
        while lr > 1e-6:
            U2 = U - lr * gradU
            V2 = V - lr * gradV
            W2 = W - lr * gradW
            R2 = np.einsum('ad,bd,cd->abc', U2, V2, W2) - T
            err2 = np.max(np.abs(R2))
            if err2 < err * 0.999:
                U, V, W = U2, V2, W2
                err = err2
                break
            lr *= 0.5
        if lr <= 1e-6:
            break
    # Round and verify the best
    Ur, Vr, Wr = _round_halfint(best[0]), _round_halfint(best[1]), _round_halfint(best[2])
    if _verify(Ur, Vr, Wr, T, tol=1e-7):
        return Ur, Vr, Wr
    return None


def _structured_init(n, m, p, r, rng):
    """Initialize U,V,W of rank r by sampling from the schoolbook structure,
    then adding random sparse perturbations. This biases columns toward
    the true support pattern, giving the polish a warm start."""
    U = np.zeros((n * m, r), dtype=np.float64)
    V = np.zeros((m * p, r), dtype=np.float64)
    W = np.zeros((p * n, r), dtype=np.float64)
    for d in range(r):
        # Each column is a product of three sparse vectors.
        # Choose a random (i,j,k) triple as the base support.
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, m))
        k = int(rng.integers(0, p))
        U[i * m + j, d] = 1.0
        V[j * p + k, d] = 1.0
        W[k * n + i, d] = 1.0
        # Add small random half-integer noise on a few extra entries to
        # allow the polish to discover shared structure across columns.
        for _ in range(2):
            a = int(rng.integers(0, n * m))
            b = int(rng.integers(0, m * p))
            c = int(rng.integers(0, p * n))
            U[a, d] += rng.choice([-0.5, 0.5])
            V[b, d] += rng.choice([-0.5, 0.5])
            W[c, d] += rng.choice([-0.5, 0.5])
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin under 120s / 150s kill

    T = _build_target_tensor(n, m, p)

    # Guaranteed fallback
    U_s, V_s, W_s, r_s = _schoolbook(n, m, p)
    best_U, best_V, best_W, best_r = U_s, V_s, W_s, r_s

    # Search for a lower-rank exact decomposition.
    # Try a sequence of candidate ranks below schoolbook.
    rng = np.random.default_rng(12345)
    candidate_ranks = sorted(set([r_s - 5, r_s - 10, r_s - 15, r_s - 20]), reverse=True)
    for r in candidate_ranks:
        if r < 1:
            continue
        if time.time() > deadline - 15:
            break
        # Multiple random restarts at this rank.
        attempts = 0
        max_attempts = 6
        while attempts < max_attempts and time.time() < deadline - 15:
            attempts += 1
            U, V, W = _structured_init(n, m, p, r, rng)
            polished = _polish(U, V, W, T, n, m, p, deadline - 10)
            if polished is not None:
                pU, pV, pW = polished
                if r < best_r:
                    best_U, best_V, best_W, best_r = pU, pV, pW, r
                break  # found a valid one at this rank; try next (lower) rank
    
    # Convert to lists of floats to ensure compatibility with the evaluator's
    # interface expectations (integer/half-integer entries, standard Python types).
    U_list, V_list, W_list = _to_lists(best_U, best_V, best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
