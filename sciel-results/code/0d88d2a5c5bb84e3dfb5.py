"""Matrix multiplication 2x4 by 4x8: rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify_exact(U, V, W, T, tol_int=0):
    """Exact integer verification: T == sum_c U[:,c] V[:,c] W[:,c] (outer)."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    # Reconstruct using integer arithmetic via numpy int
    Ui = np.asarray(U, dtype=object)
    Vi = np.asarray(V, dtype=object)
    Wi = np.asarray(W, dtype=object)
    recon = np.zeros((n * m, m * p, p * n), dtype=object)
    for c in range(r):
        uc = Ui[:, c]
        vc = Vi[:, c]
        wc = Wi[:, c]
        recon += np.multiply.outer(uc, vc)[:, :, None] * wc[None, None, :]
    # compare
    diff = recon - T
    return bool(np.all(np.asarray(diff, dtype=object) == 0))

def _snap_to_halfint(M, grid=2):
    """Snap to multiples of 1/grid (here grid=2 -> half-integers)."""
    return np.round(np.asarray(M, dtype=float) * grid) / grid

def _try_rank_numeric(T, r, n, m, p, seed=0, iters=60):
    """Numerical search for rank-r decomposition, return snapped U,V,W or None."""
    rng = np.random.RandomState(seed)
    N = n * m
    M = m * p
    P = p * n
    best = None
    best_err = np.inf
    for attempt in range(3):
        U = rng.randn(N, r)
        V = rng.randn(M, r)
        W = rng.randn(P, r)
        # Normalize
        for c in range(r):
            s = max(np.linalg.norm(U[:, c]), np.linalg.norm(V[:, c]), np.linalg.norm(W[:, c]), 1e-12)
            U[:, c] /= s
            V[:, c] /= s
            W[:, c] *= s
        for it in range(iters):
            # Compute residual R = T - sum_c U[:,c] V[:,c] W[:,c]
            recon = np.einsum('ac,bc,dc->abd', U, V, W)
            R = T - recon
            # Gauss-Newton style: update each factor by projecting residual
            # dU[:,c] = (R * (V[:,c] W[:,c])) / (||V[:,c]||^2 ||W[:,c]||^2)
            for c in range(r):
                vc = V[:, c]; wc = W[:, c]
                nvc = np.dot(vc, vc); nwc = np.dot(wc, wc)
                if nvc < 1e-12 or nwc < 1e-12:
                    continue
                gradU = R * (vc[:, None] * wc[None, :]) / (nvc * nwc)
                U[:, c] += gradU
                # V
                uc = U[:, c]
                nuc = np.dot(uc, uc)
                if nwc < 1e-12 or nuc < 1e-12:
                    continue
                gradV = R * (uc[:, None] * wc[None, :]) / (nuc * nwc)
                V[:, c] += gradV
                # W
                uc = U[:, c]; vc = V[:, c]
                nvc = np.dot(vc, vc); nuc = np.dot(uc, uc)
                if nvc < 1e-12 or nuc < 1e-12:
                    continue
                gradW = R * (uc[:, None] * vc[None, :]) / (nuc * nvc)
                W[:, c] += gradW
            err = np.linalg.norm(R)
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
        if best is not None:
            U, V, W = best
        # snap
        Us = _snap_to_halfint(U)
        Vs = _snap_to_halfint(V)
        Ws = _snap_to_halfint(W)
        if _verify_exact(Us, Vs, Ws, T):
            return Us, Vs, Ws
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 110.0  # safety margin before 120s
    n, m, p = DIMENSIONS
    T = _build_tensor()
    school = _schoolbook()
    best = school
    best_rank = n * m * p

    # Try ranks from a reasonable lower bound upward
    # Lower bound: at least max(n*m, m*p, p*n) and at least something
    lb = max(n * m, m * p, p * n)
    # We'll try a handful of candidate ranks
    candidates = []
    for r in range(lb, best_rank):
        candidates.append(r)
    # Shuffle a bit but keep low ranks first; cap total attempts by time
    seed = 0
    for r in candidates:
        if time.time() > deadline - 5.0:
            break
        # limit attempts per rank by remaining time
        max_attempts = 4
        for a in range(max_attempts):
            if time.time() > deadline - 3.0:
                break
            res = _try_rank_numeric(T, r, n, m, p, seed=seed, iters=40)
            seed += 1
            if res is not None:
                Us, Vs, Ws = res
                U = [[float(x) for x in row] for row in Us]
                V = [[float(x) for x in row] for row in Vs]
                W = [[float(x) for x in row] for row in Ws]
                # convert to int where possible
                def to_int_if(x):
                    return int(x) if abs(x - round(x)) < 1e-9 else x
                U = [[to_int_if(x) for x in row] for row in U]
                V = [[to_int_if(x) for x in row] for row in V]
                W = [[to_int_if(x) for x in row] for row in W]
                best = {"U": U, "V": V, "W": W}
                best_rank = r
                # Found a lower rank; keep searching for even lower? break to save time
                break
        # if we found rank r, we could try to go lower, but continue loop naturally
    return best
# EVOLVE-BLOCK-END
