"""Rewrite: structured sparse-support + linear solve + half-integer snapping."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

DEADLINE_MARGIN = 10.0  # seconds to keep before hard kill at 150s
_START_TIME = None


def _elapsed():
    return time.time() - _START_TIME


def _build_tensor(n, m, p):
    """Tensor T[i,j,k,l] such that T[i,j,k,l] = 1 if j==k else 0.
    This encodes (A B)[i,l] = sum_j A[i,j] B[j,l].
    Output factor uses official trace convention: W index is k*n + i.
    """
    T = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k] = 1.0
    return T


def _reconstruct(U, V, W, n, m, p, rank):
    """Reconstruct tensor from factors.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank).
    Trace convention: W index = k*n + i.
    T[i,j,j',k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    recon = np.zeros((n, m, m, p))
    for c in range(rank):
        u4 = U[:, c].reshape(n, m)
        v4 = V[:, c].reshape(m, p)
        w4 = W[:, c].reshape(p, n)
        recon += (u4[:, :, np.newaxis, np.newaxis] *
                  v4[np.newaxis, np.newaxis, :, :] *
                  w4.T[np.newaxis, np.newaxis, :, :])
    return recon


def _snap_halfinteger(X, tol=1e-6):
    """Snap entries to nearest half-integer if within tol."""
    Y = np.round(X * 2.0) / 2.0
    if np.max(np.abs(X - Y)) <= tol:
        return Y, True
    return Y, False


def _schoolbook(n, m, p):
    """Trivial rank n*m*p decomposition."""
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


def _supports_for_size(size, rank, max_nonzero_per_row=2):
    """Yield support masks for a (size, rank) factor.
    A support mask is a boolean (size, rank) array.
    To keep the search tractable we constrain each row to have at most
    `max_nonzero_per_row` nonzeros, but we generate them via a randomized
    greedy process rather than full enumeration (which is combinatorial).
    """
    yield None  # placeholder; actual generation is done by caller with seeds


def _random_support(size, rank, rng, density=0.4):
    """Random sparse support: each entry included with prob `density`,
    but guarantee each column has at least one nonzero (needed for identifiability)."""
    S = rng.random((size, rank)) < density
    # Ensure each column has >=1 nonzero
    for c in range(rank):
        if not S[:, c].any():
            S[rng.randint(size), c] = True
    return S


def _solve_linear_for_support(U_S, V_S, W_S, T_flat, n, m, p, rank):
    """Given boolean supports U_S (n*m,rank), V_S (m*p,rank), W_S (p*n,rank),
    solve the linear system for the entries consistent with the supports.
    We build the linear map L: params -> T, where params are only the
    supported entries (others forced to 0). Use least squares.
    """
    um = n * m
    vp = m * p
    pn = p * n

    # Determine total number of free params
    nU = int(np.count_nonzero(U_S))
    nV = int(np.count_nonzero(V_S))
    nW = int(np.count_nonzero(W_S))
    nparams = nU + nV + nW

    # Flatten T to length n*m*m*p with index order (i,j,j',k)
    eq_dim = n * m * m * p

    # Build A (eq_dim x nparams) sparsely-ish via dense (small sizes)
    A = np.zeros((eq_dim, nparams))

    # Map each free param to a (factor, row, col)
    U_idx = {}  # (row,col) -> param index
    pi = 0
    for r in range(um):
        for c in range(rank):
            if U_S[r, c]:
                U_idx[(r, c)] = pi
                pi += 1
    V_idx = {}
    for r in range(vp):
        for c in range(rank):
            if V_S[r, c]:
                V_idx[(r, c)] = pi
                pi += 1
    W_idx = {}
    for r in range(pn):
        for c in range(rank):
            if W_S[r, c]:
                W_idx[(r, c)] = pi
                pi += 1
    assert pi == nparams

    # For each output (i, j, k) [note: only T[i,j,j,k] is nonzero in target,
    # but the tensor has all (i,j,j',k); we must match the full tensor which
    # is zero for j' != j].
    for i in range(n):
        for j in range(m):
            for jp in range(m):  # inner index position (called j' in tensor)
                for k in range(p):
                    eq = (i * m + j) * (m * p) + (jp * p + k)
                    target = 1.0 if (jp == j) else 0.0
                    # T[i,j,jp,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                    # This is bilinear in params, NOT linear! 
                    # We handle linearity by fixing two factors and solving the third.
                    # For a fully linear system we'd need rank-1 structure.
                    pass
    # Because the system is trilinear (each equation is a product of one entry
    # from each factor), a single linear solve is not directly possible unless
    # we fix two factors. We therefore use alternating linearization instead.
    return None, None, None


def _alternating_solve(U_S, V_S, W_S, T, n, m, p, rank, rng, max_iter=200):
    """Alternating least squares: fix two factors, solve the third linearly,
    respecting the supports (unsupported entries = 0)."""
    um, vp, pn = n * m, m * p, p * n
    T_flat = T.reshape(-1)  # order (i,j,jp,k)

    # Initialize supported entries randomly
    U = np.where(U_S, rng.standard_normal((um, rank)) * 0.5, 0.0)
    V = np.where(V_S, rng.standard_normal((vp, rank)) * 0.5, 0.0)
    W = np.where(W_S, rng.standard_normal((pn, rank)) * 0.5, 0.0)

    def residual():
        recon = _reconstruct(U, V, W, n, m, p, rank)
        return np.linalg.norm(recon.reshape(-1) - T_flat)

    best_res = np.inf
    best = (U.copy(), V.copy(), W.copy())

    for it in range(max_iter):
        # Solve for U given V, W: linear in U entries
        # T[i,j,jp,k] = sum_c U[i*m+j,c] * (V[j*p+k,c]*W[k*n+i,c])
        # For each row r = i*m+j of U, the equation involves that row only.
        # Build per-row linear system.
        for i in range(n):
            for j in range(m):
                r = i * m + j
                # unknowns: U[r, c] for c where U_S[r,c]
                cols = [c for c in range(rank) if U_S[r, c]]
                if not cols:
                    continue
                # equations: for each (jp,k)
                Aeq = np.zeros((m * p, len(cols)))
                b = np.zeros(m * p)
                for jp in range(m):
                    for k in range(p):
                        eq = jp * p + k
                        b[eq] = 1.0 if (jp == j) else 0.0
                        for cc, c in enumerate(cols):
                            Aeq[eq, cc] = V[j * p + k, c] * W[k * n + i, c]
                sol, *_ = np.linalg.lstsq(Aeq, b, rcond=None)
                for cc, c in enumerate(cols):
                    U[r, c] = sol[cc]

        # Solve for V given U, W
        for j in range(m):
            for k in range(p):
                r = j * p + k
                cols = [c for c in range(rank) if V_S[r, c]]
                if not cols:
                    continue
                Aeq = np.zeros((n * m, len(cols)))
                b = np.zeros(n * m)
                for i in range(n):
                    for jp in range(m):
                        eq = i * m + jp
                        b[eq] = 1.0 if (jp == j) else 0.0
                        for cc, c in enumerate(cols):
                            Aeq[eq, cc] = U[i * m + jp, c] * W[k * n + i, c]
                sol, *_ = np.linalg.lstsq(Aeq, b, rcond=None)
                for cc, c in enumerate(cols):
                    V[r, c] = sol[cc]

        # Solve for W given U, V
        for k in range(p):
            for i in range(n):
                r = k * n + i
                cols = [c for c in range(rank) if W_S[r, c]]
                if not cols:
                    continue
                Aeq = np.zeros((m * m, len(cols)))
                b = np.zeros(m * m)
                for j in range(m):
                    for jp in range(m):
                        eq = j * m + jp
                        b[eq] = 1.0 if (jp == j) else 0.0
                        for cc, c in enumerate(cols):
                            Aeq[eq, cc] = U[i * m + j, c] * V[jp * p + k, c]
                sol, *_ = np.linalg.lstsq(Aeq, b, rcond=None)
                for cc, c in enumerate(cols):
                    W[r, c] = sol[cc]

        res = residual()
        if res < best_res:
            best_res = res
            best = (U.copy(), V.copy(), W.copy())
        if res < 1e-9:
            break

    U, V, W = best
    # Snap
    Us, okU = _snap_halfinteger(U)
    Vs, okV = _snap_halfinteger(V)
    Ws, okW = _snap_halfinteger(W)
    if okU and okV and okW:
        recon = _reconstruct(Us, Vs, Ws, n, m, p, rank)
        if np.allclose(recon.reshape(-1), T_flat, atol=1e-9):
            return Us, Vs, Ws
    # Also try snapping the unsnapped solution and checking
    recon = _reconstruct(U, V, W, n, m, p, rank)
    if np.allclose(recon.reshape(-1), T_flat, atol=1e-9):
        Us, okU = _snap_halfinteger(U)
        Vs, okV = _snap_halfinteger(V)
        Ws, okW = _snap_halfinteger(W)
        if okU and okV and okW:
            recon2 = _reconstruct(Us, Vs, Ws, n, m, p, rank)
            if np.allclose(recon2.reshape(-1), T_flat, atol=1e-9):
                return Us, Vs, Ws
    return None


def _try_rank(rank, n, m, p, T, n_attempts=40, density=0.5):
    um, vp, pn = n * m, m * p, p * n
    T_flat = T.reshape(-1)
    for attempt in range(n_attempts):
        if _elapsed() > DEADLINE_MARGIN:
            return None
        seed = 12345 + attempt * 7919
        rng = np.random.default_rng(seed)
        # Random sparse supports
        U_S = _random_support(um, rank, rng, density=density)
        V_S = _random_support(vp, rank, rng, density=density)
        W_S = _random_support(pn, rank, rng, density=density)
        res = _alternating_solve(U_S, V_S, W_S, T, n, m, p, rank, rng, max_iter=120)
        if res is not None:
            return res
    return None


def _structured_greedy(rank, n, m, p, T):
    """A more structured attempt: initialize factors with a near-schoolbook
    pattern and then do ALS with dense supports, hoping to discover a
    lower-rank exact half-integer solution via the algebraic structure."""
    um, vp, pn = n * m, m * p, p * n
    T_flat = T.reshape(-1)
    for attempt in range(20):
        if _elapsed() > DEADLINE_MARGIN:
            return None
        seed = 98765 + attempt * 104729
        rng = np.random.default_rng(seed)
        U = rng.standard_normal((um, rank)) * 0.3
        V = rng.standard_normal((vp, rank)) * 0.3
        W = rng.standard_normal((pn, rank)) * 0.3
        U_S = np.ones((um, rank), dtype=bool)
        V_S = np.ones((vp, rank), dtype=bool)
        W_S = np.ones((pn, rank), dtype=bool)
        res = _alternating_solve(U_S, V_S, W_S, T, n, m, p, rank, rng, max_iter=150)
        if res is not None:
            return res
    return None


def construct():
    global _START_TIME
    _START_TIME = time.time()
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)

    # Try ranks from the theoretical lower bound up to schoolbook-1.
    # Lower bound: at least max(n*m, m*p, p*n) = 18? No, rank can be below that
    # in general (e.g., Strassen). For 2x3x3, the known minimum is unknown;
    # try from a reasonable lower bound.
    # A simple lower bound: rank >= max(n*p, n*m, m*p) is NOT valid; use
    # rank >= n*p = 6 and rank >= n*m = 6 and rank >= m*p = 9 -> rank >= 9.
    # Actually a valid bound: rank >= max(n*m, m*p, p*n) is false. Use
    # rank >= n*p = 6. We'll start at 6.
    lo = n * p  # = 6
    hi = n * m * p  # = 18 schoolbook

    for rank in range(lo, hi):
        if _elapsed() > DEADLINE_MARGIN:
            break
        # Structured greedy first (dense supports), then sparse random
        res = _structured_greedy(rank, n, m, p, T)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        if _elapsed() > DEADLINE_MARGIN:
            break
        res = _try_rank(rank, n, m, p, T)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook rank 18
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
