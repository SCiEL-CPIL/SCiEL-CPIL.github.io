"""Matrix multiplication 3x3 by 3x6: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

N_DIM, M_DIM, P_DIM = 3, 3, 6


def _build_support(n, m, p):
    """Build on-support triples and index groups."""
    nm, mp, pn = n * m, m * p, p * n
    on = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                on.append((a, b, c))
    on = np.array(on, dtype=np.int64)
    return on


def _verify(U, V, W, n, m, p):
    """Exact verification of the decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != nm or V.shape[0] != mp or W.shape[0] != pn:
        return False

    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-8:
                    return False

    # Check off-support
    for a in range(nm):
        ia, ja = a // m, a % m
        for b in range(mp):
            jb, kb = b // p, b % p
            if ja != jb:
                continue
            for c in range(pn):
                kc, ic = c // n, c % n
                if kb == kc and ic == ia:
                    continue  # on-support
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val) > 1e-8:
                    return False
    return True


def _schoolbook(n, m, p):
    """Guaranteed rank n*m*p decomposition."""
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _als_decompose(n, m, p, r, rng, n_iters=100, deadline=None):
    """Alternating Least Squares for tensor decomposition.
    Returns (U, V, W, loss) or None if timeout."""
    nm, mp, pn = n * m, m * p, p * n
    on = _build_support(n, m, p)
    on_a, on_b, on_c = on[:, 0], on[:, 1], on[:, 2]

    # Initialize
    U = rng.standard_normal((nm, r)) * 0.5
    V = rng.standard_normal((mp, r)) * 0.5
    W = rng.standard_normal((pn, r)) * 0.5

    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            return None

        # Update U: for each a, solve U[a,:] such that U[a,:] @ (V[b,:] * W[c,:]) = 1 for all (a,b,c) in on
        for a in range(nm):
            mask = (on_a == a)
            if not np.any(mask):
                continue
            # Build matrix: rows are (V[b,:] * W[c,:]) for each on-triple with this a
            VW = V[on_b[mask]] * W[on_c[mask]]  # (n_a, r)
            # Solve U[a,:] @ VW.T = 1
            # This is an overdetermined system; use least squares
            try:
                U[a, :] = np.linalg.lstsq(VW, np.ones(len(VW)), rcond=None)[0]
            except:
                pass

        # Update V: for each b, solve V[b,:] such that (U[a,:] * W[c,:]) @ V[b,:] = 1
        for b in range(mp):
            mask = (on_b == b)
            if not np.any(mask):
                continue
            UW = U[on_a[mask]] * W[on_c[mask]]  # (n_b, r)
            try:
                V[b, :] = np.linalg.lstsq(UW, np.ones(len(UW)), rcond=None)[0]
            except:
                pass

        # Update W: for each c, solve W[c,:] such that (U[a,:] * V[b,:]) @ W[c,:] = 1
        for c in range(pn):
            mask = (on_c == c)
            if not np.any(mask):
                continue
            UV = U[on_a[mask]] * V[on_b[mask]]  # (n_c, r)
            try:
                W[c, :] = np.linalg.lstsq(UV, np.ones(len(UV)), rcond=None)[0]
            except:
                pass

    # Compute final loss
    VW = V[on_b] * W[on_c]
    vals = np.einsum('tr,ar->t', VW, U[on_a])
    loss = np.sum((vals - 1.0) ** 2)
    return U, V, W, loss


def _try_rank(n, m, p, r, n_restarts, rng, deadline):
    """Try to find an exact decomposition of rank r."""
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        res = _als_decompose(n, m, p, r, rng, n_iters=80, deadline=deadline)
        if res is None:
            continue
        U, V, W, loss = res
        if loss < 1e-6:
            # Try to snap to simple values
            if _verify(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x6."""
    start = time.time()
    deadline = start + 140
    n, m, p = N_DIM, M_DIM, P_DIM

    base = _schoolbook(n, m, p)
    best = base
    best_rank = n * m * p

    rng = np.random.default_rng(42)

    # Try ranks from lower bound upward
    min_r = 18  # max(nm, mp, pn) = max(9, 18, 18)
    for r in range(min_r, best_rank):
        if time.time() > deadline - 5:
            break
        remaining = deadline - 5 - time.time()
        if remaining < 3:
            break
        # Allocate restarts based on remaining time
        n_restarts = max(1, min(8, int(remaining / 3)))
        res = _try_rank(n, m, p, r, n_restarts, rng, deadline - 5)
        if res is not None:
            best = res
            best_rank = r
            break

    # Final verification
    if time.time() < deadline - 2:
        try:
            if not _verify(best["U"], best["V"], best["W"], n, m, p):
                return base
        except:
            return base

    return best
# EVOLVE-BLOCK-END
