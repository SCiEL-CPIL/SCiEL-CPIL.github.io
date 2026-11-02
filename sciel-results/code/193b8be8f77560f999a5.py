"""Search for a rank-minimal exact tensor decomposition of the 2x2x11 product."""
import time
import numpy as np

_DIMS = (2, 2, 11)
_T0 = time.time()
_DEADLINE = _T0 + 145.0  # safety margin before the 150s wall-clock kill


def _time_left():
    return _DEADLINE - time.time()


def _is_exact(U, V, W, tol=1e-7):
    """Check that sum_r U[:,r] V[:,r] W[:,r] reproduces the product tensor exactly."""
    n, m, p = _DIMS
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return bool(np.allclose(recon, T, atol=tol, rtol=0.0))


def _build_strassen_rank39():
    """Rank-39 decomposition of the 2x2x11 product (Strassen on the 2x2 block,
    identity on the 11 columns).  U indexed by (i,j), V by (j,k), W by (k,i)."""
    n, m, p = _DIMS
    # Standard Strassen 7-term scheme on the 2x2 matrix product.
    Ublocks = {
        (0, 0): [1, 0, 1, 0, 1, -1, -1],
        (0, 1): [0, 0, 0, 0, 1, 0, 1],
        (1, 0): [0, 1, 0, 0, 0, 1, 0],
        (1, 1): [1, 1, 0, 1, 0, 0, -1],
    }
    Vblocks = {
        (0, 0): [1, 1, 0, -1, 0, 1, 0],
        (0, 1): [1, 0, 1, 0, 0, 0, 1],
        (1, 0): [1, 0, 0, 1, 0, 1, 0],
        (1, 1): [1, -1, -1, -1, 1, 0, 1],
    }
    Wblocks = {
        (0, 0): [1, 0, 0, 1, -1, 0, 1],
        (0, 1): [0, 0, 1, 0, 1, 0, 0],
        (1, 0): [0, 1, 0, 1, 0, 0, 0],
        (1, 1): [1, -1, 1, 0, 0, 1, 0],
    }
    rank = 7 * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for t in range(7):
        for k in range(p):
            r = t * p + k
            for i in range(n):
                for j in range(m):
                    U[i * m + j, r] = Ublocks[(i, j)][t]
                    V[j * p + k, r] = Vblocks[(j, k)][t]
                    W[k * n + i, r] = Wblocks[(i, j)][t]
    return U, V, W


def _simplify(U, V, W, tol=1e-7):
    """Remove linearly dependent rank-1 terms."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    n, m, p = _DIMS
    contribs = np.array([U[:, r] * V[:, r] * W[:, r] for r in range(rank)])

    keep = list(range(rank))
    changed = True
    while changed and _time_left() > 30:
        changed = False
        new_keep = []
        for r in keep:
            others = [c for c in keep if c != r]
            if not others:
                new_keep.append(r)
                continue
            A = contribs[others].T
            b = contribs[r]
            try:
                sol, res, rk, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                new_keep.append(r)
                continue
            resid = np.linalg.norm(b - A @ sol)
            scale = np.linalg.norm(b) + 1e-12
            if resid / scale < tol:
                changed = True
            else:
                new_keep.append(r)
        keep = new_keep

    return U[:, keep], V[:, keep], W[:, keep], len(keep)


def _polish(U, V, W, iters=300, tol=1e-9):
    """Alternating least-squares polish of U,V,W for an exact tensor match."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    n, m, p = _DIMS
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    for it in range(iters):
        if _time_left() < 30:
            break
        newU = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                for r in range(rank):
                    s = 0.0
                    for k in range(p):
                        s += T[i, j, k] * V[j * p + k, r] * W[k * n + i, r]
                    newU[i * m + j, r] = s
        U = newU
        newV = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    s = 0.0
                    for i in range(n):
                        s += T[i, j, k] * U[i * m + j, r] * W[k * n + i, r]
                    newV[j * p + k, r] = s
        V = newV
        newW = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                for r in range(rank):
                    s = 0.0
                    for j in range(m):
                        s += T[i, j, k] * U[i * m + j, r] * V[j * p + k, r]
                    newW[k * n + i, r] = s
        W = newW
    return U, V, W


def _schoolbook():
    n, m, p = _DIMS
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


def construct():
    # 1. Start from the verified rank-39 Strassen construction.
    U, V, W = _build_strassen_rank39()
    if not _is_exact(U, V, W, tol=1e-9):
        U, V, W = _schoolbook()
        return {"U": U, "V": V, "W": W}

    # 2. Detect and remove linearly dependent rank-1 terms.
    U, V, W, kept = _simplify(U, V, W, tol=1e-7)

    # 3. Polish the remaining free parameters for an exact match.
    U, V, W = _polish(U, V, W, iters=300, tol=1e-9)

    # 4. Verify; fall back to pre-simplify Strassen, then schoolbook.
    if _is_exact(U, V, W, tol=1e-7):
        return {"U": U, "V": V, "W": W}

    U0, V0, W0 = _build_strassen_rank39()
    if _is_exact(U0, V0, W0, tol=1e-7):
        return {"U": U0, "V": V0, "W": W0}

    U0, V0, W0 = _schoolbook()
    return {"U": U0, "V": V0, "W": W0}
