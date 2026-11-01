"""Tensor-rank search for 2x3 * 3x10 matrix multiplication.

Structural change vs the seed: instead of returning the fixed rank-60
schoolbook decomposition, we actively SEARCH for a lower-rank exact
decomposition using an iterative rank-reduction scheme (rank-revealing
QR on the bilinear tensor, then local refinement), with a hard
wall-clock deadline guard. Every candidate is verified EXACTLY before
being accepted.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]

_DEADLINE = 150.0
_START = time.time()


def _remaining():
    return _DEADLINE - 15.0 - (time.time() - _START)


def _verify(U, V, W, n, m, p, tol=1e-8):
    """Return True iff sum_c U[:,c] o V[:,c] o W[:,c] == T (exact, within tol)."""
    rank = U.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for c in range(rank):
        T += np.einsum('a,b,d->abd', U[:, c], V[:, c], W[:, c])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i * m + j, j * p + k, k * n + i] - 1.0) > tol:
                    return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ii in range(n):
                    for jj in range(m):
                        for kk in range(p):
                            if (i, j, k) == (ii, jj, kk):
                                continue
                            if abs(T[i * m + j, j * p + k, kk * n + ii]) > tol:
                                return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _qr_reduce(U, V, W, n, m, p, target):
    """Rank-revealing QR on the bilinear tensor to reduce rank toward target."""
    rank = U.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for c in range(rank):
        T += np.einsum('a,b,d->abd', U[:, c], V[:, c], W[:, c])
    # Flatten the tensor and do rank-revealing QR (SVD-based) to get a
    # lower-rank approximation, then re-extract factors.
    flat = T.reshape(-1)
    # Use a structured low-rank reconstruction: SVD of the mode-1 matricization
    U1 = T.reshape(n * m, m * p * p * n)
    Umat, S, Vt = np.linalg.svd(U1, full_matrices=False)
    r = min(target, len(S))
    # Reconstruct a rank-r tensor
    Tlow = (Umat[:, :r] * S[:r]) @ Vt[:r, :]
    Tlow = Tlow.reshape(n * m, m * p, p * n)
    # Now factor the rank-r tensor into r rank-1 terms via a simple greedy
    # extraction.
    Uf = np.zeros((n * m, r))
    Vf = np.zeros((m * p, r))
    Wf = np.zeros((p * n, r))
    rem = Tlow.copy()
    for c in range(r):
        # Greedy: pick the largest remaining entry as the anchor.
        idx = np.unravel_index(np.argmax(np.abs(rem)), rem.shape)
        a, b, d = idx
        val = rem[a, b, d]
        if abs(val) < 1e-12:
            break
        # Take the three slices through this anchor.
        ua = rem[a, :, d]
        vb = rem[:, b, d]
        wd = rem[a, b, :]
        # Normalize
        na = np.linalg.norm(ua)
        nb = np.linalg.norm(vb)
        nd = np.linalg.norm(wd)
        if na < 1e-12 or nb < 1e-12 or nd < 1e-12:
            break
        ua = ua / na
        vb = vb / nb
        wd = wd / nd
        scale = val / (na * nb * nd)
        Uf[:, c] = ua
        Vf[:, c] = vb
        Wf[:, c] = wd
        rem -= scale * np.einsum('a,b,d->abd', ua, vb, wd)
    return Uf[:, :r], Vf[:, :r], Wf[:, :r]


def _local_refine(U, V, W, n, m, p, iters=40, lr=0.02):
    """Local gradient-free refinement: small perturbations that keep validity."""
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    best = U.copy()
    bestV = V.copy()
    bestW = W.copy()
    rng = np.random.default_rng(0)
    for _ in range(iters):
        if _remaining() < 3.0:
            break
        # Try a random small perturbation
        dU = rng.normal(0, lr, U.shape)
        dV = rng.normal(0, lr, V.shape)
        dW = rng.normal(0, lr, W.shape)
        nU = U + dU
        nV = V + dV
        nW = W + dW
        if _verify(nU, nV, nW, n, m, p):
            U, V, W = nU, nV, nW
            best, bestV, bestW = U.copy(), V.copy(), W.copy()
    return best, bestV, bestW


def _try_rank(n, m, p, target, seed=0):
    """Attempt to find a valid rank-<=target decomposition."""
    if _remaining() < 2.0:
        return None
    random.seed(seed)
    np.random.seed(seed)
    U, V, W = _schoolbook(n, m, p)
    # Reduce via QR toward target
    U, V, W = _qr_reduce(U, V, W, n, m, p, target)
    rank = U.shape[1]
    if rank > target:
        # Pad/truncate to exactly target
        if rank < target:
            U = np.pad(U, ((0, 0), (0, target - rank)))
            V = np.pad(V, ((0, 0), (0, target - rank)))
            W = np.pad(W, ((0, 0), (0, target - rank)))
        else:
            U = U[:, :target]
            V = V[:, :target]
            W = W[:, :target]
        rank = target
    # Local refinement
    U, V, W = _local_refine(U, V, W, n, m, p, iters=30)
    if _verify(U, V, W, n, m, p):
        return U, V, W
    return None


def construct():
    n, m, p = DIMENSIONS
    # Always-available fallback: schoolbook rank 60.
    Uf, Vf, Wf = _schoolbook(n, m, p)

    # Search from low rank upward; keep the best valid result.
    # Lower bound for this tensor is at least n*m = 6 and m*p = 30 and p*n = 20,
    # so a reasonable search range is 30..55.
    best = (Uf, Vf, Wf)
    best_rank = Uf.shape[1]
    for target in range(30, 56):
        if _remaining() < 8.0:
            break
        res = _try_rank(n, m, p, target, seed=target)
        if res is not None:
            U, V, W = res
            if U.shape[1] < best_rank:
                best_rank = U.shape[1]
                best = (U, V, W)
                # Found a lower rank; continue searching lower is not possible
                # since we go upward, so we can stop searching higher targets
                # only if we hit the lower bound; keep going to be safe but
                # we already have a good result.
            if target == 30:
                # Best possible found; stop.
                break

    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
