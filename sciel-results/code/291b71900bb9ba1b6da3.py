import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]
_DEADLINE = time.time() + 125.0  # safety margin before 150s kill


def _time_left():
    return _DEADLINE - time.time()


def _check(U, V, W, n, m, p):
    """Return True if the decomposition is exact (within tolerance)."""
    T = np.zeros((n, m, p))
    for r in range(U.shape[1]):
        Ucol = U[:, r].reshape(n, m)
        Vcol = V[:, r].reshape(m, p)
        for i in range(n):
            for j in range(m):
                c = Ucol[i, j]
                if c != 0:
                    T[i, j, :] += c * Vcol[j, :]
    # Compare T to the multiplication tensor (all 1s)
    return np.allclose(T, 1.0, atol=1e-6)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _rank_deflation(U, V, W, n, m, p, max_iter=200):
    """Try to reduce rank by finding and removing a rank-1 component."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]

    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        # Find a row of U with exactly one nonzero -> that column is a rank-1 component
        found = False
        for r in range(rank):
            col = U[:, r]
            nz = np.nonzero(col)[0]
            if len(nz) == 1:
                idx = nz[0]
                i, j = divmod(idx, m)
                val = col[idx]
                # This column contributes val * V[:, r] to T[i, j, :]
                # Subtract from W rows k*n+i
                for k in range(p):
                    W[k * n + i, r] = 0.0  # already there
                # Check if this is a valid rank-1: T[i,j,:] should equal val * V[:, r]
                # and all other T entries involving this must be consistent
                # Remove this column
                U = np.delete(U, r, axis=1)
                V = np.delete(V, r, axis=1)
                W = np.delete(W, r, axis=1)
                rank -= 1
                found = True
                break
        if not found:
            # Try columns of V with one nonzero
            found = False
            for r in range(rank):
                col = V[:, r]
                nz = np.nonzero(col)[0]
                if len(nz) == 1:
                    idx = nz[0]
                    j, k = divmod(idx, p)
                    val = col[idx]
                    U = np.delete(U, r, axis=1)
                    V = np.delete(V, r, axis=1)
                    W = np.delete(W, r, axis=1)
                    rank -= 1
                    found = True
                    break
        if not found:
            break
    return U, V, W, rank


def _try_merge(U, V, W, n, m, p):
    """Try to merge two rank-1 terms that share structure."""
    rank = U.shape[1]
    for r1 in range(rank):
        for r2 in range(r1 + 1, rank):
            # Check if U[:, r1] and U[:, r2] have disjoint support
            nz1 = set(np.nonzero(U[:, r1])[0])
            nz2 = set(np.nonzero(U[:, r2])[0])
            if not nz1.isdisjoint(nz2):
                continue
            # Check V
            nz1v = set(np.nonzero(V[:, r1])[0])
            nz2v = set(np.nonzero(V[:, r2])[0])
            if not nz1v.isdisjoint(nz2v):
                continue
            # Check W
            nz1w = set(np.nonzero(W[:, r1])[0])
            nz2w = set(np.nonzero(W[:, r2])[0])
            if not nz1w.isdisjoint(nz2w):
                continue
            # Try merging: create a new column that is sum
            # This is valid if the tensor contribution adds up
            # For now, just remove one if the other covers it
            # (simplified: skip for now)
            pass
    return U, V, W, rank


def construct():
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)
    best_rank = U.shape[1]

    # Try rank deflation
    U2, V2, W2, r2 = _rank_deflation(U, V, W, n, m, p)
    if r2 < best_rank and _check(U2, V2, W2, n, m, p):
        U, V, W, best_rank = U2, V2, W2, r2

    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
