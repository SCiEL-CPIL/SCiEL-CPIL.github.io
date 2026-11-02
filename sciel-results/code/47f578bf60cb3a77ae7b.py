"""Rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 7]
START = time.time()
DEADLINE = 150.0 - 10.0  # 140s safety margin


def _tensor_entries(n, m, p):
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i, j, k)] = (i, k, j)
    return T


def _check(U, V, W, T):
    n, m, p = 2, 2, 7
    for (i, k, j), val in T.items():
        s = 0.0
        for c in range(len(U[0])):
            s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
        if abs(s - val) > 1e-6:
            return False
    return True


def _check_fast(U, V, W, T):
    n, m, p = 2, 2, 7
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    for (i, k, j), val in T.items():
        s = float(np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]))
        if abs(s - val) > 1e-6:
            return False
    return True


def _base_decomposition(n=2, m=2, p=7):
    U = []
    V = []
    W = []
    for b in range(0, p - 1, 2):
        k0, k1 = b, b + 1
        # Strassen 2x2 * 2x2 block (columns k0,k1)
        # M1=(A00+A11)(B00+B11)
        U.append([1, 0, 0, 0]); V.append([1, 0, 0, 0]); W.append([1, 0, 1, 0])
        # M2=(A00+A11)(B10-B11)
        U.append([1, 0, 0, 0]); V.append([0, 1, 0, -1]); W.append([0, 1, 0, 1])
        # M3=(A00-A11)(B10+B11)
        U.append([1, 0, 0, 0]); V.append([0, 1, 0, 1]); W.append([0, 1, -1, -1])
        # M4=A10*(B00+B11)
        U.append([0, 1, 0, 0]); V.append([1, 0, 0, 0]); W.append([0, 0, 1, 0])
        # M5=A11*(B10+B11)
        U.append([0, 0, 0, 1]); V.append([0, 1, 0, 1]); W.append([0, 0, 0, 1])
        # M6=(A00+A10)B00
        U.append([1, 1, 0, 0]); V.append([1, 0, 0, 0]); W.append([1, 0, 0, 0])
        # M7=(A11-A00)B11
        U.append([1, 0, 0, -1]); V.append([0, 0, 0, 1]); W.append([0, 0, 1, 1])
    # last column k=6 schoolbook
    U.append([1, 0, 0, 0]); V.append([1, 0, 0, 0]); W.append([1, 0, 0, 0])
    U.append([0, 1, 0, 0]); V.append([1, 0, 0, 0]); W.append([0, 0, 1, 0])
    U.append([0, 0, 0, 1]); V.append([1, 0, 0, 0]); W.append([0, 0, 0, 1])
    U.append([0, 0, 1, 0]); V.append([1, 0, 0, 0]); W.append([0, 1, 0, 0])
    return U, V, W


def _try_reduce(U, V, W, T, target_rank, rng, iters):
    """Search for a linear combination of columns that yields a valid lower-rank decomp."""
    n, m, p = 2, 2, 7
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    # Precompute per-tensor-entry column vectors (products U*V*W per column)
    # For entry (i,k,j): vector over columns = U[i*m+j] * V[j*p+k] * W[k*n+i]
    entries = list(T.keys())
    vecs = []
    for (i, k, j) in entries:
        v = U[i * m + j] * V[j * p + k] * W[k * n + i]
        vecs.append(v)
    vecs = np.array(vecs)  # (E, r)
    vals = np.array([T[e] for e in entries], dtype=float)
    # We want coefficient vector x (length r) with support <= target_rank such that
    # vecs @ x == vals. Find a sparse solution.
    # Use iterative hard thresholding / basis pursuit via random restarts.
    for it in range(iters):
        if time.time() > DEADLINE:
            break
        # start from a random sparse mask
        mask = rng.choice(r, size=target_rank, replace=False)
        M = np.zeros(r)
        M[mask] = 1.0
        # solve least squares on masked coords, iterate
        x = np.zeros(r)
        for inner in range(30):
            x = M * np.linalg.lstsq(vecs[:, M == 1], vals, rcond=None)[0]
            # re-threshold: keep target_rank largest |x|
            idx = np.argsort(np.abs(x))[::-1][:target_rank]
            newM = np.zeros(r)
            newM[idx] = 1.0
            if np.array_equal(newM, M):
                break
            M = newM
        # check
        pred = vecs @ x
        if np.allclose(pred, vals, atol=1e-5):
            # build factor matrices
            nz = np.nonzero(np.abs(x) > 1e-9)[0]
            U2 = U[:, nz]
            V2 = V[:, nz]
            W2 = W[:, nz]
            return U2, V2, W2
    return None


def _try_random_search(U, V, W, T, target_rank, rng, budget):
    """Random search over sparse coefficient vectors."""
    n, m, p = 2, 2, 7
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    entries = list(T.keys())
    vecs = []
    for (i, k, j) in entries:
        v = U[i * m + j] * V[j * p + k] * W[k * n + i]
        vecs.append(v)
    vecs = np.array(vecs)
    vals = np.array([T[e] for e in entries], dtype=float)
    for it in range(budget):
        if time.time() > DEADLINE:
            break
        mask = set(rng.choice(r, size=target_rank, replace=False))
        M = np.zeros(r)
        for c in mask:
            M[c] = 1.0
        try:
            sol = np.linalg.lstsq(vecs[:, M == 1], vals, rcond=None)[0]
        except Exception:
            continue
        x = np.zeros(r)
        x[M == 1] = sol
        pred = vecs @ x
        if np.allclose(pred, vals, atol=1e-5):
            nz = np.nonzero(np.abs(x) > 1e-9)[0]
            return U[:, nz], V[:, nz], W[:, nz]
    return None


def construct():
    n, m, p = DIMENSIONS
    T = _tensor_entries(n, m, p)
    U, V, W = _base_decomposition(n, m, p)
    assert _check(U, V, W, T), "base decomposition invalid"
    best = (U, V, W)
    best_rank = len(U[0])

    rng = random.Random(12345)
    # Try to reduce below 25
    for target in [24, 23, 22, 21, 20]:
        if time.time() > DEADLINE - 5:
            break
        res = _try_reduce(U, V, W, T, target, rng, iters=200)
        if res is not None:
            U2, V2, W2 = res
            if _check(U2.tolist(), V2.tolist(), W2.tolist(), T):
                best = (U2, V2, W2)
                best_rank = U2.shape[1]
                break
        # also random search
        if time.time() > DEADLINE - 5:
            break
        res2 = _try_random_search(U, V, W, T, target, rng, budget=100)
        if res2 is not None:
            U2, V2, W2 = res2
            if _check(U2.tolist(), V2.tolist(), W2.tolist(), T):
                best = (U2, V2, W2)
                best_rank = U2.shape[1]
                break

    Uf, Vf, Wf = best
    return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}
