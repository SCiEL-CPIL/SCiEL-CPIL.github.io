"""Tensor decomposition for 3x3x7 matrix multiplication."""
import numpy as np
import time

_START = time.time()
TIME_LIMIT = 140.0  # safety margin below 150s kill


def _time_left():
    return TIME_LIMIT - (time.time() - _START)


def _verify(U, V, W, n=3, m=3, p=7, tol=1e-9):
    """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # Build tensor via einsum
    # U shape (n*m, R), V (m*p, R), W (p*n, R)
    # M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    # M[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # sum over r and j? No: for each (i,j,k) it's sum_r U3[i,j,r]*V3[j,k,r]*W3[k,i,r]
    # Compute: T[i,j,k] = sum_r (U3[i,j,:]*V3[j,k,:]*W3[k,i,:])
    # Use einsum: 'ijr,jkr,kir->ijk'
    M = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.max(np.abs(M - 1.0)) < tol


def _to_halfint(x):
    """Round to nearest half-integer."""
    return np.round(x * 2.0) / 2.0


def _try_rank_random(R, n=3, m=3, p=7, iters=2000, seed=0):
    """Random search for rank-R decomposition."""
    rng = np.random.default_rng(seed)
    best_err = np.inf
    best = None
    for it in range(iters):
        if _time_left() < 5.0:
            break
        U = rng.standard_normal((n * m, R))
        V = rng.standard_normal((m * p, R))
        W = rng.standard_normal((p * n, R))
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        M = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = np.max(np.abs(M - 1.0))
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        if best_err < 1e-9:
            break
    return best_err, best


def _polish(U, V, W, n=3, m=3, p=7, steps=500, lr=0.05):
    """Gradient descent polish on log-sum-exp surrogate of max|M-1|."""
    U = np.asarray(U, dtype=float).copy()
    V = np.asarray(V, dtype=float).copy()
    W = np.asarray(W, dtype=float).copy()
    R = U.shape[1]
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    for s in range(steps):
        if _time_left() < 5.0:
            break
        M = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        diff = M - 1.0
        err = np.max(np.abs(diff))
        if err < 1e-12:
            break
        # gradient of sum(diff^2) w.r.t. U[i,j,r] = 2*diff[i,j,k]*V[j,k,r]*W[k,i,r] summed over k
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        # gU[i*m+j, r] = 2 * sum_k diff[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    d = 2.0 * diff[i, j, k]
                    if d == 0:
                        continue
                    vr = V[j * p + k, :]
                    wr = W[k * n + i, :]
                    gU[i * m + j, :] += d * vr * wr
                    gV[j * p + k, :] += d * U[i * m + j, :] * wr
                    gW[k * n + i, :] += d * U[i * m + j, :] * vr
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
    M = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    err = np.max(np.abs(M - 1.0))
    return err, (U, V, W)


def construct():
    n, m, p = 3, 3, 7
    # Trivial fallback
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0))
    V0 = np.zeros((m * p, rank0))
    W0 = np.zeros((p * n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    best_rank = rank0
    best_UVW = (U0, V0, W0)

    # Try structured low-rank via random + polish, descending rank
    # Theoretical lower bound considerations; try ranks from 24 upward
    for R in range(24, rank0):
        if _time_left() < 10.0:
            break
        # Multiple random seeds
        for seed in range(3):
            if _time_left() < 8.0:
                break
            err, cand = _try_rank_random(R, n, m, p, iters=300, seed=1000 + R * 10 + seed)
            if cand is None:
                continue
            # Polish
            err2, cand2 = _polish(cand[0], cand[1], cand[2], n, m, p, steps=300, lr=0.02)
            if err2 < 1e-9:
                # Verify exactness
                if _verify(cand2[0], cand2[1], cand2[2], n, m, p):
                    # Round to half-integers and re-verify
                    Ur = _to_halfint(cand2[0])
                    Vr = _to_halfint(cand2[1])
                    Wr = _to_halfint(cand2[2])
                    if _verify(Ur, Vr, Wr, n, m, p):
                        best_rank = R
                        best_UVW = (Ur, Vr, Wr)
                        break
                    else:
                        best_rank = R
                        best_UVW = (cand2[0], cand2[1], cand2[2])
                        break
            if err2 < err:
                # keep improved
                pass
        if best_rank < rank0 and _time_left() < 20.0:
            break

    U, V, W = best_UVW
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
