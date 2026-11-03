"""Tensor-rank search for 2x3x4 matrix multiplication, with a hard deadline guard."""
import time
import random
import math

import numpy as np

DIMENSIONS = [2, 3, 4]


def _snap(x, tol=1e-7):
    """Snap a float to the nearest half-integer if close enough."""
    r = np.rint(2.0 * x) / 2.0
    if abs(x - r) <= tol:
        return float(r)
    return float(x)


def _matrices_to_halfints(M):
    """Convert a 2D list/array to a list of lists of int or float (half-integers)."""
    M = np.asarray(M, dtype=float)
    out = []
    for row in M:
        out.append([_snap(v) for v in row])
    return out


def _verify(U, V, W, n=2, m=3, p=4, tol=1e-7):
    """Check that the decomposition exactly reproduces the multiplication tensor."""
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # Compute T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i * m + j] * vc[j * p + k] * wc[k * n + i]
    # Expected: T[i, j, k] = 1 for all (i, j, k) since (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor entry is 1 for every (i,j,k).
    return np.allclose(T, 1.0, atol=tol, rtol=0.0)


def _schoolbook(n=2, m=3, p=4):
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


def _random_init(rank, n=2, m=3, p=4, seed=None):
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    # Scale to reduce initial error magnitude
    U /= np.linalg.norm(U) * 2
    V /= np.linalg.norm(V) * 2
    W /= np.linalg.norm(W) * 2
    return U, V, W


def _loss(U, V, W, n=2, m=3, p=4):
    """Sum of squared errors of the tensor entries vs 1."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i * m + j] * vc[j * p + k] * wc[k * n + i]
    return np.sum((T - 1.0) ** 2)


def _gradient_step(U, V, W, lr, n=2, m=3, p=4):
    """One full-batch gradient descent step on the loss (sum of squares)."""
    rank = U.shape[1]
    # Compute T and its gradient
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i * m + j] * vc[j * p + k] * wc[k * n + i]
    E = T - 1.0  # 2 * E is dL/dT

    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    e = E[i, j, k]
                    dU[i * m + j, c] += 2 * e * vc[j * p + k] * wc[k * n + i]
                    dV[j * p + k, c] += 2 * e * uc[i * m + j] * wc[k * n + i]
                    dW[k * n + i, c] += 2 * e * uc[i * m + j] * vc[j * p + k]

    U = U - lr * dU
    V = V - lr * dV
    W = W - lr * dW
    return U, V, W


def _refine(U, V, W, n=2, m=3, p=4, max_iter=400, lr0=0.05, deadline=None):
    """Polish a decomposition with adaptive learning rate."""
    best = (U.copy(), V.copy(), W.copy())
    best_loss = _loss(U, V, W, n, m, p)
    lr = lr0
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        U, V, W = _gradient_step(U, V, W, lr, n, m, p)
        loss = _loss(U, V, W, n, m, p)
        if loss < best_loss:
            best_loss = loss
            best = (U.copy(), V.copy(), W.copy())
            lr = min(lr * 1.05, 0.2)
        else:
            lr *= 0.85
            if lr < 1e-5:
                break
    return best[0], best[1], best[2]


def _try_rank(rank, n=2, m=3, p=4, deadline=None, n_restarts=6):
    """Try to find a valid rank-`rank` decomposition; return (U, V, W) or None."""
    best = None
    best_loss = float('inf')
    for r in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        seed = 1000 + rank * 7 + r
        U, V, W = _random_init(rank, n, m, p, seed=seed)
        # If rank == n*m*p, also try starting from a perturbed schoolbook projection
        if rank == n * m * p:
            U0, V0, W0 = _schoolbook(n, m, p)
            U, V, W = U0 + 0.01 * np.random.RandomState(seed).randn(*U0.shape), \
                        V0 + 0.01 * np.random.RandomState(seed + 1).randn(*V0.shape), \
                        W0 + 0.01 * np.random.RandomState(seed + 2).randn(*W0.shape)
        U, V, W = _refine(U, V, W, n, m, p, max_iter=500, lr0=0.08, deadline=deadline)
        loss = _loss(U, V, W, n, m, p)
        if loss < best_loss:
            best_loss = loss
            best = (U, V, W)
            if loss < 1e-9:
                break
    if best is None:
        return None
    U, V, W = best
    if _verify(U, V, W, n, m, p, tol=1e-7):
        return (U, V, W)
    # Try a final tighter polish
    U, V, W = _refine(U, V, W, n, m, p, max_iter=800, lr0=0.02, deadline=deadline)
    if _verify(U, V, W, n, m, p, tol=1e-7):
        return (U, V, W)
    return None


def construct():
    start = time.time()
    deadline = start + 120.0  # safety margin before the 150s kill
    n, m, p = DIMENSIONS

    # Fallback: schoolbook (always valid)
    U0, V0, W0 = _schoolbook(n, m, p)
    best_rank = n * m * p
    best = (U0, V0, W0)

    # Try to find a lower rank, descending from best_rank-1 down to a floor.
    # Theoretical lower bound for matmul(n,m,p) is >= max(n*m, m*p, p*n) = max(6,12,8)=12.
    lower = max(n * m, m * p, p * n)
    for rank in range(best_rank - 1, lower - 1, -1):
        if time.time() > deadline:
            break
        res = _try_rank(rank, n, m, p, deadline=deadline, n_restarts=4)
        if res is not None:
            best_rank = rank
            best = res
            # Found a valid one; stop searching lower (we want minimal rank).
            break

    U, V, W = best
    return {
        "U": _matrices_to_halfints(U),
        "V": _matrices_to_halfints(V),
        "W": _matrices_to_halfints(W),
    }
