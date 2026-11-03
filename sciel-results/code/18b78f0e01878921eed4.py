"""Rewritten program: search for rank-minimal exact tensor decomposition of 2x4x6 matmul."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 6]
DEADLINE = 120.0
SAFETY_MARGIN = 8.0
RNG_SEED = 12345

# EVOLVE-BLOCK-START
def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _tensor_residual(U, V, W):
    """Return the 3-way residual tensor T - sum_c U[:,c] V[:,c] W[:,c]."""
    n, m, p = DIMENSIONS
    a, b, c = len(U), len(V), len(W)
    rank = len(U[0])
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    T = np.zeros((a, b, c), dtype=float)
    for idx in range(rank):
        T += np.einsum('i,j,k->ijk', Ua[:, idx], Va[:, idx], Wa[:, idx])
    # True matmul tensor: T[i,j,k] = 1 iff i<m and j<p and (i*p+j)==k
    for i in range(n):
        for j in range(m):
            k = i * p + j
            if k < p:
                T[i, j, k] -= 1
    return T


def _rank_of(U, V, W):
    """Count columns where all three factor rows are nonzero."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    rank = 0
    for idx in range(Ua.shape[1]):
        if (np.linalg.norm(Ua[:, idx]) > 1e-7 and
                np.linalg.norm(Va[:, idx]) > 1e-7 and
                np.linalg.norm(Wa[:, idx]) > 1e-7):
            rank += 1
    return rank


def _random_factors(rank, seed):
    rng = np.random.default_rng(seed)
    n, m, p = DIMENSIONS
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    return U.tolist(), V.tolist(), W.tolist()


def _refine(U, V, W, steps, lr, deadline, rng):
    """Gradient descent on ||sum U_c V_c W_c - T||^2 to reduce residual."""
    n, m, p = DIMENSIONS
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    rank = Ua.shape[1]
    a, b, c = Ua.shape[0], Va.shape[0], Wa.shape[0]
    T = np.zeros((a, b, c), dtype=float)
    for i in range(n):
        for j in range(m):
            k = i * p + j
            if k < p:
                T[i, j, k] = 1.0
    for _ in range(steps):
        if time.time() > deadline:
            break
        # Build residual
        R = np.zeros((a, b, c), dtype=float)
        for idx in range(rank):
            R += np.einsum('i,j,k->ijk', Ua[:, idx], Va[:, idx], Wa[:, idx])
        R -= T
        # Gradients
        # dR/dU[i, idx] = sum_jk R[i,j,k] V[j,idx] W[k,idx]
        dU = np.einsum('ijk,jd,kd->id', R, Va, Wa)
        dV = np.einsum('ijk,id,kd->jd', R, Ua, Wa)
        dW = np.einsum('ijk,id,jd->kd', R, Ua, Va)
        Ua -= lr * dU
        Va -= lr * dV
        Wa -= lr * dW
    return Ua.tolist(), Va.tolist(), Wa.tolist()


def _verify(U, V, W):
    """Check exactness: residual should be exactly zero (within 1e-6)."""
    R = _tensor_residual(U, V, W)
    return np.linalg.norm(R) < 1e-6


def construct():
    start = time.time()
    deadline = start + DEADLINE - SAFETY_MARGIN
    rng = np.random.default_rng(RNG_SEED)
    best = None
    best_rank = None

    # Try candidate ranks in ascending order; stop as soon as one verifies exactly.
    for rank in range(18, 50):
        if time.time() > deadline:
            break
        for attempt in range(3):
            if time.time() > deadline:
                break
            seed = int(rng.integers(0, 2**31 - 1))
            U, V, W = _random_factors(rank, seed)
            U, V, W = _refine(U, V, W, steps=300, lr=1e-3, deadline=deadline, rng=rng)
            if _verify(U, V, W):
                r = _rank_of(U, V, W)
                if best is None or r < best_rank:
                    best = (U, V, W)
                    best_rank = r
                # Convert to lists of floats (exact small ints if possible)
                U = [[float(x) for x in row] for row in np.asarray(U, dtype=float)]
                V = [[float(x) for x in row] for row in np.asarray(V, dtype=float)]
                W = [[float(x) for x in row] for row in np.asarray(W, dtype=float)]
                return {"U": U, "V": V, "W": W}
    if best is not None:
        U, V, W = best
        U = [[float(x) for x in row] for row in np.asarray(U, dtype=float)]
        V = [[float(x) for x in row] for row in np.asarray(V, dtype=float)]
        W = [[float(x) for x in row] for row in np.asarray(W, dtype=float)]
        return {"U": U, "V": V, "W": W}
    # Fallback: guaranteed schoolbook rank n*m*p = 48
    return _build_schoolbook()
# EVOLVE-BLOCK-END
