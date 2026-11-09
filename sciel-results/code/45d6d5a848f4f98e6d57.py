"""Matrix multiplication 2x4 by 4x8 tensor decomposition.

We construct an exact decomposition using Strassen's rank-7 algorithm for 2x2x2
applied to the 8 independent sub-blocks, giving rank 56. Then we attempt to
reduce the rank via alternating least squares (ALS) with rational snapping and
exact Fraction verification.
"""

import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def _build_strassen_rank56():
    """Build the baseline rank-56 Strassen decomposition."""
    n, m, p = DIMENSIONS
    strassen_products = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for ar in range(2):
        for qc in range(4):
            for u_coeff, v_coeff, w_coeff in strassen_products:
                for (li, lj), c in u_coeff.items():
                    gi = li
                    gj = 2 * ar + lj
                    U[gi * m + gj, col] += c
                for (lj, lk), c in v_coeff.items():
                    gj = 2 * ar + lj
                    gk = 2 * qc + lk
                    V[gj * p + gk, col] += c
                for (lk, li), c in w_coeff.items():
                    gk = 2 * qc + lk
                    gi = li
                    W[gk * n + gi, col] += c
                col += 1

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1 for all (i,j,k)."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :]))
                if abs(val - 1.0) > tol:
                    return False
    return True


def _verify_exact(U, V, W, n, m, p):
    """Exact verification using Fraction."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for r in range(rank):
                    total += (Fraction(U[i*m+j, r]).limit_denominator(10**6)
                              * Fraction(V[j*p+k, r]).limit_denominator(10**6)
                              * Fraction(W[k*n+i, r]).limit_denominator(10**6))
                if total != 1:
                    return False
    return True


def _snap_to_half_integer(matrix):
    out = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if abs(v) < 1e-9:
                out[i, j] = 0.0
            else:
                out[i, j] = round(v * 2.0) / 2.0
    return out


def _als_reduce(U0, V0, W0, target_rank, n, m, p, max_iter=200, tol=1e-8, time_budget=None):
    """ALS rank reduction. Returns (U, V, W, residual) or None."""
    rank = target_rank
    U = U0[:, :rank].copy()
    V = V0[:, :rank].copy()
    W = W0[:, :rank].copy()

    n, m, p = 2, 4, 8
    total = n * m * p

    def residual(U, V, W):
        prod = U[:total, :] * V[:total, :]  # placeholder, will fix
        # Actually need proper indexing
        return float('inf')

    # Precompute index arrays
    idx_i = np.repeat(np.arange(n), m * p)
    idx_j = np.tile(np.repeat(np.arange(m), p), n)
    idx_k = np.tile(np.arange(p), n * m)
    u_idx = idx_i * m + idx_j
    v_idx = idx_j * p + idx_k
    w_idx = idx_k * n + idx_i

    def residual(U, V, W):
        prod = U[u_idx, :] * V[v_idx, :]
        val = prod @ W[w_idx, :]
        return float(np.sum((val - 1.0) ** 2))

    best_res = residual(U, V, W)
    start = time.time()

    for it in range(max_iter):
        if time_budget is not None and time.time() > time_budget:
            break
        # Update W
        M_W = np.zeros((p * n, rank))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    M_W[k * n + i, :] += U[i * m + j, :] * V[j * p + k, :]
        try:
            w_vec, *_ = np.linalg.lstsq(M_W, np.ones(p * n), rcond=None)
            W = w_vec.reshape(rank, p * n).T
        except np.linalg.LinAlgError:
            break
        # Update U
        M_U = np.zeros((n * m, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M_U[i * m + j, :] += V[j * p + k, :] * W[k * n + i, :]
        try:
            u_vec, *_ = np.linalg.lstsq(M_U, np.ones(n * m), rcond=None)
            U = u_vec.reshape(rank, n * m).T
        except np.linalg.LinAlgError:
            break
        # Update V
        M_V = np.zeros((m * p, rank))
        for j in range(m):
            for k in range(p):
                for i in range(n):
                    M_V[j * p + k, :] += U[i * m + j, :] * W[k * n + i, :]
        try:
            v_vec, *_ = np.linalg.lstsq(M_V, np.ones(m * p), rcond=None)
            V = v_vec.reshape(rank, m * p).T
        except np.linalg.LinAlgError:
            break

        res = residual(U, V, W)
        if res < best_res:
            best_res = res
            if res < tol:
                break

    return U, V, W, best_res


def _try_snap_and_verify(U, V, W, n, m, p):
    Us = _snap_to_half_integer(U)
    Vs = _snap_to_half_integer(V)
    Ws = _snap_to_half_integer(W)
    if _verify_decomposition(Us, Vs, Ws, n, m, p, tol=1e-9):
        if _verify_exact(Us, Vs, Ws, n, m, p):
            return Us, Vs, Ws
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    U0, V0, W0 = _build_strassen_rank56()
    if not _verify_decomposition(U0, V0, W0, n, m, p):
        raise RuntimeError("Baseline decomposition failed verification")

    best_rank = 56
    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    np.random.seed(42)
    for target_rank in range(55, 40, -1):
        if time.time() > deadline - 20:
            break
        for trial in range(4):
            if time.time() > deadline - 25:
                break
            perm = np.random.permutation(56)
            noise = 0.05
            U_init = U0[:, perm[:target_rank]] * (1 + noise * np.random.randn(target_rank))
            V_init = V0[:, perm[:target_rank]] * (1 + noise * np.random.randn(target_rank))
            W_init = W0[:, perm[:target_rank]] * (1 + noise * np.random.randn(target_rank))
            U, V, W, res = _als_reduce(U_init, V_init, W_init, target_rank, n, m, p,
                                        max_iter=100, tol=1e-8, time_budget=deadline - 30)
            if res < 1e-6:
                snapped = _try_snap_and_verify(U, V, W, n, m, p)
                if snapped is not None:
                    Us, Vs, Ws = snapped
                    best_rank = target_rank
                    best_result = {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
                    break
    return best_result
# EVOLVE-BLOCK-END
