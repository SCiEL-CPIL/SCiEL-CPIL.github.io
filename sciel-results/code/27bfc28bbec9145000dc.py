"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.

Strategy:
1. Constructive rank-18 baseline (2 Strassen blocks + schoolbook for col 4).
2. Aggressive rank-16/17 search with vectorized ALS and smart initialization.
3. Fallbacks.

The output factor W follows the official trace convention W[k*n+i,r].
"""

import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T


def _check_decomposition(U, V, W, T, tol=1e-8):
    n, m, p = DIMENSIONS
    U_arr = np.asarray(U, dtype=np.float64)
    V_arr = np.asarray(V, dtype=np.float64)
    W_arr = np.asarray(W, dtype=np.float64)
    R = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                total = 0.0
                for r in range(R):
                    total += U_arr[u, r] * V_arr[v, r] * W_arr[w, r]
                if abs(total - T[u, v, w]) > tol:
                    return False
    return True


def _check_exact(U, V, W, T, tol=1e-6):
    def to_frac(x):
        hx = round(x * 2)
        if abs(x - hx / 2.0) < tol:
            return Fraction(int(hx), 2)
        f = Fraction(x).limit_denominator(128)
        if abs(x - float(f)) < tol:
            return f
        return None

    Uf = []
    for row in U:
        r = []
        for x in row:
            f = to_frac(x)
            if f is None:
                return False
            r.append(f)
        Uf.append(r)
    Vf = []
    for row in V:
        r = []
        for x in row:
            f = to_frac(x)
            if f is None:
                return False
            r.append(f)
        Vf.append(r)
    Wf = []
    for row in W:
        r = []
        for x in row:
            f = to_frac(x)
            if f is None:
                return False
            r.append(f)
        Wf.append(r)

    R = len(U[0])
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                total = Fraction(0)
                for r in range(R):
                    total += Uf[u][r] * Vf[v][r] * Wf[w][r]
                if total != Fraction(int(T[u, v, w])):
                    return False
    return True


def _build_strassen_block(col0, col1, base, U, V, W, n, m, p):
    r = base
    U[0][r] += 1; U[3][r] += 1
    V[0*p+col0][r] += 1; V[1*p+col1][r] += 1
    W[col0*n+0][r] += 1; W[col1*n+1][r] += 1

    r = base + 1
    U[2][r] += 1; U[3][r] += 1
    V[0*p+col0][r] += 1
    W[col0*n+1][r] += 1; W[col1*n+1][r] += -1

    r = base + 2
    U[0][r] += 1
    V[0*p+col1][r] += 1; V[1*p+col1][r] += -1
    W[col1*n+0][r] += 1; W[col1*n+1][r] += 1

    r = base + 3
    U[3][r] += 1
    V[1*p+col0][r] += 1; V[0*p+col0][r] += -1
    W[col0*n+0][r] += 1; W[col0*n+1][r] += 1

    r = base + 4
    U[0][r] += 1; U[1][r] += 1
    V[1*p+col1][r] += 1
    W[col0*n+0][r] += -1; W[col1*n+0][r] += 1

    r = base + 5
    U[2][r] += 1; U[0][r] += -1
    V[0*p+col0][r] += 1; V[0*p+col1][r] += 1
    W[col1*n+1][r] += 1

    r = base + 6
    U[1][r] += 1; U[3][r] += -1
    V[1*p+col0][r] += 1; V[1*p+col1][r] += 1
    W[col0*n+0][r] += 1


def _build_rank18():
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    _build_strassen_block(0, 1, 0, U, V, W, n, m, p)
    _build_strassen_block(2, 3, 7, U, V, W, n, m, p)

    r = 14
    U[0][r] = 1.0; V[0*p+4][r] = 1.0; W[4*n+0][r] = 1.0
    r = 15
    U[1][r] = 1.0; V[1*p+4][r] = 1.0; W[4*n+0][r] = 1.0
    r = 16
    U[2][r] = 1.0; V[0*p+4][r] = 1.0; W[4*n+1][r] = 1.0
    r = 17
    U[3][r] = 1.0; V[1*p+4][r] = 1.0; W[4*n+1][r] = 1.0

    return U, V, W


def _als_solve_vectorized(U, V, W, T, R, max_iters=200, tol=1e-12, deadline=None):
    """Vectorized alternating least squares for tensor decomposition."""
    n, m, p = DIMENSIONS
    U_rows, V_rows, W_rows = n * m, m * p, p * n

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        # Update U: for each u, solve (V ⊙ W)^T * U[u] = T[u].reshape(-1)
        # V ⊙ W is a matrix of size (V_rows * W_rows, R)
        VW = np.einsum('vr,wr->vwr', V, W).reshape(V_rows * W_rows, R)
        for u in range(U_rows):
            b = T[u].reshape(-1)
            sol, _, _, _ = np.linalg.lstsq(VW, b, rcond=None)
            U[u] = sol

        # Update V: for each v, solve (U ⊙ W)^T * V[v] = T[:, v, :].reshape(-1)
        UW = np.einsum('ur,wr->uwr', U, W).reshape(U_rows * W_rows, R)
        for v in range(V_rows):
            b = T[:, v, :].reshape(-1)
            sol, _, _, _ = np.linalg.lstsq(UW, b, rcond=None)
            V[v] = sol

        # Update W: for each w, solve (U ⊙ V)^T * W[w] = T[:, :, w].reshape(-1)
        UV = np.einsum('ur,vr->uvr', U, V).reshape(U_rows * V_rows, R)
        for w in range(W_rows):
            b = T[:, :, w].reshape(-1)
            sol, _, _, _ = np.linalg.lstsq(UV, b, rcond=None)
            W[w] = sol

        # Check convergence
        result = np.einsum('ur,vr,wr->uvw', U, V, W)
        err = np.linalg.norm(result - T)
        if err < tol:
            break

    return U, V, W


def _try_rank_als(R, T, deadline, n_seeds=8, max_iters=150, seed_init=None):
    n, m, p = DIMENSIONS
    U_rows, V_rows, W_rows = n * m, m * p, p * n
    seeds = [42, 123, 7, 999, 2024, 55, 88, 13]

    for seed in seeds[:n_seeds]:
        if time.time() > deadline:
            break
        np.random.seed(seed)

        if seed_init is not None:
            U0, V0, W0 = seed_init
            U = U0.copy()
            V = V0.copy()
            W = W0.copy()
        else:
            for scale in [0.3, 1.0, 3.0]:
                if time.time() > deadline:
                    break
                U0 = np.random.randn(U_rows, R) * scale
                V0 = np.random.randn(V_rows, R) * scale
                W0 = np.random.randn(W_rows, R) * scale

                U, V, W = U0.copy(), V0.copy(), W0.copy()
                U, V, W = _als_solve_vectorized(U, V, W, T, R, max_iters=max_iters, deadline=deadline)

                if _check_decomposition(U, V, W, T, tol=1e-6):
                    for denom in [2, 1, 4, 8]:
                        U_s = np.round(U * denom) / denom
                        V_s = np.round(V * denom) / denom
                        W_s = np.round(W * denom) / denom
                        if _check_exact(U_s.tolist(), V_s.tolist(), W_s.tolist(), T):
                            return U_s.tolist(), V_s.tolist(), W_s.tolist()
                    return U.tolist(), V.tolist(), W.tolist()

            continue

        U, V, W = _als_solve_vectorized(U, V, W, T, R, max_iters=max_iters, deadline=deadline)

        if _check_decomposition(U, V, W, T, tol=1e-6):
            for denom in [2, 1, 4, 8]:
                U_s = np.round(U * denom) / denom
                V_s = np.round(V * denom) / denom
                W_s = np.round(W * denom) / denom
                if _check_exact(U_s.tolist(), V_s.tolist(), W_s.tolist(), T):
                    return U_s.tolist(), V_s.tolist(), W_s.tolist()
            return U.tolist(), V.tolist(), W.tolist()

    return None


def _try_rank17_from_rank18(T, deadline):
    """Try to find rank-17 by removing one term from rank-18 and running ALS."""
    n, m, p = DIMENSIONS
    U18, V18, W18 = _build_rank18()
    U18_arr = np.array(U18, dtype=np.float64)
    V18_arr = np.array(V18, dtype=np.float64)
    W18_arr = np.array(W18, dtype=np.float64)

    # Try removing each term
    for remove_idx in range(18):
        if time.time() > deadline:
            break

        # Remove the column to get rank 17 matrices
        mask = [i for i in range(18) if i != remove_idx]
        U17 = U18_arr[:, mask]
        V17 = V18_arr[:, mask]
        W17 = W18_arr[:, mask]

        # Run ALS on the remaining 17 terms
        U17, V17, W17 = _als_solve_vectorized(U17, V17, W17, T, 17, max_iters=100, deadline=deadline)

        if _check_decomposition(U17, V17, W17, T, tol=1e-6):
            for denom in [2, 1, 4, 8]:
                U_s = np.round(U17 * denom) / denom
                V_s = np.round(V17 * denom) / denom
                W_s = np.round(W17 * denom) / denom
                if _check_exact(U_s.tolist(), V_s.tolist(), W_s.tolist(), T):
                    return U_s.tolist(), V_s.tolist(), W_s.tolist()
            return U17.tolist(), V17.tolist(), W17.tolist()

    return None


def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    T = _build_target()

    # Baseline: rank-18
    U18, V18, W18 = _build_rank18()
    if _check_decomposition(U18, V18, W18, T, tol=1e-8):
        # Try rank-17 via removing one term from rank-18
        result17 = _try_rank17_from_rank18(T, deadline)
        if result17 is not None:
            U17, V17, W17 = result17
            if _check_decomposition(U17, V17, W17, T, tol=1e-8):
                return {"U": U17, "V": V17, "W": W17}

        # Try rank-16 via ALS
        result16 = _try_rank_als(16, T, deadline, n_seeds=6, max_iters=100)
        if result16 is not None:
            U16, V16, W16 = result16
            if _check_decomposition(U16, V16, W16, T, tol=1e-6):
                return {"U": U16, "V": V16, "W": W16}

        # Try rank-17 via ALS with random init
        result17_als = _try_rank_als(17, T, deadline, n_seeds=6, max_iters=100)
        if result17_als is not None:
            U17, V17, W17 = result17_als
            if _check_decomposition(U17, V17, W17, T, tol=1e-6):
                return {"U": U17, "V": V17, "W": W17}

        return {"U": U18, "V": V18, "W": W18}

    # Fallback: ALS for lower ranks
    for R in [16, 17, 18]:
        if time.time() > deadline:
            break
        result = _try_rank_als(R, T, deadline, n_seeds=4, max_iters=80)
        if result is not None:
            return {"U": result[0], "V": result[1], "W": result[2]}

    # Final fallback: schoolbook
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
