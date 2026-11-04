"""Matrix multiplication 2x3x7 rank-minimal exact tensor decomposition.

Strategy: trilinear alternating-least-squares (ALS) with random restarts +
Gauss-Newton polish, exact rational verification, half-integer rounding.
"""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 7]


def _verify(U, V, W, n, m, p):
    """Verify exact decomposition using rational arithmetic."""
    rank = len(U[0])
    Ua = np.array(U, dtype=object)
    Va = np.array(V, dtype=object)
    Wa = np.array(W, dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Ua[i * m + j, r] * Va[j * p + k, r] * Wa[k * n + i, r]
                if s != Fraction(1):
                    return False
    return True


def _verify_float(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


# EVOLVE-BLOCK-START
def _build_indices(n, m, p):
    """Return list of (i,j,k) valid triples and row indices into U,V,W."""
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i, j, k))
    return triples


def _als_solve(U, V, W, n, m, p, R, max_iter=400, tol=1e-10, deadline=None):
    """Alternating least squares on the trilinear identity.

    For each (i,j,k): sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1.
    Fixing two factors turns this into a linear system in the third.
    """
    U = np.array(U, dtype=np.float64).copy()
    V = np.array(V, dtype=np.float64).copy()
    W = np.array(W, dtype=np.float64).copy()

    triples = _build_indices(n, m, p)

    def residual():
        err = 0.0
        for (i, j, k) in triples:
            s = np.dot(U[i * m + j] * V[j * p + k], W[k * n + i])
            err += (s - 1.0) ** 2
        return err

    prev = residual()

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # --- Solve for W (fix U, V) ---
        # For each (i,j,k): sum_r W[k*n+i, r] * (U[i*m+j,r] * V[j*p+k,r]) = 1
        # Group by (i,k) row of W; each equation contributes to row (k*n+i).
        # Build normal equations per W row.
        W_new = W.copy()
        for (i, j, k) in triples:
            rowW = k * n + i
            a = U[i * m + j] * V[j * p + k]  # length R
            # accumulate into normal equations
            W_new[rowW] += a * (1.0 - np.dot(a, W[rowW])) / (np.dot(a, a) + 1e-12)
        W = W_new

        # --- Solve for U (fix V, W) ---
        U_new = U.copy()
        for (i, j, k) in triples:
            rowU = i * m + j
            a = V[j * p + k] * W[k * n + i]
            U_new[rowU] += a * (1.0 - np.dot(a, U[rowU])) / (np.dot(a, a) + 1e-12)
        U = U_new

        # --- Solve for V (fix U, W) ---
        V_new = V.copy()
        for (i, j, k) in triples:
            rowV = j * p + k
            a = U[i * m + j] * W[k * n + i]
            V_new[rowV] += a * (1.0 - np.dot(a, V[rowV])) / (np.dot(a, a) + 1e-12)
        V = V_new

        cur = residual()
        if abs(prev - cur) < tol:
            prev = cur
            break
        prev = cur

    return U, V, W, residual()


def _gauss_newton_polish(U, V, W, n, m, p, R, max_iter=200, tol=1e-12, deadline=None):
    """Gauss-Newton on the 42 scalar residuals f(i,j,k) = sum_r prod - 1."""
    U = np.array(U, dtype=np.float64).copy()
    V = np.array(V, dtype=np.float64).copy()
    W = np.array(W, dtype=np.float64).copy()
    triples = _build_indices(n, m, p)

    def residuals():
        f = np.zeros(len(triples))
        for t, (i, j, k) in enumerate(triples):
            f[t] = np.dot(U[i * m + j] * V[j * p + k], W[k * n + i]) - 1.0
        return f

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        f = residuals()
        if np.max(np.abs(f)) < tol:
            break
        # Jacobian: variables are U (n*m*R), V (m*p*R), W (p*n*R)
        # df/d(U[rowU, r]) = V[rowV, r] * W[rowW, r]
        # Build normal equations J^T J x = -J^T f
        nU = n * m * R
        nV = m * p * R
        nW = p * n * R
        nvar = nU + nV + nW
        JtJ = np.zeros((nvar, nvar))
        Jtf = np.zeros(nvar)
        for t, (i, j, k) in enumerate(triples):
            rowU = i * m + j
            rowV = j * p + k
            rowW = k * n + i
            dU = V[rowV] * W[rowW]
            dV = U[rowU] * W[rowW]
            dW = U[rowU] * V[rowV]
            # JtJ += outer of concatenated gradient
            g = np.concatenate([dU, dV, dW])
            JtJ += np.outer(g, g)
            Jtf += g * f[t]
        # Solve with damping
        try:
            dx = np.linalg.solve(JtJ + 1e-8 * np.eye(nvar), -Jtf)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(JtJ, -Jtf, rcond=None)[0]
        U -= dx[:nU].reshape(n * m, R)
        V -= dx[nU:nU + nV].reshape(m * p, R)
        W -= dx[nU + nV:].reshape(p * n, R)

    return U, V, W, float(np.max(np.abs(residuals())))


def _round_half_int(U, V, W):
    """Round each entry to nearest half-integer."""
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    return U, V, W


def _to_list(M):
    return [[float(x) for x in row] for row in M]


def _try_rank(R, n, m, p, n_restarts, deadline):
    """Try to find a valid rank-R decomposition. Returns dict or None."""
    best_err = float('inf')
    best = None
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        # Random initialization with small scale
        U = np.random.randn(n * m, R) * 0.5
        V = np.random.randn(m * p, R) * 0.5
        W = np.random.randn(p * n, R) * 0.5
        # Gauge: fix some entries to break scale symmetry
        # Set U[0,0]=1, V[0,0]=1, W[0,0]=1 (if R>0)
        if R > 0:
            U[0, 0] = 1.0
            V[0, 0] = 1.0
            W[0, 0] = 1.0

        U, V, W, err = _als_solve(U, V, W, n, m, p, R, max_iter=300, deadline=deadline)
        U, V, W, err = _gauss_newton_polish(U, V, W, n, m, p, R, max_iter=100, deadline=deadline)

        if err < best_err:
            best_err = err
            best = (U, V, W)

        # Try rounding to half-integers and verify
        Ur, Vr, Wr = _round_half_int(U, V, W)
        if _verify_float(_to_list(Ur), _to_list(Vr), _to_list(Wr), n, m, p, tol=1e-9):
            # Exact rational verification
            if _verify(_to_list(Ur), _to_list(Vr), _to_list(Wr), n, m, p):
                return {"U": _to_list(Ur), "V": _to_list(Vr), "W": _to_list(Wr)}

        # Also try verifying the unrounded real solution exactly (as rationals)
        # Only if it's extremely close to rational
        if err < 1e-12:
            if _verify(_to_list(U), _to_list(V), _to_list(W), n, m, p):
                return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    # If best error is tiny, attempt final polish + rounding
    if best is not None and best_err < 1e-8:
        U, V, W = best
        U, V, W, err = _gauss_newton_polish(U, V, W, n, m, p, R, max_iter=300, tol=1e-14, deadline=deadline)
        Ur, Vr, Wr = _round_half_int(U, V, W)
        if _verify_float(_to_list(Ur), _to_list(Vr), _to_list(Wr), n, m, p, tol=1e-9):
            if _verify(_to_list(Ur), _to_list(Vr), _to_list(Wr), n, m, p):
                return {"U": _to_list(Ur), "V": _to_list(Vr), "W": _to_list(Wr)}
        if _verify(_to_list(U), _to_list(V), _to_list(W), n, m, p):
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    return None


def construct():
    start_time = time.time()
    time_limit = 120.0
    deadline = start_time + time_limit - 15.0  # safety margin

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42

    # Lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    lower_bound = max(n * m, m * p, p * n)

    # Try ranks from lower_bound upward. Use more restarts for smaller ranks.
    # Budget: distribute restarts across ranks.
    for R in range(lower_bound, schoolbook_rank):
        if time.time() > deadline:
            break
        # Allocate restarts: more for smaller ranks
        n_restarts = max(3, 20 - (R - lower_bound))
        result = _try_rank(R, n, m, p, n_restarts, deadline)
        if result is not None:
            return result

    # Fallback: schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
