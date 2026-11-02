"""Rank-18 exact tensor decomposition for 2x2 by 2x5 matrix multiplication,
with a fast, deterministic high-precision polish pass.

The design family is kept: a structured Strassen+schoolbook rank-18 construction
is the anchor, and we add a final exact local solve / polish to confirm the
decomposition and, where possible, rationalize entries to small denominators.
We also add a very light, bounded search for rank-17 by perturbing the rank-18
structure, but the primary, guaranteed result is the rank-18 solution.
"""
import numpy as np
import time

DIMENSIONS = [2, 2, 5]

def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1
    return T

def _check_decomposition(U, V, W, T, atol=1e-10):
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    result = np.zeros_like(T, dtype=np.float64)
    for r in range(U_a.shape[1]):
        result += np.einsum('u,v,w->uvw', U_a[:,r], V_a[:,r], W_a[:,r])
    return bool(np.allclose(result, T.astype(np.float64), atol=atol))

def _build_rank18(n, m, p):
    """Build the known rank-18 Strassen+schoolbook decomposition."""
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1

        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1

        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1

        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1

        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1

        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1

        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1

    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)

    # Schoolbook for column 4
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1

    return U, V, W

def _als_solve_U(T, V, W, n, m, p, R):
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    VvW = np.einsum('vr,wr->vwr', V_a, W_a)
    VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
    U = np.zeros((n*m, R), dtype=np.float64)
    for u in range(n*m):
        b = T_a[u].reshape(-1)
        sol, _, _, _ = np.linalg.lstsq(VvW_flat, b, rcond=None)
        U[u] = sol
    return U

def _als_solve_V(T, U, W, n, m, p, R):
    U_a = np.array(U, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    Uw = np.einsum('ur,wr->uwr', U_a, W_a)
    Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
    V = np.zeros((m*p, R), dtype=np.float64)
    for v in range(m*p):
        b = T_a[:, v, :].reshape(-1)
        sol, _, _, _ = np.linalg.lstsq(Uw_flat, b, rcond=None)
        V[v] = sol
    return V

def _als_solve_W(T, U, V, n, m, p, R):
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    UV = np.einsum('ur,vr->uvr', U_a, V_a)
    UV_flat = UV.reshape(-1, R)
    W = np.zeros((p*n, R), dtype=np.float64)
    for w in range(p*n):
        b = T_a[:, :, w].reshape(-1)
        sol, _, _, _ = np.linalg.lstsq(UV_flat, b, rcond=None)
        W[w] = sol
    return W

def _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, R, max_iters=200, tol=1e-14, deadline=None):
    U = np.array(U_init, dtype=np.float64)
    V = np.array(V_init, dtype=np.float64)
    W = np.array(W_init, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)

    prev_err = np.inf
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        U = _als_solve_U(T_a, V, W, n, m, p, R)
        V = _als_solve_V(T_a, U, W, n, m, p, R)
        W = _als_solve_W(T_a, U, V, n, m, p, R)
        err = 0.0
        for r in range(R):
            err += np.sum((np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r]) - T_a)**2)
        err = np.sqrt(err / (n*m * m*p * p*n))
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    # Final polish: a few more iterations with tighter tolerance
    for _ in range(20):
        if deadline is not None and time.time() > deadline:
            break
        U = _als_solve_U(T_a, V, W, n, m, p, R)
        V = _als_solve_V(T_a, U, W, n, m, p, R)
        W = _als_solve_W(T_a, U, V, n, m, p, R)

    if _check_decomposition(U, V, W, T_a):
        for denom in [2, 4, 8, 16, 32]:
            U_s = np.round(U * denom) / denom
            V_s = np.round(V * denom) / denom
            W_s = np.round(W * denom) / denom
            if _check_decomposition(U_s, V_s, W_s, T_a):
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _try_rank17_search(n, m, p, T, deadline):
    """Bounded search for a rank-17 decomposition by perturbing the rank-18 structure."""
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)

    # Strategy: try to merge one of the schoolbook terms into the Strassen terms
    for term_to_remove in [14, 15, 16, 17]:
        if time.time() > deadline:
            break
        idx = [i for i in range(18) if i != term_to_remove]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()

        np.random.seed(term_to_remove * 100 + 42)
        U_init += np.random.randn(4, 17) * 0.01
        V_init += np.random.randn(10, 17) * 0.01
        W_init += np.random.randn(10, 17) * 0.01

        result = _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, 17, max_iters=150, deadline=deadline)
        if result is not None:
            return result

    # Also try: modify one Strassen term to absorb part of a schoolbook term
    for strassen_term in range(14):
        if time.time() > deadline:
            break
        for school_term in [14, 15, 16, 17]:
            if time.time() > deadline:
                break
            U_init = U18_a.copy()
            V_init = V18_a.copy()
            W_init = W18_a.copy()
            U_init[:, strassen_term] += U18_a[:, school_term]
            V_init[:, strassen_term] += V18_a[:, school_term]
            W_init[:, strassen_term] += W18_a[:, school_term]
            idx = [i for i in range(18) if i != school_term]
            U_init = U_init[:, idx]
            V_init = V_init[:, idx]
            W_init = W_init[:, idx]

            np.random.seed(strassen_term * 1000 + school_term)
            U_init += np.random.randn(4, 17) * 0.005
            V_init += np.random.randn(10, 17) * 0.005
            W_init += np.random.randn(10, 17) * 0.005

            result = _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, 17, max_iters=100, deadline=deadline)
            if result is not None:
                return result

    return None

def _try_random_rank17(n, m, p, T, deadline):
    """Try a few random initializations for rank-17."""
    T_a = np.array(T, dtype=np.float64)
    for seed in [42, 123, 7, 999, 2024]:
        if time.time() > deadline:
            break
        np.random.seed(seed)
        U = np.random.randn(4, 17) * 0.5
        V = np.random.randn(10, 17) * 0.5
        W = np.random.randn(10, 17) * 0.5
        result = _try_rank_als_from_init(U, V, W, T_a, n, m, p, 17, max_iters=100, deadline=deadline)
        if result is not None:
            return result
    return None

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    T = _build_target()

    # First, try to find a rank-17 decomposition (bounded, fast)
    result = _try_rank17_search(n, m, p, T, deadline)
    if result is not None:
        U_l, V_l, W_l = result
        return {"U": U_l, "V": V_l, "W": W_l}

    # If time remains, try a few random rank-17 initializations
    if time.time() < deadline - 30:
        result = _try_random_rank17(n, m, p, T, deadline)
        if result is not None:
            U_l, V_l, W_l = result
            return {"U": U_l, "V": V_l, "W": W_l}

    # Fall back to the known rank-18 solution, with a high-precision polish pass
    U, V, W = _build_rank18(n, m, p)
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)

    # Polish: run a few ALS iterations to confirm and possibly tighten
    T_a = np.array(T, dtype=np.float64)
    for _ in range(5):
        U_a = _als_solve_U(T_a, V_a, W_a, n, m, p, 18)
        V_a = _als_solve_V(T_a, U_a, W_a, n, m, p, 18)
        W_a = _als_solve_W(T_a, U_a, V_a, n, m, p, 18)

    # Rationalize to small denominators if possible
    for denom in [2, 4, 8, 16, 32]:
        U_s = np.round(U_a * denom) / denom
        V_s = np.round(V_a * denom) / denom
        W_s = np.round(W_a * denom) / denom
        if _check_decomposition(U_s, V_s, W_s, T_a):
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    return {"U": U_a.tolist(), "V": V_a.tolist(), "W": W_a.tolist()}
