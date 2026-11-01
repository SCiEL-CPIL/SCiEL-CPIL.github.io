"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.

Combined approach:
1. Structured ALS (batched least-squares per factor) with multiple initializations
2. Exact rational polish pass (half-integer / small-denominator snapping + Fraction verify)
3. Direct small-integer/half-integer search for low ranks
4. Schoolbook fallback (rank 20)

The output factor W follows the official trace convention W[k*n+i, r].
"""

import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 5]


def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
    return T


def _decompose(U, V, W):
    """Reconstruct tensor from factor matrices."""
    R = U.shape[1]
    result = np.zeros((U.shape[0], V.shape[0], W.shape[0]), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return result


def _check_decomposition(U, V, W, T, tol=1e-8):
    result = _decompose(U, V, W)
    return np.allclose(result, T, atol=tol)


def _check_exact(U, V, W, T):
    """Exact integer/half-integer verification using Fraction."""
    def to_frac(x):
        # Try multiple denominators to find the best rational approximation
        for denom in [1, 2, 4, 8, 16, 32, 64]:
            num = round(x * denom)
            candidate = Fraction(int(num), denom)
            if abs(x - float(candidate)) < 1e-9:
                return candidate
        # Fallback: use limit_denominator
        f = Fraction(x).limit_denominator(128)
        return f

    Uf = [[to_frac(x) for x in row] for row in U]
    Vf = [[to_frac(x) for x in row] for row in V]
    Wf = [[to_frac(x) for x in row] for row in W]
    R = U.shape[1]
    for u in range(U.shape[0]):
        for v in range(V.shape[0]):
            for w in range(W.shape[0]):
                total = Fraction(0)
                for r in range(R):
                    total += Uf[u][r] * Vf[v][r] * Wf[w][r]
                if total != Fraction(int(T[u, v, w])):
                    return False
    return True


def _snap(M, denom):
    return np.round(M * denom) / denom


def _polish_rational(U, V, W, T):
    """Try to snap U,V,W to small-denominator rationals that still decompose exactly.
    Returns (U,V,W) as float lists if a valid snap is found, else None."""
    # Try a range of denominators with tighter tolerance
    for denom in [1, 2, 4, 8, 16, 32, 64]:
        U_s = _snap(U, denom)
        V_s = _snap(V, denom)
        W_s = _snap(W, denom)
        if _check_decomposition(U_s, V_s, W_s, T, tol=1e-6):
            if _check_exact(U_s, V_s, W_s, T):
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
    
    # Try mixed denominators: snap each matrix to its own best denominator
    for dU in [1, 2, 4]:
        for dV in [1, 2, 4]:
            for dW in [1, 2, 4]:
                U_s = _snap(U, dU)
                V_s = _snap(V, dV)
                W_s = _snap(W, dW)
                if _check_exact(U_s, V_s, W_s, T):
                    return U_s.tolist(), V_s.tolist(), W_s.tolist()
    
    # Local perturbation search: try small adjustments around the current solution
    # This helps when the solution is close to rational but not quite snapped
    for scale in [0.1, 0.05, 0.01]:
        for trial in range(20):
            U_p = U + np.random.randn(*U.shape) * scale
            V_p = V + np.random.randn(*V.shape) * scale
            W_p = W + np.random.randn(*W.shape) * scale
            for denom in [1, 2, 4]:
                U_s = _snap(U_p, denom)
                V_s = _snap(V_p, denom)
                W_s = _snap(W_p, denom)
                if _check_exact(U_s, V_s, W_s, T):
                    return U_s.tolist(), V_s.tolist(), W_s.tolist()
    
    return None


def _als_update_U(T, V, W):
    """Solve for U given V, W using least squares (batched over u)."""
    R = V.shape[1]
    # For each u: sum_r U[u,r] * (V[v,r]*W[w,r]) = T[u,v,w]
    # Matrix A has rows (v,w), cols r: A[(v,w), r] = V[v,r]*W[w,r]
    A = np.einsum('vr,wr->vwr', V, W).reshape(-1, R)
    U = np.zeros((T.shape[0], R), dtype=np.float64)
    for u in range(T.shape[0]):
        b = T[u].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        U[u] = sol
    return U


def _als_update_V(T, U, W):
    R = U.shape[1]
    A = np.einsum('ur,wr->uwr', U, W).reshape(-1, R)
    V = np.zeros((T.shape[1], R), dtype=np.float64)
    for v in range(T.shape[1]):
        b = T[:, v, :].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        V[v] = sol
    return V


def _als_update_W(T, U, V):
    R = U.shape[1]
    A = np.einsum('ur,vr->uvr', U, V).reshape(-1, R)
    W = np.zeros((T.shape[2], R), dtype=np.float64)
    for w in range(T.shape[2]):
        b = T[:, :, w].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        W[w] = sol
    return W


def _compute_error(U, V, W, T):
    result = _decompose(U, V, W)
    return np.linalg.norm(result - T.astype(np.float64))


def _try_als_rank(R, T, deadline, n_seeds=5, max_iters=100):
    n, m, p = DIMENSIONS
    U_rows, V_rows, W_rows = n * m, m * p, p * n

    seeds = [42, 123, 7, 999, 2024, 555, 777, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]

    for seed in seeds[:n_seeds]:
        if time.time() > deadline:
            break
        np.random.seed(seed)

        inits = []
        # Random small
        inits.append((np.random.randn(U_rows, R) * 0.3,
                      np.random.randn(V_rows, R) * 0.3,
                      np.random.randn(W_rows, R) * 0.3))
        # Random medium
        inits.append((np.random.randn(U_rows, R) * 1.0,
                      np.random.randn(V_rows, R) * 1.0,
                      np.random.randn(W_rows, R) * 1.0))
        # Structured schoolbook-based init
        U_init = np.zeros((U_rows, R))
        V_init = np.zeros((V_rows, R))
        W_init = np.zeros((W_rows, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= R:
                        break
                    U_init[i * m + j, col] = 1
                    V_init[j * p + k, col] = 1
                    W_init[k * n + i, col] = 1
                    col += 1
        if col < R:
            U_init[:, col:R] = np.random.randn(U_rows, R - col) * 0.1
            V_init[:, col:R] = np.random.randn(V_rows, R - col) * 0.1
            W_init[:, col:R] = np.random.randn(W_rows, R - col) * 0.1
        inits.append((U_init, V_init, W_init))
        
        # Add a sparse random initialization
        U_init_sparse = np.zeros((U_rows, R))
        V_init_sparse = np.zeros((V_rows, R))
        W_init_sparse = np.zeros((W_rows, R))
        for r in range(R):
            # Each column has a few non-zero entries
            nz_u = np.random.choice(U_rows, size=min(3, U_rows), replace=False)
            nz_v = np.random.choice(V_rows, size=min(3, V_rows), replace=False)
            nz_w = np.random.choice(W_rows, size=min(3, W_rows), replace=False)
            U_init_sparse[nz_u, r] = np.random.choice([-1, 1], size=len(nz_u))
            V_init_sparse[nz_v, r] = np.random.choice([-1, 1], size=len(nz_v))
            W_init_sparse[nz_w, r] = np.random.choice([-1, 1], size=len(nz_w))
        inits.append((U_init_sparse, V_init_sparse, W_init_sparse))

        for U0, V0, W0 in inits:
            if time.time() > deadline:
                break
            U, V, W = U0.copy(), V0.copy(), W0.copy()
            prev_err = np.inf
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                U = _als_update_U(T, V, W)
                V = _als_update_V(T, U, W)
                W = _als_update_W(T, U, V)
                err = _compute_error(U, V, W, T)
                if abs(prev_err - err) < 1e-14:
                    break
                prev_err = err

            # High-precision polish: a few extra tight iterations
            for _ in range(30):
                if time.time() > deadline:
                    break
                U = _als_update_U(T, V, W)
                V = _als_update_V(T, U, W)
                W = _als_update_W(T, U, V)

            if _check_decomposition(U, V, W, T, tol=1e-7):
                polished = _polish_rational(U, V, W, T)
                if polished is not None:
                    return polished
                # Return as floats if exact-enough
                return U.tolist(), V.tolist(), W.tolist()
    return None


def _try_direct_small_rank(R, T, deadline):
    n, m, p = DIMENSIONS
    U_rows, V_rows, W_rows = n * m, m * p, p * n
    if R > 12:
        return None

    # Use a richer value set including half-integers
    val_set = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    np.random.seed(42)
    T_f = T.astype(np.float64)

    for attempt in range(20):
        if time.time() > deadline:
            break
        U = np.zeros((U_rows, R))
        V = np.zeros((V_rows, R))
        W = np.zeros((W_rows, R))
        success = True
        for r in range(R):
            if time.time() > deadline:
                success = False
                break
            best_col_err = np.inf
            best_u = best_v = best_w = None
            partial = np.zeros_like(T_f)
            for r2 in range(r):
                partial += np.einsum('u,v,w->uvw', U[:, r2], V[:, r2], W[:, r2])
            # Greedy column selection with more samples
            for sample in range(60):
                u_col = np.random.choice(val_set, size=U_rows)
                v_col = np.random.choice(val_set, size=V_rows)
                w_col = np.random.choice(val_set, size=W_rows)
                new_sum = partial + np.einsum('u,v,w->uvw', u_col, v_col, w_col)
                err = np.linalg.norm(new_sum - T_f)
                if err < best_col_err:
                    best_col_err = err
                    best_u, best_v, best_w = u_col.copy(), v_col.copy(), w_col.copy()
            if best_u is not None:
                U[:, r] = best_u
                V[:, r] = best_v
                W[:, r] = best_w
            else:
                success = False
                break
        if not success:
            continue
        if _check_decomposition(U, V, W, T, tol=1e-7):
            return U.tolist(), V.tolist(), W.tolist()
    return None


def _schoolbook_decomposition():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    T = _build_target()

    best_result = None
    best_rank = n * m * p + 1

    # Try ranks from low to high; return the first (lowest) valid one found.
    # Start from rank 7 since lower ranks are theoretically impossible for this problem
    for R in range(7, n * m * p + 1):
        if time.time() > deadline:
            break

        # Direct small-integer search for very low ranks
        if R <= 12:
            result = _try_direct_small_rank(R, T, deadline)
            if result is not None:
                U_l, V_l, W_l = result
                return {"U": U_l, "V": V_l, "W": W_l}

        # ALS with rational polish
        # Use more seeds and iterations for critical rank range
        if R <= 10:
            n_seeds = 8
            max_iters = 120
        elif R <= 14:
            n_seeds = 5
            max_iters = 100
        else:
            n_seeds = 3
            max_iters = 80
        result = _try_als_rank(R, T, deadline, n_seeds=n_seeds, max_iters=max_iters)
        if result is not None:
            U_l, V_l, W_l = result
            return {"U": U_l, "V": V_l, "W": W_l}

        # Track best (lowest) rank reached via ALS even if not exact, to be safe
        if R < best_rank:
            best_rank = R

    # Fall back to schoolbook (rank 20)
    U_l, V_l, W_l = _schoolbook_decomposition()
    return {"U": U_l, "V": V_l, "W": W_l}
