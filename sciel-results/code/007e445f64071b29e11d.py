import time
import numpy as np
from scipy.optimize import least_squares, minimize

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    # Dimensions
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    num_eq = n * m * p  # 84

    def get_residuals(U, V, W):
        """Compute residuals for given U, V, W matrices."""
        res = np.zeros(num_eq)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m+j, :]
                    v_val = V[j*p+k, :]
                    w_val = W[k*n+i, :]
                    res[idx] = np.sum(u_val * v_val * w_val) - 1.0
                    idx += 1
        return res

    def flatten_UVW(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    def unflatten_UVW(x, r):
        U = x[:dim_U * r].reshape(dim_U, r)
        V = x[dim_U * r : dim_U * r + dim_V * r].reshape(dim_V, r)
        W = x[dim_U * r + dim_V * r :].reshape(dim_W, r)
        return U, V, W

    def verify(U, V, W, tol=1e-8):
        """Verify exact decomposition within tolerance."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1.0))
        return max_err < tol

    def snap_to_half_integers(arr):
        return np.round(arr * 2.0) / 2.0

    # Start with schoolbook decomposition (rank 84)
    rank_school = n * m * p
    U_school = np.zeros((dim_U, rank_school))
    V_school = np.zeros((dim_V, rank_school))
    W_school = np.zeros((dim_W, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1

    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = rank_school

    def try_remove_term(U, V, W, r, term_idx):
        """Try to remove term_idx and re-solve for rank r-1."""
        if time.time() > deadline:
            return None

        r_new = r - 1
        U_new = np.delete(U, term_idx, axis=1)
        V_new = np.delete(V, term_idx, axis=1)
        W_new = np.delete(W, term_idx, axis=1)

        x0 = flatten_UVW(U_new, V_new, W_new)

        def fun(x):
            U_t, V_t, W_t = unflatten_UVW(x, r_new)
            return get_residuals(U_t, V_t, W_t)

        try:
            sol = least_squares(fun, x0, method='trf', max_nfev=200, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if sol.success or sol.cost < 1e-12:
                U_sol, V_sol, W_sol = unflatten_UVW(sol.x, r_new)
                if verify(U_sol, V_sol, W_sol, tol=1e-8):
                    return U_sol, V_sol, W_sol
        except Exception:
            pass
        return None

    def greedy_reduce(U, V, W, r, max_rounds=50):
        """Greedy term removal."""
        U_curr, V_curr, W_curr = U.copy(), V.copy(), W.copy()
        r_curr = r
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            for t in range(r_curr):
                if time.time() > deadline:
                    break
                result = try_remove_term(U_curr, V_curr, W_curr, r_curr, t)
                if result is not None:
                    U_curr, V_curr, W_curr = result
                    r_curr -= 1
                    improved = True
                    break
            if not improved:
                break
        return U_curr, V_curr, W_curr, r_curr

    # Attempt greedy reduction from schoolbook
    if time.time() < deadline:
        U_red, V_red, W_red, r_red = greedy_reduce(U_school, V_school, W_school, rank_school)
        if r_red < best_rank:
            best_U, best_V, best_W, best_rank = U_red, V_red, W_red, r_red

    # Try random restarts at lower ranks to find better decompositions
    # The tensor for 3x4x7 matrix multiplication has known lower bounds.
    # Strassen-like algorithms suggest rank can be significantly lower.
    # Let's try to find decompositions at various target ranks.

    target_ranks = [60, 50, 45, 40, 36, 32, 30, 28, 25, 22, 20, 18, 16, 14, 12, 10]

    for target_r in target_ranks:
        if time.time() > deadline - 20:
            break
        if target_r >= best_rank:
            continue

        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break

            U0 = np.random.randn(dim_U, target_r) * 0.1
            V0 = np.random.randn(dim_V, target_r) * 0.1
            W0 = np.random.randn(dim_W, target_r) * 0.1

            x0 = flatten_UVW(U0, V0, W0)

            def fun(x):
                U_t, V_t, W_t = unflatten_UVW(x, target_r)
                return get_residuals(U_t, V_t, W_t)

            try:
                sol = least_squares(fun, x0, method='trf', max_nfev=500, xtol=1e-12, ftol=1e-12, gtol=1e-12)
                if sol.success or sol.cost < 1e-12:
                    U_sol, V_sol, W_sol = unflatten_UVW(sol.x, target_r)
                    if verify(U_sol, V_sol, W_sol, tol=1e-8):
                        best_U, best_V, best_W, best_rank = U_sol, V_sol, W_sol, target_r
                        break  # Found a valid decomposition at this rank, try lower
            except Exception:
                pass

    # Try to snap the best solution to half-integers
    U_snap = snap_to_half_integers(best_U)
    V_snap = snap_to_half_integers(best_V)
    W_snap = snap_to_half_integers(best_W)

    if verify(U_snap, V_snap, W_snap, tol=1e-6):
        best_U = U_snap
        best_V = V_snap
        best_W = W_snap

    # Return as lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
