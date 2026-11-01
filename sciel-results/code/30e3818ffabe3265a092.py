import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill

    # Schoolbook rank-24 solution (always valid fallback)
    rank_seed = n * m * p
    U_seed = np.zeros((n * m, rank_seed), dtype=np.float64)
    V_seed = np.zeros((m * p, rank_seed), dtype=np.float64)
    W_seed = np.zeros((p * n, rank_seed), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_seed.copy(), V_seed.copy(), W_seed.copy()
    best_rank = rank_seed

    # Constraint list: for each (i,j,k) we need sum_k U[i*m+j,k]*V[j*p+k,k]*W[k*n+i,k] = 1
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                constraints.append((i * m + j, j * p + k, k * n + i))
    num_constraints = len(constraints)
    u_idxs = np.array([c[0] for c in constraints])
    v_idxs = np.array([c[1] for c in constraints])
    w_idxs = np.array([c[2] for c in constraints])

    def error_function(U, V, W):
        vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
        return np.sum((vals - 1.0) ** 2)

    def normalize_columns(X):
        norms = np.linalg.norm(X, axis=0)
        norms[norms < 1e-12] = 1.0
        return X / norms

    def round_and_verify(U, V, W, tol=1e-9):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if error_function(U_r, V_r, W_r) < tol:
            return U_r, V_r, W_r
        return None

    # --- Smooth objective + analytic gradient over the 24 scalar constraints ---
    def objective_and_grad(params, r):
        U = params[:r * n * m].reshape(n * m, r)
        V = params[r * n * m:r * (n * m + m * p)].reshape(m * p, r)
        W = params[r * (n * m + m * p):].reshape(p * n, r)
        # vals[c] = sum_k U[u_c,k] V[v_c,k] W[w_c,k]
        vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
        diff = vals - 1.0  # shape (num_constraints,)
        obj = np.sum(diff ** 2)
        # Gradients:
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        # dU[u_c,k] += 2*diff[c]*V[v_c,k]*W[w_c,k]
        # Vectorized via index arrays:
        # For each constraint c and each rank k:
        #   dU[u_c, k] += 2*diff[c]*V[v_c,k]*W[w_c,k]
        # We'll do this with einsum-like accumulation using advanced indexing.
        # Build per-constraint contributions.
        # U contribution:
        dU_flat = np.zeros_like(params[:r * n * m])
        # For each c in range(num_constraints):
        #   for k in range(r):
        #     dU[u_c * r + k] += 2*diff[c]*V[v_c,k]*W[w_c,k]
        # Vectorize:
        Vv = V[v_idxs, :]  # (num_constraints, r)
        Ww = W[w_idxs, :]  # (num_constraints, r)
        # contrib_U[c, k] = 2*diff[c]*Vv[c,k]*Ww[c,k]
        contrib_U = 2.0 * diff[:, None] * Vv * Ww  # (num_constraints, r)
        # Scatter into dU_flat: dU_flat[u_c * r + k] += contrib_U[c, k]
        # Use np.add.at
        idx_U = (u_idxs[:, None] * r + np.arange(r)[None, :]).ravel()
        np.add.at(dU_flat, idx_U, contrib_U.ravel())
        # V contribution
        Uu = U[u_idxs, :]  # (num_constraints, r)
        contrib_V = 2.0 * diff[:, None] * Uu * Ww
        idx_V = (v_idxs[:, None] * r + np.arange(r)[None, :]).ravel()
        dV_flat = np.zeros_like(params[r * n * m:r * (n * m + m * p)])
        np.add.at(dV_flat, idx_V, contrib_V.ravel())
        # W contribution
        Uu2 = U[u_idxs, :]
        Vv2 = V[v_idxs, :]
        contrib_W = 2.0 * diff[:, None] * Uu2 * Vv2
        idx_W = (w_idxs[:, None] * r + np.arange(r)[None, :]).ravel()
        dW_flat = np.zeros_like(params[r * (n * m + m * p):])
        np.add.at(dW_flat, idx_W, contrib_W.ravel())
        grad = np.concatenate([dU_flat, dV_flat, dW_flat])
        return obj, grad

    def try_rank_smooth(r, num_restarts=25, max_iters=150):
        if time.time() > deadline - 10.0:
            return None
        size = r * (n * m + m * p + p * n)
        best_err = float('inf')
        best_params = None

        for restart in range(num_restarts):
            if time.time() > deadline - 10.0:
                return None
            np.random.seed(12345 + restart)

            # Structured initialization: start from a "near-schoolbook" pattern
            # For rank r < 24, we can't use all 24 columns, so we build a
            # structured init: assign each of the r columns to a random (i,j,k)
            # triple, set the corresponding U,V,W entries to 1, then add small
            # random perturbation to break symmetry and allow the optimizer to
            # find a lower-rank solution.
            U0 = np.zeros((n * m, r))
            V0 = np.zeros((m * p, r))
            W0 = np.zeros((p * n, r))
            # Assign each column to a random constraint
            assigned = np.random.randint(0, num_constraints, size=r)
            for c in range(r):
                cc = assigned[c]
                u_c, v_c, w_c = constraints[cc]
                U0[u_c, c] = 1.0
                V0[v_c, c] = 1.0
                W0[w_c, c] = 1.0
            # Add small perturbation
            scale = 0.1 + 0.05 * (restart % 4)
            U0 += np.random.randn(n * m, r) * scale
            V0 += np.random.randn(m * p, r) * scale
            W0 += np.random.randn(p * n, r) * scale
            params = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])

            def objective(params):
                return objective_and_grad(params, r)[0]

            def gradient(params):
                return objective_and_grad(params, r)[1]

            try:
                res = minimize(objective, params, jac=gradient, method='L-BFGS-B',
                               options={'maxiter': max_iters, 'ftol': 1e-12,
                                        'gtol': 1e-10, 'maxfun': 300})
            except Exception:
                continue

            if time.time() > deadline - 10.0:
                return None

            err = res.fun
            if err < best_err:
                best_err = err
                best_params = res.x.copy()
                if err < 1e-8:
                    U = res.x[:r * n * m].reshape(n * m, r)
                    V = res.x[r * n * m:r * (n * m + m * p)].reshape(m * p, r)
                    W = res.x[r * (n * m + m * p):].reshape(p * n, r)
                    result = round_and_verify(U, V, W)
                    if result:
                        return result
        return None

    # --- High-precision Newton polish: solve the linearized system around the
    #     current point to drive error to machine precision ---
    def newton_polish(U, V, W, r, max_iters=20):
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            # Build Jacobian of the 24 constraint equations w.r.t. all params
            size = r * (n * m + m * p + p * n)
            J = np.zeros((num_constraints, size))
            vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
            diff = vals - 1.0
            # J[c, u_c*r+k] = V[v_c,k]*W[w_c,k]
            # J[c, v_c*r+k] = U[u_c,k]*W[w_c,k]
            # J[c, w_c*r+k] = U[u_c,k]*V[v_c,k]
            for c in range(num_constraints):
                u_c, v_c, w_c = constraints[c]
                for k in range(r):
                    J[c, u_c * r + k] = V[v_c, k] * W[w_c, k]
                    J[c, r * n * m + v_c * r + k] = U[u_c, k] * W[w_c, k]
                    J[c, r * (n * m + m * p) + w_c * r + k] = U[u_c, k] * V[v_c, k]
            # Solve J * delta = -diff
            try:
                delta, _, _, _ = np.linalg.lstsq(J, -diff, rcond=None)
            except np.linalg.LinAlgError:
                break
            # Damped update
            step = 1.0
            for _ in range(20):
                U_new = (U.ravel() + step * delta[:r * n * m]).reshape(n * m, r)
                V_new = (V.ravel() + step * delta[r * n * m:r * (n * m + m * p)]).reshape(m * p, r)
                W_new = (W.ravel() + step * delta[r * (n * m + m * p):]).reshape(p * n, r)
                new_err = error_function(U_new, V_new, W_new)
                if new_err < error_function(U, V, W):
                    U, V, W = U_new, V_new, W_new
                    break
                step *= 0.5
            if error_function(U, V, W) < 1e-14:
                break
        return U, V, W

    # Try ranks from 19 up to 23
    for r in range(19, 24):
        if time.time() > deadline - 20.0:
            break
        sol = try_rank_smooth(r, num_restarts=30, max_iters=120)
        if sol:
            best_U, best_V, best_W = sol
            best_rank = r
            break

    # Final safety: verify the returned solution
    if error_function(best_U, best_V, best_W) > 1e-8:
        best_U, best_V, best_W = U_seed.copy(), V_seed.copy(), W_seed.copy()
        best_rank = rank_seed

    # Ensure entries are integer or half-integer
    best_U = np.round(best_U * 2) / 2
    best_V = np.round(best_V * 2) / 2
    best_W = np.round(best_W * 2) / 2

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
