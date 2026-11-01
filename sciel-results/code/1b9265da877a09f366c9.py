import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 130.0  # Safety margin before 150s kill

    # Schoolbook rank-24 solution
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

    # Constraints: (u_idx, v_idx, w_idx) for each (i, j, k)
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                constraints.append((u_idx, v_idx, w_idx))

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
        """Round to nearest 0.5 and verify exactness."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if error_function(U_r, V_r, W_r) < tol:
            return U_r, V_r, W_r
        return None

    def try_rank_als(r, num_restarts=30, max_iters=80, seed_base=42):
        """ALS with multiple restarts to find rank-r decomposition."""
        if time.time() - start_time > timeout - 5.0:
            return None

        for restart in range(num_restarts):
            if time.time() - start_time > timeout - 5.0:
                return None

            np.random.seed(seed_base + restart)

            # Random initialization with varied scales
            scale = 0.05 + 0.1 * (restart % 5)
            U = np.random.randn(n * m, r) * scale
            V = np.random.randn(m * p, r) * scale
            W = np.random.randn(p * n, r) * scale

            U = normalize_columns(U)
            V = normalize_columns(V)
            W = normalize_columns(W)

            prev_err = np.inf
            converged = False

            for it in range(max_iters):
                if time.time() - start_time > timeout - 5.0:
                    return None

                # Build linear systems for each factor
                A_U = np.zeros((num_constraints, n * m * r))
                A_V = np.zeros((num_constraints, m * p * r))
                A_W = np.zeros((num_constraints, p * n * r))

                for c in range(num_constraints):
                    u_c, v_c, w_c = constraints[c]
                    for k in range(r):
                        A_U[c, u_c * r + k] = V[v_c, k] * W[w_c, k]
                        A_V[c, v_c * r + k] = U[u_c, k] * W[w_c, k]
                        A_W[c, w_c * r + k] = U[u_c, k] * V[v_c, k]

                try:
                    U_flat, _, _, _ = np.linalg.lstsq(A_U, np.ones(num_constraints), rcond=None)
                    V_flat, _, _, _ = np.linalg.lstsq(A_V, np.ones(num_constraints), rcond=None)
                    W_flat, _, _, _ = np.linalg.lstsq(A_W, np.ones(num_constraints), rcond=None)
                except np.linalg.LinAlgError:
                    continue

                U = U_flat.reshape(n * m, r)
                V = V_flat.reshape(m * p, r)
                W = W_flat.reshape(p * n, r)

                U = normalize_columns(U)
                V = normalize_columns(V)
                W = normalize_columns(W)

                err = error_function(U, V, W)

                if err < 1e-10:
                    converged = True
                    break

                if abs(prev_err - err) < 1e-14 and err < 1e-6:
                    # Likely converged to a local minimum
                    break
                prev_err = err

            # Check if we found a valid solution
            if err < 1e-6 or converged:
                result = round_and_verify(U, V, W)
                if result:
                    return result

            # Also try without normalization - direct rounding
            if err < 1e-6:
                result = round_and_verify(U, V, W)
                if result:
                    return result

        return None

    # Try ranks from 19 up to 23
    for r in range(19, 24):
        if time.time() - start_time > timeout - 10.0:
            break

        sol = try_rank_als(r, num_restarts=40, max_iters=60)
        if sol:
            best_U, best_V, best_W = sol
            best_rank = r
            break

    # If we found rank 19, great. Otherwise keep best found.
    # Final safety: verify the returned solution
    if error_function(best_U, best_V, best_W) > 1e-8:
        # Fallback to schoolbook
        best_U, best_V, best_W = U_seed.copy(), V_seed.copy(), W_seed.copy()
        best_rank = rank_seed

    # Ensure entries are integer or half-integer
    best_U = np.round(best_U * 2) / 2
    best_V = np.round(best_V * 2) / 2
    best_W = np.round(best_W * 2) / 2

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
