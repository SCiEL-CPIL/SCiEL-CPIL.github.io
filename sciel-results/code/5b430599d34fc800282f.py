import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s

    # The rank-17 solution
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))

    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1

    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1

    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1

    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1

    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1

    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1

    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1

    U17[2, 7] = 1; V17[6, 7] = 1; W17[0, 7] = 1
    U17[2, 8] = 1; V17[7, 8] = 1; W17[2, 8] = 1
    U17[5, 9] = 1; V17[6, 9] = 1; W17[1, 9] = 1
    U17[5, 10] = 1; V17[7, 10] = 1; W17[3, 10] = 1

    U17[0, 11] = 1; V17[2, 11] = 1; W17[4, 11] = 1
    U17[1, 12] = 1; V17[5, 12] = 1; W17[4, 12] = 1
    U17[2, 13] = 1; V17[8, 13] = 1; W17[4, 13] = 1
    U17[3, 14] = 1; V17[2, 14] = 1; W17[5, 14] = 1
    U17[4, 15] = 1; V17[5, 15] = 1; W17[5, 15] = 1
    U17[5, 16] = 1; V17[8, 16] = 1; W17[5, 16] = 1

    # Precompute indices for speed
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))

    def fast_residual(U, V, W):
        """Compute the residual for the tensor decomposition."""
        res = 0.0
        for u_idx, v_idx, w_idx in indices:
            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
            res += (val - 1.0) ** 2
        return res

    def fast_residual_grad(U, V, W):
        """Compute the gradient of the residual."""
        U_grad = np.zeros_like(U)
        V_grad = np.zeros_like(V)
        W_grad = np.zeros_like(W)
        for u_idx, v_idx, w_idx in indices:
            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
            diff = val - 1.0
            VW = V[v_idx, :] * W[w_idx, :]
            U_grad[u_idx, :] += 2 * diff * VW
            V_grad[v_idx, :] += 2 * diff * U[u_idx, :] * W[w_idx, :]
            W_grad[w_idx, :] += 2 * diff * U[u_idx, :] * V[v_idx, :]
        return U_grad, V_grad, W_grad

    best_solution = (U17, V17, W17)
    best_res = fast_residual(U17, V17, W17)

    # Try to find rank-16 solution using L-BFGS
    R = 16

    def make_objective(R_val):
        def objective(x):
            U = x[:n*m*R_val].reshape(n*m, R_val)
            V = x[n*m*R_val:n*m*R_val+m*p*R_val].reshape(m*p, R_val)
            W = x[n*m*R_val+m*p*R_val:].reshape(p*n, R_val)
            return fast_residual(U, V, W)
        return objective

    def make_grad(R_val):
        def grad(x):
            U = x[:n*m*R_val].reshape(n*m, R_val)
            V = x[n*m*R_val:n*m*R_val+m*p*R_val].reshape(m*p, R_val)
            W = x[n*m*R_val+m*p*R_val:].reshape(p*n, R_val)
            Ug, Vg, Wg = fast_residual_grad(U, V, W)
            return np.concatenate([Ug.flatten(), Vg.flatten(), Wg.flatten()])
        return grad

    obj = make_objective(R)
    grad = make_grad(R)

    # Strategy 1: Try removing each column from rank-17 and optimize with L-BFGS
    for remove_col in range(17):
        if time.time() > deadline:
            break

        keep_cols = [r for r in range(17) if r != remove_col]
        U0 = U17[:, keep_cols].copy()
        V0 = V17[:, keep_cols].copy()
        W0 = W17[:, keep_cols].copy()

        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])

        try:
            result = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                             options={'maxiter': 150, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50})
            if result.fun < best_res:
                U_opt = result.x[:n*m*R].reshape(n*m, R)
                V_opt = result.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                W_opt = result.x[n*m*R+m*p*R:].reshape(p*n, R)
                best_res = result.fun
                best_solution = (U_opt.copy(), V_opt.copy(), W_opt.copy())
                if best_res < 1e-10:
                    break
        except:
            pass

    # Strategy 2: Random restarts with L-BFGS
    np.random.seed(42)
    if best_res >= 1e-10:
        for attempt in range(80):
            if time.time() > deadline:
                break

            U = np.random.randn(n * m, R) * 0.3
            V = np.random.randn(m * p, R) * 0.3
            W = np.random.randn(p * n, R) * 0.3

            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            try:
                result = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                                 options={'maxiter': 80, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50})
                if result.fun < best_res:
                    U_opt = result.x[:n*m*R].reshape(n*m, R)
                    V_opt = result.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                    W_opt = result.x[n*m*R+m*p*R:].reshape(p*n, R)
                    best_res = result.fun
                    best_solution = (U_opt.copy(), V_opt.copy(), W_opt.copy())
                    if best_res < 1e-10:
                        break
            except:
                pass

    # If we found a good rank-16 solution, try to round to half-integers
    if best_res < 1e-8:
        U_best, V_best, W_best = best_solution

        # Final polish: run L-BFGS with very tight tolerances
        x_polish = np.concatenate([U_best.flatten(), V_best.flatten(), W_best.flatten()])
        try:
            result_polish = minimize(obj, x_polish, jac=grad, method='L-BFGS-B',
                                    options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100})
            if result_polish.fun < best_res:
                U_best = result_polish.x[:n*m*R].reshape(n*m, R)
                V_best = result_polish.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                W_best = result_polish.x[n*m*R+m*p*R:].reshape(p*n, R)
                best_res = result_polish.fun
                best_solution = (U_best.copy(), V_best.copy(), W_best.copy())
        except:
            pass

        # Try multiple rounding strategies
        rounding_methods = [
            ('half', lambda x: np.round(x * 2) / 2),
            ('int', lambda x: np.round(x)),
            ('quarter', lambda x: np.round(x * 4) / 4),
        ]

        for name, round_fn in rounding_methods:
            U_round = round_fn(U_best)
            V_round = round_fn(V_best)
            W_round = round_fn(W_best)

            res_round = fast_residual(U_round, V_round, W_round)
            if res_round < 1e-10:
                return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}

        # Try mixed rounding: round each matrix independently
        for u_round in [lambda x: np.round(x * 2) / 2, lambda x: np.round(x)]:
            for v_round in [lambda x: np.round(x * 2) / 2, lambda x: np.round(x)]:
                for w_round in [lambda x: np.round(x * 2) / 2, lambda x: np.round(x)]:
                    U_r = u_round(U_best)
                    V_r = v_round(V_best)
                    W_r = w_round(W_best)
                    res_r = fast_residual(U_r, V_r, W_r)
                    if res_r < 1e-10:
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # Fall back to rank-17
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
