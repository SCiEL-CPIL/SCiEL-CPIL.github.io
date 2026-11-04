import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 6]

def _verify_half_integer(arr, tol=1e-9):
    doubled = arr * 2
    rounded = np.round(doubled)
    if np.max(np.abs(doubled - rounded)) > tol:
        return False
    return True

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half(arr):
    return np.round(arr * 2) / 2.0

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _optimize_rank(r, n, m, p, time_limit=10, warm_start=None):
    start_time = time.time()
    total_vars = n*m*r + m*p*r + p*n*r

    def objective(x):
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = x[n*m*r + m*p*r :].reshape(p*n, r)

        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        return err

    def gradient(x):
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = x[n*m*r + m*p*r :].reshape(p*n, r)

        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    diff = 2 * (val - 1.0)

                    grad_U[u_idx, :] += diff * (V[v_idx, :] * W[w_idx, :])
                    grad_V[v_idx, :] += diff * (U[u_idx, :] * W[w_idx, :])
                    grad_W[w_idx, :] += diff * (U[u_idx, :] * V[v_idx, :])

        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    def _try_polish(x, max_iter=5000, ftol=1e-18, gtol=1e-14):
        try:
            res = minimize(objective, x, jac=gradient, method='L-BFGS-B',
                           options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol})
            return res.x, res.fun
        except Exception:
            return x, objective(x)

    def _generate_inits(seed_base):
        inits = []

        for scale in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]:
            np.random.seed(seed_base + int(scale * 10000))
            inits.append(np.random.randn(total_vars) * scale)

        for density in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
            np.random.seed(seed_base + 1000 + int(density * 100))
            x0 = np.zeros(total_vars)
            num_nonzero = max(1, int(total_vars * density))
            indices = np.random.choice(total_vars, num_nonzero, replace=False)
            x0[indices] = np.random.choice([-1, 1], num_nonzero) * np.random.uniform(0.3, 2.0, num_nonzero)
            inits.append(x0)

        for seed in [seed_base + 2000, seed_base + 2100]:
            np.random.seed(seed)
            x0 = np.zeros(total_vars)
            block_size = max(1, r // 3)
            for start in range(0, total_vars, block_size):
                end = min(start + block_size, total_vars)
                if np.random.random() < 0.6:
                    x0[start:end] = np.random.choice([-1, 1], end - start) * np.random.uniform(0.2, 1.5)
            inits.append(x0)

        return inits

    best_loss = float('inf')
    best_x = None

    all_inits = []
    if warm_start is not None:
        all_inits.append(warm_start)

    all_inits.extend(_generate_inits(42))
    all_inits.extend(_generate_inits(123))

    for idx, x0 in enumerate(all_inits):
        if time.time() - start_time > time_limit - 3:
            break

        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})

            if res.fun < best_loss:
                best_loss = res.fun
                best_x = res.x.copy()

                U = best_x[:n*m*r].reshape(n*m, r)
                V = best_x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W = best_x[n*m*r + m*p*r :].reshape(p*n, r)

                U_snap = _snap_to_half(U)
                V_snap = _snap_to_half(V)
                W_snap = _snap_to_half(W)

                if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-5):
                    if time.time() - start_time < time_limit - 2:
                        polished_x, polished_loss = _try_polish(best_x)
                        if polished_loss < best_loss:
                            best_x = polished_x
                            best_loss = polished_loss
                        U = best_x[:n*m*r].reshape(n*m, r)
                        V = best_x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                        W = best_x[n*m*r + m*p*r :].reshape(p*n, r)
                        U_snap = _snap_to_half(U)
                        V_snap = _snap_to_half(V)
                        W_snap = _snap_to_half(W)

                    if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
                        return U_snap, V_snap, W_snap, True, None

        except Exception:
            pass

    if best_x is not None and best_loss < 1e-3:
        if time.time() - start_time < time_limit - 2:
            polished_x, polished_loss = _try_polish(best_x)
            if polished_loss < best_loss:
                best_x = polished_x
                best_loss = polished_loss

        U = best_x[:n*m*r].reshape(n*m, r)
        V = best_x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = best_x[n*m*r + m*p*r :].reshape(p*n, r)

        U_snap = _snap_to_half(U)
        V_snap = _snap_to_half(V)
        W_snap = _snap_to_half(W)

        if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
            return U_snap, V_snap, W_snap, True, None

    return None, None, None, False, best_x

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115

    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    school_rank = n * m * p

    # Strategy 1: Reduce from schoolbook by dropping one term at a time
    # Try rank school_rank-1, school_rank-2, ... with warm starts
    for r in range(school_rank - 1, max(school_rank - 12, 23), -1):
        if time.time() > deadline - 35:
            break

        # Create warm start: use first r columns of schoolbook
        warm = np.zeros(n*m*r + m*p*r + p*n*r)
        warm[:n*m*r] = U_school[:, :r].flatten()
        warm[n*m*r : n*m*r + m*p*r] = V_school[:, :r].flatten()
        warm[n*m*r + m*p*r :] = W_school[:, :r].flatten()

        t_use = min(8, max(3, deadline - time.time() - 30))

        result = _optimize_rank(r, n, m, p, time_limit=t_use, warm_start=warm)

        if len(result) == 5:
            U, V, W, success, warm_out = result
        else:
            U, V, W, success = result
            warm_out = None

        if success:
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}

    # Strategy 2: Try mid-range ranks with random initializations
    for r in [40, 36, 32, 30, 28, 34, 38, 44, 48, 52, 56, 60, 64, 68, 70]:
        if time.time() > deadline - 15:
            break

        t_use = min(10, max(5, deadline - time.time() - 12))

        result = _optimize_rank(r, n, m, p, time_limit=t_use)

        if len(result) == 5:
            U, V, W, success, warm_out = result
        else:
            U, V, W, success = result
            warm_out = None

        if success:
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: schoolbook
    U_list = [[float(x) for x in row] for row in U_school]
    V_list = [[float(x) for x in row] for row in V_school]
    W_list = [[float(x) for x in row] for row in W_school]
    return {"U": U_list, "V": V_list, "W": W_list}
