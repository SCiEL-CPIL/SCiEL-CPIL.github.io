"""
Rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication.
Keeps the current family (Strassen-36 base + drop-and-optimize) but adds:
  - a cheap full tensor-checker that runs only when the smooth objective is tiny,
  - a tighter, faster polish pass (more iterations, lower ftol/gtol),
  - a few extra cheap random-restart initializations,
  - explicit deadline guard so we always return the best valid result.
"""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 5]


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill

    # ── Build rank-36 Strassen-based decomposition ──────────────────────────
    def build_strassen_36():
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def u_idx(i, j): return i * m + j
        def v_idx(j, k): return j * p + k
        def w_idx(k, i): return k * n + i

        col = 0

        def strassen_block(y, z, w):
            nonlocal col
            y00, y01, y10, y11 = y
            z00, z01, z10, z11 = z
            w0, w1, w2, w3 = w

            U[u_idx(*y00)][col] += 1; U[u_idx(*y11)][col] += 1
            V[v_idx(*z00)][col] += 1; V[v_idx(*z11)][col] += 1
            W[w_idx(*w0)][col] += 1; W[w_idx(*w3)][col] += 1
            col += 1
            U[u_idx(*y10)][col] += 1; U[u_idx(*y11)][col] += 1
            V[v_idx(*z00)][col] += 1
            W[w_idx(*w2)][col] += 1; W[w_idx(*w3)][col] -= 1
            col += 1
            U[u_idx(*y00)][col] += 1
            V[v_idx(*z01)][col] += 1; V[v_idx(*z11)][col] -= 1
            W[w_idx(*w1)][col] += 1; W[w_idx(*w3)][col] += 1
            col += 1
            U[u_idx(*y11)][col] += 1
            V[v_idx(*z10)][col] += 1; V[v_idx(*z00)][col] -= 1
            W[w_idx(*w0)][col] += 1; W[w_idx(*w2)][col] += 1
            col += 1
            U[u_idx(*y00)][col] += 1; U[u_idx(*y01)][col] += 1
            V[v_idx(*z11)][col] += 1
            W[w_idx(*w0)][col] -= 1; W[w_idx(*w1)][col] += 1
            col += 1
            U[u_idx(*y10)][col] += 1; U[u_idx(*y00)][col] -= 1
            V[v_idx(*z00)][col] += 1; V[v_idx(*z01)][col] += 1
            W[w_idx(*w3)][col] += 1
            col += 1
            U[u_idx(*y01)][col] += 1; U[u_idx(*y11)][col] -= 1
            V[v_idx(*z10)][col] += 1; V[v_idx(*z11)][col] += 1
            W[w_idx(*w0)][col] += 1
            col += 1

        strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)],
                       [(0,0),(1,0),(0,1),(1,1)])
        strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)],
                       [(0,0),(1,0),(0,1),(1,1)])
        strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)],
                       [(2,0),(3,0),(2,1),(3,1)])
        strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)],
                       [(2,0),(3,0),(2,1),(3,1)])

        for row in range(2):
            for col_a in range(2):
                U[u_idx(row, col_a)][col] += 1
                V[v_idx(col_a, 4)][col] += 1
                W[w_idx(4, row)][col] += 1
                col += 1
        for row in range(2):
            for col_a in range(2, 4):
                U[u_idx(row, col_a)][col] += 1
                V[v_idx(col_a, 4)][col] += 1
                W[w_idx(4, row)][col] += 1
                col += 1

        assert col == rank
        return {"U": U, "V": V, "W": W}

    best_result = build_strassen_36()
    best_rank = 36

    # ── Precompute target indices ───────────────────────────────────────────
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i*m+j, j*p+k, k*n+i))
    target_set = set(target_indices)

    # ── Fast full tensor check (only called when smooth obj is tiny) ────────
    def full_check(U, V, W, tol=1e-7):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        # target entries must equal 1
        for (a, b, c) in target_indices:
            if abs(np.dot(W[c, :], U[a, :] * V[b, :]) - 1.0) > tol:
                return False
        # all non-target entries must be 0
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if (a, b, c) in target_set:
                        continue
                    if abs(np.dot(W[c, :], U[a, :] * V[b, :])) > tol:
                        return False
        return True

    # ── Objective with analytical gradient (vectorized per entry) ──────────
    target_A = np.array([x[0] for x in target_indices])
    target_B = np.array([x[1] for x in target_indices])
    target_C = np.array([x[2] for x in target_indices])

    def objective_with_grad(params, rank):
        num_u = n * m * rank
        num_v = m * p * rank
        U = params[:num_u].reshape((n * m, rank))
        V = params[num_u:num_u + num_v].reshape((m * p, rank))
        W = params[num_u + num_v:].reshape((p * n, rank))

        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        err = 0.0

        for idx in range(len(target_A)):
            a, b, c = target_A[idx], target_B[idx], target_C[idx]
            uv = U[a, :] * V[b, :]
            val = np.dot(W[c, :], uv)
            residual = val - 1.0
            err += residual * residual
            grad_U[a, :] += 2.0 * residual * V[b, :] * W[c, :]
            grad_V[b, :] += 2.0 * residual * U[a, :] * W[c, :]
            grad_W[c, :] += 2.0 * residual * uv

        # Penalize non-target entries that are "near-misses" (cheap subset)
        # We only check a small set to keep the objective cheap; the full check
        # is done separately when the objective is tiny.
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if (a, b, c) in target_set:
                        continue
                    # Only check near-misses (differ in at most one index)
                    i, j = divmod(a, m)
                    j2, k = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    matches = (j == j2) + (k == k2) + (i == i2)
                    if matches < 2:
                        continue
                    uv = U[a, :] * V[b, :]
                    val = np.dot(W[c, :], uv)
                    err += val * val
                    grad_U[a, :] += 2.0 * val * V[b, :] * W[c, :]
                    grad_V[b, :] += 2.0 * val * U[a, :] * W[c, :]
                    grad_W[c, :] += 2.0 * val * uv

        return err, np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    # ── Try to solve and round ──────────────────────────────────────────────
    U0 = np.array(best_result['U'], dtype=float)
    V0 = np.array(best_result['V'], dtype=float)
    W0 = np.array(best_result['W'], dtype=float)

    def try_solve(U_init, V_init, W_init, rank, max_iter=400, ftol=1e-14, gtol=1e-10):
        params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        try:
            res = minimize(objective_with_grad, params0, args=(rank,), method='L-BFGS-B',
                           jac=True, options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol})
            if res.fun < 1e-5:
                num_u = n * m * rank
                num_v = m * p * rank
                U_sol = res.x[:num_u].reshape((n * m, rank))
                V_sol = res.x[num_u:num_u + num_v].reshape((m * p, rank))
                W_sol = res.x[num_u + num_v:].reshape((p * n, rank))

                # Polish: one more short L-BFGS-B run with tighter tolerances
                try:
                    res2 = minimize(objective_with_grad, res.x, args=(rank,), method='L-BFGS-B',
                                    jac=True, options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-12})
                    if res2.fun < res.fun:
                        U_sol = res2.x[:num_u].reshape((n * m, rank))
                        V_sol = res2.x[num_u:num_u + num_v].reshape((m * p, rank))
                        W_sol = res2.x[num_u + num_v:].reshape((p * n, rank))
                except Exception:
                    pass

                # Try rounding to half-integers
                U_r = np.round(U_sol * 2) / 2
                V_r = np.round(V_sol * 2) / 2
                W_r = np.round(W_sol * 2) / 2
                if full_check(U_r, V_r, W_r, tol=1e-6):
                    return U_r, V_r, W_r

                # Try rounding to integers
                U_r = np.round(U_sol)
                V_r = np.round(V_sol)
                W_r = np.round(W_sol)
                if full_check(U_r, V_r, W_r, tol=1e-6):
                    return U_r, V_r, W_r

                # Try without rounding
                if full_check(U_sol, V_sol, W_sol, tol=1e-7):
                    return U_sol, V_sol, W_sol
        except Exception:
            pass
        return None

    rng = np.random.RandomState(42)

    # Strategy 1: Drop one column from rank-36, add noise, optimize for rank-35
    for attempt in range(25):
        if time.time() > deadline:
            break
        drop_idx = rng.randint(0, 35)
        U_init = np.delete(U0, drop_idx, axis=1).copy()
        V_init = np.delete(V0, drop_idx, axis=1).copy()
        W_init = np.delete(W0, drop_idx, axis=1).copy()

        noise = rng.uniform(0.01, 0.12)
        U_init += rng.normal(0, noise, U_init.shape)
        V_init += rng.normal(0, noise, V_init.shape)
        W_init += rng.normal(0, noise, W_init.shape)

        result = try_solve(U_init, V_init, W_init, 35, max_iter=350, ftol=1e-14, gtol=1e-10)
        if result:
            U_r, V_r, W_r = result
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # Strategy 2: Drop one column, no noise
    for drop_idx in range(36):
        if time.time() > deadline:
            break
        U_init = np.delete(U0, drop_idx, axis=1).copy()
        V_init = np.delete(V0, drop_idx, axis=1).copy()
        W_init = np.delete(W0, drop_idx, axis=1).copy()

        result = try_solve(U_init, V_init, W_init, 35, max_iter=400, ftol=1e-15, gtol=1e-11)
        if result:
            U_r, V_r, W_r = result
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # Strategy 3: Random structured initialization
    for attempt in range(15):
        if time.time() > deadline:
            break
        U_init = U0.copy()
        V_init = V0.copy()
        W_init = W0.copy()
        zero_col = rng.randint(0, 35)
        keep = [i for i in range(36) if i != zero_col]
        U_init = U_init[:, keep].copy()
        V_init = V_init[:, keep].copy()
        W_init = W_init[:, keep].copy()

        U_init[:, -1] += rng.normal(0, 0.25, n*m)
        V_init[:, -1] += rng.normal(0, 0.25, m*p)
        W_init[:, -1] += rng.normal(0, 0.25, p*n)

        result = try_solve(U_init, V_init, W_init, 35, max_iter=300, ftol=1e-13, gtol=1e-9)
        if result:
            U_r, V_r, W_r = result
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # Strategy 4: Try rank-34
    for attempt in range(10):
        if time.time() > deadline:
            break
        drop_idx = rng.randint(0, 35)
        U_init = np.delete(U0, drop_idx, axis=1).copy()
        V_init = np.delete(V0, drop_idx, axis=1).copy()
        W_init = np.delete(W0, drop_idx, axis=1).copy()
        drop_idx2 = rng.randint(0, 34)
        U_init = np.delete(U_init, drop_idx2, axis=1).copy()
        V_init = np.delete(V_init, drop_idx2, axis=1).copy()
        W_init = np.delete(W_init, drop_idx2, axis=1).copy()

        noise = rng.uniform(0.05, 0.15)
        U_init += rng.normal(0, noise, U_init.shape)
        V_init += rng.normal(0, noise, V_init.shape)
        W_init += rng.normal(0, noise, W_init.shape)

        result = try_solve(U_init, V_init, W_init, 34, max_iter=400, ftol=1e-14, gtol=1e-10)
        if result:
            U_r, V_r, W_r = result
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    return best_result
