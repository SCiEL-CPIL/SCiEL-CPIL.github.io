# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations


def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    deadline = time.time() + 575
    pairs = list(combinations(range(n), 2))
    n_pairs = len(pairs)

    def check_feasible(cx, cy, s, tol=1e-7):
        for i in range(n):
            if cx[i] - s[i]/2 < -tol or cx[i] + s[i]/2 > 1 + tol:
                return False
            if cy[i] - s[i]/2 < -tol or cy[i] + s[i]/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                if t <= 0:
                    continue
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                if dx < t and dy < t:
                    return False
        return True

    def _valid(centers, sides, tol=1e-7):
        if centers.shape != (n, 2) or sides.shape != (n,):
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        cx = centers[:, 0]
        cy = centers[:, 1]
        s = sides
        for i in range(n):
            if cx[i] - s[i]/2 < -tol or cx[i] + s[i]/2 > 1 + tol:
                return False
            if cy[i] - s[i]/2 < -tol or cy[i] + s[i]/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                if t <= 0:
                    continue
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                if dx < t and dy < t:
                    return False
        return True

    # --- Heuristic local improvement ---
    def local_improve(cx, cy, s, max_iter=300):
        improved = True
        it = 0
        step_pos = 0.005
        step_size = 0.003
        while improved and it < max_iter and time.time() < deadline - 60:
            improved = False
            it += 1
            # Try moving each square and growing
            for i in range(n):
                for dx in [-step_pos, 0, step_pos]:
                    for dy in [-step_pos, 0, step_pos]:
                        if dx == 0 and dy == 0:
                            continue
                        cx_new = cx.copy()
                        cy_new = cy.copy()
                        cx_new[i] += dx
                        cy_new[i] += dy
                        if check_feasible(cx_new, cy_new, s):
                            for j in range(n):
                                s_try = s.copy()
                                s_try[j] += step_size
                                if check_feasible(cx_new, cy_new, s_try):
                                    cx, cy, s = cx_new, cy_new, s_try
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break
                if improved:
                    break
            # Try growing each square in place
            for i in range(n):
                s_new = s.copy()
                s_new[i] += step_size
                if check_feasible(cx, cy, s_new):
                    s = s_new
                    improved = True
        return cx, cy, s

    # --- SLSQP solver for a given separation pattern ---
    def solve_pattern(pattern, x0, maxiter=300):
        """
        pattern: array of length n_pairs, 0=x-sep (i left of j), 1=y-sep (i below j)
        x: [s_0..s_9, cx_0..cx_9, cy_0..cy_9]
        """
        if time.time() > deadline - 30:
            return None, None

        def objective(x):
            return -np.sum(x[:n])

        def objective_grad(x):
            g = np.zeros(3 * n)
            g[:n] = -1.0
            return g

        constraints = []

        # Boundary constraints
        for i in range(n):
            # cx_i - s_i/2 >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[n+i] - x[i]/2,
                'jac': lambda x, i=i: _jac_boundary_x_low(i, n)
            })
            # 1 - cx_i - s_i/2 >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: 1 - x[n+i] - x[i]/2,
                'jac': lambda x, i=i: _jac_boundary_x_high(i, n)
            })
            # cy_i - s_i/2 >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[2*n+i] - x[i]/2,
                'jac': lambda x, i=i: _jac_boundary_y_low(i, n)
            })
            # 1 - cy_i - s_i/2 >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: 1 - x[2*n+i] - x[i]/2,
                'jac': lambda x, i=i: _jac_boundary_y_high(i, n)
            })

        # Separation constraints
        for k, (i, j) in enumerate(pairs):
            if pattern[k] == 0:
                # i left of j: cx_j - cx_i >= (s_i + s_j)/2
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: x[n+j] - x[n+i] - (x[i] + x[j])/2,
                    'jac': lambda x, i=i, j=j: _jac_sep_x(i, j, n)
                })
            else:
                # i below j: cy_j - cy_i >= (s_i + s_j)/2
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: x[2*n+j] - x[2*n+i] - (x[i] + x[j])/2,
                    'jac': lambda x, i=i, j=j: _jac_sep_y(i, j, n)
                })

        bounds = [(0.001, 1)] * n + [(0, 1)] * n + [(0, 1)] * n

        try:
            res = minimize(
                objective, x0, jac=objective_grad,
                method='SLSQP', bounds=bounds, constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-14}
            )
            if res.success:
                s_sol = res.x[:n]
                cx_sol = res.x[n:2*n]
                cy_sol = res.x[2*n:3*n]
                centers_sol = np.column_stack([cx_sol, cy_sol])
                if _valid(centers_sol, s_sol):
                    return -res.fun, res.x
        except Exception:
            pass
        return None, None

    def _jac_boundary_x_low(i, n):
        g = np.zeros(3 * n)
        g[i] = -0.5
        g[n + i] = 1.0
        return g

    def _jac_boundary_x_high(i, n):
        g = np.zeros(3 * n)
        g[i] = -0.5
        g[n + i] = -1.0
        return g

    def _jac_boundary_y_low(i, n):
        g = np.zeros(3 * n)
        g[i] = -0.5
        g[2 * n + i] = 1.0
        return g

    def _jac_boundary_y_high(i, n):
        g = np.zeros(3 * n)
        g[i] = -0.5
        g[2 * n + i] = -1.0
        return g

    def _jac_sep_x(i, j, n):
        g = np.zeros(3 * n)
        g[i] = -0.5
        g[j] = -0.5
        g[n + i] = -1.0
        g[n + j] = 1.0
        return g

    def _jac_sep_y(i, j, n):
        g = np.zeros(3 * n)
        g[i] = -0.5
        g[j] = -0.5
        g[2 * n + i] = -1.0
        g[2 * n + j] = 1.0
        return g

    # --- Smooth penalty-based polish ---
    def smooth_polish(cx, cy, s, max_iter=500):
        """
        Use a smooth penalty on constraint violations to potentially escape
        local optima. Variables: [s, cx, cy].
        """
        if time.time() > deadline - 20:
            return None, None, None

        x0 = np.concatenate([s, cx, cy])

        def penalty_objective(x, mu=100.0):
            sv = x[:n]
            cxv = x[n:2*n]
            cyv = x[2*n:3*n]
            obj = -np.sum(sv)
            pen = 0.0
            # Boundary penalties (soft)
            for i in range(n):
                pen += max(0, -(cxv[i] - sv[i]/2)) ** 2
                pen += max(0, cxv[i] + sv[i]/2 - 1) ** 2
                pen += max(0, -(cyv[i] - sv[i]/2)) ** 2
                pen += max(0, cyv[i] + sv[i]/2 - 1) ** 2
            # Overlap penalties (soft)
            for i in range(n):
                for j in range(i + 1, n):
                    t = (sv[i] + sv[j]) / 2
                    dx = abs(cxv[i] - cxv[j])
                    dy = abs(cyv[i] - cyv[j])
                    # violation if dx < t AND dy < t
                    viol_x = max(0, t - dx)
                    viol_y = max(0, t - dy)
                    # both must be violated for overlap
                    pen += min(viol_x, viol_y) ** 2
            return obj + mu * pen

        def penalty_grad(x, mu=100.0):
            sv = x[:n]
            cxv = x[n:2*n]
            cyv = x[2*n:3*n]
            g = np.zeros(3 * n)
            g[:n] = -1.0

            # Boundary gradients
            for i in range(n):
                # cx_i - s_i/2 >= 0
                if cxv[i] - sv[i]/2 < 0:
                    v = -(cxv[i] - sv[i]/2)
                    g[i] += mu * 2 * v * 0.5
                    g[n+i] += mu * 2 * v * (-1.0)
                # cx_i + s_i/2 <= 1
                if cxv[i] + sv[i]/2 - 1 > 0:
                    v = cxv[i] + sv[i]/2 - 1
                    g[i] += mu * 2 * v * 0.5
                    g[n+i] += mu * 2 * v * 1.0
                # cy_i - s_i/2 >= 0
                if cyv[i] - sv[i]/2 < 0:
                    v = -(cyv[i] - sv[i]/2)
                    g[i] += mu * 2 * v * 0.5
                    g[2*n+i] += mu * 2 * v * (-1.0)
                # cy_i + s_i/2 <= 1
                if cyv[i] + sv[i]/2 - 1 > 0:
                    v = cyv[i] + sv[i]/2 - 1
                    g[i] += mu * 2 * v * 0.5
                    g[2*n+i] += mu * 2 * v * 1.0

            # Overlap gradients
            for i in range(n):
                for j in range(i + 1, n):
                    t = (sv[i] + sv[j]) / 2
                    dx = cxv[i] - cxv[j]
                    dy = cyv[i] - cyv[j]
                    viol_x = max(0, t - abs(dx))
                    viol_y = max(0, t - abs(dy))
                    m = min(viol_x, viol_y)
                    if m > 0:
                        pen_val = mu * 2 * m
                        if viol_x <= viol_y:
                            # gradient w.r.t. viol_x
                            if dx > 0:
                                g[n+i] -= pen_val
                                g[n+j] += pen_val
                            else:
                                g[n+i] += pen_val
                                g[n+j] -= pen_val
                            g[i] -= pen_val * 0.5
                            g[j] -= pen_val * 0.5
                        else:
                            # gradient w.r.t. viol_y
                            if dy > 0:
                                g[2*n+i] -= pen_val
                                g[2*n+j] += pen_val
                            else:
                                g[2*n+i] += pen_val
                                g[2*n+j] -= pen_val
                            g[i] -= pen_val * 0.5
                            g[j] -= pen_val * 0.5

            return g

        try:
            res = minimize(
                penalty_objective, x0, jac=penalty_grad,
                method='L-BFGS-B',
                bounds=[(0.001, 1)] * n + [(0, 1)] * n + [(0, 1)] * n,
                options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-10}
            )
            s_sol = res.x[:n]
            cx_sol = res.x[n:2*n]
            cy_sol = res.x[2*n:3*n]
            centers_sol = np.column_stack([cx_sol, cy_sol])
            if _valid(centers_sol, s_sol):
                return cx_sol, cy_sol, s_sol
        except Exception:
            pass
        return None, None, None

    # --- Main search ---
    rng = np.random.default_rng(42)

    best_cx = None
    best_cy = None
    best_s = None
    best_obj = 0.0

    # Initial configurations
    configs = []

    # 1. 4x4 grid (first 10 cells)
    k = 4
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx0 = (cols + 0.5) * side
    cy0 = (rows + 0.5) * side
    s0 = np.full(n, side)
    configs.append((cx0.copy(), cy0.copy(), s0.copy()))

    # 2. 2x5 grid
    kx, ky = 2, 5
    cols = idx % kx
    rows = idx // kx
    cx1 = (cols + 0.5) * (1.0 / kx)
    cy1 = (rows + 0.5) * (1.0 / ky)
    s1 = np.full(n, min(1.0/kx, 1.0/ky))
    configs.append((cx1.copy(), cy1.copy(), s1.copy()))

    # 3. 5x2 grid
    kx, ky = 5, 2
    cols = idx % kx
    rows = idx // kx
    cx2 = (cols + 0.5) * (1.0 / kx)
    cy2 = (rows + 0.5) * (1.0 / ky)
    s2 = np.full(n, min(1.0/kx, 1.0/ky))
    configs.append((cx2.copy(), cy2.copy(), s2.copy()))

    # 4. 3x3 + 1
    cx3 = np.zeros(n)
    cy3 = np.zeros(n)
    s3 = np.zeros(n)
    # 3x3 grid in top-left
    for i in range(9):
        cx3[i] = (i % 3 + 0.5) / 3.0
        cy3[i] = (i // 3 + 0.5) / 3.0
        s3[i] = 1.0/3.0
    # 10th square in bottom-right
    cx3[9] = 0.75
    cy3[9] = 0.25
    s3[9] = 0.5
    configs.append((cx3.copy(), cy3.copy(), s3.copy()))

    # 5-10: Random configurations
    for r in range(6):
        cxr = rng.uniform(0.15, 0.85, n)
        cyr = rng.uniform(0.15, 0.85, n)
        sr = rng.uniform(0.15, 0.35, n)
        # Clip to feasible
        for i in range(n):
            sr[i] = min(sr[i], 0.4)
            cxr[i] = np.clip(cxr[i], sr[i]/2 + tol, 1 - sr[i]/2 - tol)
            cyr[i] = np.clip(cyr[i], sr[i]/2 + tol, 1 - sr[i]/2 - tol)
        configs.append((cxr.copy(), cyr.copy(), sr.copy()))

    # Process each configuration
    for cfg_idx, (cx, cy, s) in enumerate(configs):
        if time.time() > deadline - 60:
            break

        # Ensure feasibility
        if not check_feasible(cx, cy, s):
            # Shrink until feasible
            for _ in range(100):
                if check_feasible(cx, cy, s):
                    break
                s *= 0.9
            if not check_feasible(cx, cy, s):
                continue

        # Local improvement
        cx, cy, s = local_improve(cx, cy, s)
        obj = np.sum(s)
        if obj > best_obj:
            best_obj = obj
            best_cx = cx.copy()
            best_cy = cy.copy()
            best_s = s.copy()

        # Determine separation pattern and solve with SLSQP
        if time.time() > deadline - 40:
            break

        pattern = np.zeros(n_pairs, dtype=int)
        for k_idx, (i, j) in enumerate(pairs):
            dx = abs(cx[i] - cx[j])
            dy = abs(cy[i] - cy[j])
            if dx >= dy:
                pattern[k_idx] = 0  # x-sep
            else:
                pattern[k_idx] = 1  # y-sep

        x0 = np.concatenate([s, cx, cy])
        val, x_sol = solve_pattern(pattern, x0, maxiter=300)
        if val is not None and val > best_obj:
            best_obj = val
            best_s = x_sol[:n].copy()
            best_cx = x_sol[n:2*n].copy()
            best_cy = x_sol[2*n:3*n].copy()

    # Smooth penalty polish on best solution
    if best_cx is not None and time.time() < deadline - 20:
        cx_pol, cy_pol, s_pol = smooth_polish(best_cx, best_cy, best_s, max_iter=300)
        if cx_pol is not None:
            obj_pol = np.sum(s_pol)
            if obj_pol > best_obj:
                best_obj = obj_pol
                best_cx = cx_pol.copy()
                best_cy = cy_pol.copy()
                best_s = s_pol.copy()

    # Final SLSQP polish on the best solution with its pattern
    if best_cx is not None and time.time() < deadline - 10:
        pattern = np.zeros(n_pairs, dtype=int)
        for k_idx, (i, j) in enumerate(pairs):
            dx = abs(best_cx[i] - best_cx[j])
            dy = abs(best_cy[i] - best_cy[j])
            if dx >= dy:
                pattern[k_idx] = 0
            else:
                pattern[k_idx] = 1
        x0 = np.concatenate([best_s, best_cx, best_cy])
        val, x_sol = solve_pattern(pattern, x0, maxiter=500)
        if val is not None and val > best_obj:
            best_obj = val
            best_s = x_sol[:n].copy()
            best_cx = x_sol[n:2*n].copy()
            best_cy = x_sol[2*n:3*n].copy()

    # Fallback if nothing found
    if best_cx is None:
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        best_cx = (cols + 0.5) * side
        best_cy = (rows + 0.5) * side
        best_s = np.full(n, side)

    # Final cleanup
    best_s = np.maximum(best_s, 1e-8)
    for i in range(n):
        best_cx[i] = np.clip(best_cx[i], best_s[i]/2 - tol, 1.0 - best_s[i]/2 + tol)
        best_cy[i] = np.clip(best_cy[i], best_s[i]/2 - tol, 1.0 - best_s[i]/2 + tol)

    centers = np.stack([best_cx, best_cy], axis=1).astype(np.float64)
    sides = best_s.astype(np.float64)

    return centers, sides


# EVOLVE-BLOCK-END
