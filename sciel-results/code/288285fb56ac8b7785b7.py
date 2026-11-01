"""Circle packing in perimeter-4 rectangle: DE global search over aspect ratio + L-BFGS-B penalty refinement."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 21
_TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _extract(x, y, r, w, h):
    """Helper to extract arrays from flat vector."""
    return x, y, r, w, h


def _penalty_objective(z, beta=50.0):
    """
    Objective: minimize -sum(r) + penalty for constraint violations.
    z = [x0, y0, r0, x1, y1, r1, ..., xN-1, yN-1, rN-1, w]
    h = 2 - w
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w <= 0.1 or h <= 0.1:
        return 1e6

    penalty = 0.0

    # Wall constraints
    for i in range(N):
        if r[i] <= 0:
            penalty += 100.0 * (0.01 - r[i]) ** 2
        else:
            # x >= r
            if x[i] < r[i]:
                penalty += 100.0 * (r[i] - x[i]) ** 2
            # x + r <= w
            if x[i] + r[i] > w:
                penalty += 100.0 * (x[i] + r[i] - w) ** 2
            # y >= r
            if y[i] < r[i]:
                penalty += 100.0 * (r[i] - y[i]) ** 2
            # y + r <= h
            if y[i] + r[i] > h:
                penalty += 100.0 * (y[i] + r[i] - h) ** 2

    # Pairwise constraints
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            rs = r[i] + r[j]
            if d2 < rs * rs:
                # Violation: circles overlap
                viol = rs - np.sqrt(max(d2, 1e-12))
                penalty += 100.0 * viol * viol

    return -np.sum(r) + penalty


def _is_valid(x, y, r, w, h, tol=1e-9):
    """Check if the packing is valid."""
    if w <= 0.05 or h <= 0.05:
        return False
    for i in range(len(r)):
        if r[i] <= 0:
            return False
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            rs = r[i] + r[j]
            if d2 < rs * rs - tol:
                return False
    return True


def _cellular_init(w, h, n, rng):
    """Generate initial guess using cellular decomposition."""
    # Try several grid layouts
    best_z = None
    best_sum = -1

    for cols in range(3, 9):
        rows = int(np.ceil(n / cols))
        if rows < 2:
            continue
        if cols * rows < n:
            continue

        cw = w / cols
        ch = h / rows
        base_r = min(cw, ch) / 2.0 * 0.9

        x = np.zeros(n)
        y = np.zeros(n)
        r = np.zeros(n)

        for i in range(n):
            row, col = divmod(i, cols)
            x[i] = (col + 0.5) * cw + rng.uniform(-0.05, 0.05) * cw
            y[i] = (row + 0.5) * ch + rng.uniform(-0.05, 0.05) * ch
            r[i] = base_r

        # Clamp to walls
        for i in range(n):
            r[i] = min(r[i], x[i], w - x[i], y[i], h - y[i])
            if r[i] < 0.005:
                r[i] = 0.005

        # Fix overlaps by shrinking
        for _ in range(50):
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    rs = r[i] + r[j]
                    if d < rs - 1e-8:
                        shrink = (rs - d) / 2.0 + 0.001
                        r[i] = max(0.005, r[i] - shrink)
                        r[j] = max(0.005, r[j] - shrink)
                        changed = True
            if not changed:
                break

        s = np.sum(r)
        if s > best_sum:
            best_sum = s
            z = np.zeros(3 * N + 1)
            z[0::3][:n] = x[:n]
            z[1::3][:n] = y[:n]
            z[2::3][:n] = r[:n]
            z[-1] = w
            best_z = z

    if best_z is None:
        # Fallback
        side = int(np.ceil(np.sqrt(n)))
        z = np.zeros(3 * N + 1)
        for i in range(n):
            row, col = divmod(i, side)
            z[3 * i] = (col + 0.5) * w / side
            z[3 * i + 1] = (row + 0.5) * h / side
            z[3 * i + 2] = min(w, h) / side * 0.45
        z[-1] = w
        best_z = z

    return best_z


def _lbfgsb_refine(z0, maxiter=500):
    """Refine using L-BFGS-B with penalty objective."""
    bounds = []
    for i in range(N):
        bounds.append((0.0, 2.0))  # x
        bounds.append((0.0, 2.0))  # y
        bounds.append((0.0, 1.0))  # r
    bounds.append((0.3, 1.7))  # w (aspect ratio)

    try:
        res = minimize(_penalty_objective, z0, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50})
        return res.x
    except Exception:
        return z0


def _slsqp_refine(z0, w, h, maxiter=300):
    """SLSQP refinement with hard constraints for a fixed aspect ratio."""
    def objective(z):
        r = z[2::3][:N]
        return -np.sum(r)

    def objective_jac(z):
        jac = np.zeros(3 * N + 1)
        jac[2::3][:N] = -1.0
        return jac

    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: z[3 * i] - z[3 * i + 2],
                            'jac': lambda z, i=i: np.concatenate([np.eye(3 * N + 1)[3 * i], np.zeros(1)])})
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: w - z[3 * i] - z[3 * i + 2],
                            'jac': lambda z, i=i: np.concatenate([-np.eye(3 * N + 1)[3 * i], np.zeros(1)])})
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: z[3 * i + 1] - z[3 * i + 2],
                            'jac': lambda z, i=i: np.concatenate([np.eye(3 * N + 1)[3 * i + 1], np.zeros(1)])})
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: h - z[3 * i + 1] - z[3 * i + 2],
                            'jac': lambda z, i=i: np.concatenate([-np.eye(3 * N + 1)[3 * i + 1], np.zeros(1)])})

    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({'type': 'ineq',
                                'fun': lambda z, i=i, j=j:
                                (z[3 * i] - z[3 * j]) ** 2 + (z[3 * i + 1] - z[3 * j + 1]) ** 2 - (z[3 * i + 2] + z[3 * j + 2]) ** 2,
                                'jac': lambda z, i=i, j=j: _pairwise_jac(z, i, j)})

    bounds = [(0, 2.0)] * (3 * N) + [(w, w)]

    try:
        res = minimize(objective, z0, method='SLSQP', jac=objective_jac,
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 60})
        return res.x
    except Exception:
        return z0


def _pairwise_jac(z, i, j):
    jac = np.zeros(3 * N + 1)
    dx = z[3 * i] - z[3 * j]
    dy = z[3 * i + 1] - z[3 * j + 1]
    ri = z[3 * i + 2]
    rj = z[3 * j + 2]
    jac[3 * i] = 2 * dx
    jac[3 * i + 1] = 2 * dy
    jac[3 * i + 2] = -2 * (ri + rj)
    jac[3 * j] = -2 * dx
    jac[3 * j + 1] = -2 * dy
    jac[3 * j + 2] = -2 * (ri + rj)
    return jac


def _greedy_expand(x, y, r, w, h, time_budget):
    """Iteratively expand each circle to its maximum possible radius."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    improved = True
    rounds = 0
    while improved and rounds < 30:
        if time.time() > time_budget:
            break
        improved = False
        rounds += 1
        for i in range(N):
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-9:
                r[i] = r_max
                improved = True
    return x, y, r


def _jiggle_polish(z, w, h, rng, n_iter=20, time_budget=None):
    """
    Jiggle polish: perturb positions slightly, then re-optimize radii via greedy expansion.
    This helps escape local minima in the radius landscape.
    """
    x = z[0::3][:N].copy()
    y = z[1::3][:N].copy()
    r = z[2::3][:N].copy()
    best_val = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    for it in range(n_iter):
        if time_budget is not None and time.time() > time_budget:
            break
        # Jiggle positions
        xj = x + rng.normal(0, 0.005, N)
        yj = y + rng.normal(0, 0.005, N)

        # Clamp to walls
        for i in range(N):
            xj[i] = np.clip(xj[i], r[i], w - r[i])
            yj[i] = np.clip(yj[i], r[i], h - r[i])

        # Greedy expand radii from jigged positions
        x2, y2, r2 = _greedy_expand(xj, yj, r.copy(), w, h, time_budget)

        if _is_valid(x2, y2, r2, w, h):
            val = np.sum(r2)
            if val > best_val:
                best_val = val
                best_x, best_y, best_r = x2.copy(), y2.copy(), r2.copy()

    # Update z
    z_new = z.copy()
    z_new[0::3][:N] = best_x
    z_new[1::3][:N] = best_y
    z_new[2::3][:N] = best_r
    return z_new


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT

    best_value = 0.0
    best_result = None

    rng = np.random.default_rng(12345)

    # Phase 1: Differential Evolution over aspect ratio + positions
    # We'll use DE to search over w and a reduced set of parameters
    # For DE, we optimize over w and all x, y, r (3N+1 vars)
    # But DE with 64 vars is expensive, so we use a compact representation:
    # Optimize over w and grid-based initial positions, then refine

    # Actually, let's do a hybrid:
    # 1. Try several aspect ratios with cellular init + L-BFGS-B
    # 2. Use DE only on aspect ratio (1D) with a surrogate from best L-BFGS-B results
    # 3. Final SLSQP + jiggle polish

    # Phase 1a: Multi-start with L-BFGS-B refinement
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        if time.time() > deadline - 20:
            break

        z0 = _cellular_init(w0, h0, N, rng)

        # L-BFGS-B refinement (allows w to change)
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(400, max(100, remaining * 25)))
        z_ref = _lbfgsb_refine(z0, maxiter=maxiter)

        w_ref = z_ref[-1]
        h_ref = 2.0 - w_ref
        x_ref = z_ref[0::3][:N]
        y_ref = z_ref[1::3][:N]
        r_ref = z_ref[2::3][:N]

        if _is_valid(x_ref, y_ref, r_ref, w_ref, h_ref):
            val = float(np.sum(r_ref))
            if val > best_value:
                best_value = val
                best_result = z_ref.copy()

        # SLSQP refinement with fixed aspect ratio
        if time.time() < deadline - 15:
            z_slsqp = _slsqp_refine(z_ref, w_ref, h_ref, maxiter=200)
            x_s = z_slsqp[0::3][:N]
            y_s = z_slsqp[1::3][:N]
            r_s = z_slsqp[2::3][:N]

            if _is_valid(x_s, y_s, r_s, w_ref, h_ref):
                val_s = float(np.sum(r_s))
                if val_s > best_value:
                    best_value = val_s
                    best_result = z_slsqp.copy()

            # Greedy expansion
            if time.time() < deadline - 10:
                x_g, y_g, r_g = _greedy_expand(x_s, y_s, r_s, w_ref, h_ref, deadline - 5)
                if _is_valid(x_g, y_g, r_g, w_ref, h_ref):
                    val_g = float(np.sum(r_g))
                    if val_g > best_value:
                        best_value = val_g
                        z_new = np.empty(3 * N + 1)
                        z_new[0::3][:N] = x_g
                        z_new[1::3][:N] = y_g
                        z_new[2::3][:N] = r_g
                        z_new[-1] = w_ref
                        best_result = z_new

    # Phase 1b: DE over aspect ratio (1D search)
    if time.time() < deadline - 30 and best_result is not None:
        # Use DE to find better aspect ratio
        def de_objective(w_arr):
            w = float(w_arr[0])
            h = 2.0 - w
            if h < 0.3 or w < 0.3:
                return 10.0  # High penalty

            z0 = _cellular_init(w, h, N, rng)
            # Quick L-BFGS-B
            z_ref = _lbfgsb_refine(z0, maxiter=100)
            w_r = z_ref[-1]
            h_r = 2.0 - w_r
            x_r = z_ref[0::3][:N]
            y_r = z_ref[1::3][:N]
            r_r = z_ref[2::3][:N]

            if not _is_valid(x_r, y_r, r_r, w_r, h_r):
                return 10.0
            return -np.sum(r_r)  # Minimize negative sum

        try:
            de_res = differential_evolution(
                de_objective,
                bounds=[(0.4, 1.6)],
                seed=42,
                maxiter=15,
                popsize=8,
                tol=1e-6,
                polish=False,
                init='sobol'
            )

            if de_res.success or de_res.fun < 0:
                w_de = float(de_res.x[0])
                h_de = 2.0 - w_de
                z0_de = _cellular_init(w_de, h_de, N, rng)
                z_ref_de = _lbfgsb_refine(z0_de, maxiter=300)
                w_rd = z_ref_de[-1]
                h_rd = 2.0 - w_rd
                x_rd = z_ref_de[0::3][:N]
                y_rd = z_ref_de[1::3][:N]
                r_rd = z_ref_de[2::3][:N]

                if _is_valid(x_rd, y_rd, r_rd, w_rd, h_rd):
                    val_de = float(np.sum(r_rd))
                    if val_de > best_value:
                        best_value = val_de
                        best_result = z_ref_de.copy()

                # SLSQP on DE result
                if time.time() < deadline - 15:
                    z_slsqp_de = _slsqp_refine(z_ref_de, w_rd, h_rd, maxiter=200)
                    x_sd = z_slsqp_de[0::3][:N]
                    y_sd = z_slsqp_de[1::3][:N]
                    r_sd = z_slsqp_de[2::3][:N]
                    if _is_valid(x_sd, y_sd, r_sd, w_rd, h_rd):
                        val_sd = float(np.sum(r_sd))
                        if val_sd > best_value:
                            best_value = val_sd
                            best_result = z_slsqp_de.copy()

                    if time.time() < deadline - 10:
                        x_gd, y_gd, r_gd = _greedy_expand(x_sd, y_sd, r_sd, w_rd, h_rd, deadline - 5)
                        if _is_valid(x_gd, y_gd, r_gd, w_rd, h_rd):
                            val_gd = float(np.sum(r_gd))
                            if val_gd > best_value:
                                best_value = val_gd
                                z_new = np.empty(3 * N + 1)
                                z_new[0::3][:N] = x_gd
                                z_new[1::3][:N] = y_gd
                                z_new[2::3][:N] = r_gd
                                z_new[-1] = w_rd
                                best_result = z_new
        except Exception:
            pass

    # Phase 2: Final polish on best result
    if best_result is not None and time.time() < deadline - 8:
        w = best_result[-1]
        h = 2.0 - w
        x = best_result[0::3][:N]
        y = best_result[1::3][:N]
        r = best_result[2::3][:N]

        # Jiggle polish
        if time.time() < deadline - 5:
            z_jig = _jiggle_polish(best_result, w, h, rng, n_iter=30, time_budget=deadline - 3)
            x_j = z_jig[0::3][:N]
            y_j = z_jig[1::3][:N]
            r_j = z_jig[2::3][:N]
            if _is_valid(x_j, y_j, r_j, w, h):
                val_j = float(np.sum(r_j))
                if val_j > best_value:
                    best_value = val_j
                    best_result = z_jig.copy()

        # Final SLSQP
        if time.time() < deadline - 5:
            z_final = _slsqp_refine(best_result, w, h, maxiter=200)
            x_f = z_final[0::3][:N]
            y_f = z_final[1::3][:N]
            r_f = z_final[2::3][:N]
            if _is_valid(x_f, y_f, r_f, w, h):
                val_f = float(np.sum(r_f))
                if val_f > best_value:
                    best_value = val_f
                    best_result = z_final.copy()

        # Final greedy expansion
        if time.time() < deadline - 2:
            x2, y2, r2 = _greedy_expand(best_result[0::3][:N], best_result[1::3][:N],
                                         best_result[2::3][:N], w, h, deadline - 1)
            if _is_valid(x2, y2, r2, w, h):
                val2 = float(np.sum(r2))
                if val2 > best_value:
                    best_value = val2
                    z_new = np.empty(3 * N + 1)
                    z_new[0::3][:N] = x2
                    z_new[1::3][:N] = y2
                    z_new[2::3][:N] = r2
                    z_new[-1] = w
                    best_result = z_new

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Safety: minimal shrink for strict validity
    for _ in range(20):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h, tol=1e-12):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
