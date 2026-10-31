"""Circle packing in perimeter-4 rectangle: Multi-phase optimization with SLSQP + iterative expansion."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_value = 0.0
    best_result = None

    iu = np.triu_indices(N, k=1)

    def is_valid(x, y, r, w, h, tol=1e-9):
        if w <= 0.05 or h <= 0.05:
            return False
        if np.any(r <= 0):
            return False
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        d2 = dx * dx + dy * dy
        r_sum = r[iu[0]] + r[iu[1]]
        if np.any(d2 < r_sum * r_sum - tol):
            return False
        return True

    def max_radius_at(x_i, y_i, x, y, r, w, h, exclude_i):
        """Compute maximum radius for circle i at position (x_i, y_i)."""
        r_max = min(x_i, w - x_i, y_i, h - y_i)
        for j in range(N):
            if j == exclude_i:
                continue
            dx = x_i - x[j]
            dy = y_i - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - r[j]
            if r_allowed < r_max:
                r_max = r_allowed
        return max(0.0, r_max)

    def force_relax(x, y, r, w, h, max_iter=2000, tol=1e-12):
        """Force-directed relaxation with adaptive step size."""
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        r = r.copy().astype(float)

        for it in range(max_iter):
            if time.time() > deadline - 15:
                break

            fx = np.zeros(N)
            fy = np.zeros(N)

            # Wall repulsion (strong)
            left_v = np.maximum(r - x, 0.0)
            right_v = np.maximum(x + r - w, 0.0)
            bottom_v = np.maximum(r - y, 0.0)
            top_v = np.maximum(y + r - h, 0.0)
            fx -= left_v * 100.0
            fx += right_v * 100.0
            fy -= bottom_v * 100.0
            fy += top_v * 100.0

            # Pairwise repulsion
            dx = x[iu[0]] - x[iu[1]]
            dy = y[iu[0]] - y[iu[1]]
            d2 = dx * dx + dy * dy
            r_sum = r[iu[0]] + r[iu[1]]
            d = np.sqrt(np.maximum(d2, 1e-12))
            overlap = np.maximum(r_sum - d, 0.0)

            if np.max(overlap) < tol:
                break

            f_pairs = overlap * 50.0
            ux = dx / d
            uy = dy / d

            for k in range(len(iu[0])):
                i = iu[0][k]
                j = iu[1][k]
                fx[i] += ux[k] * f_pairs[k]
                fy[i] += uy[k] * f_pairs[k]
                fx[j] -= ux[k] * f_pairs[k]
                fy[j] -= uy[k] * f_pairs[k]

            max_force = max(np.max(np.abs(fx)), np.max(np.abs(fy)), 1e-8)
            dt = min(0.01, 0.5 / max_force)

            x += dt * fx
            y += dt * fy
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

        return x, y, r

    def expand_all_radii(x, y, r, w, h, max_rounds=30, time_budget=5.0):
        """Iteratively expand all radii to their maximum."""
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        r = r.copy().astype(float)
        start = time.time()

        for rnd in range(max_rounds):
            if time.time() - start > time_budget:
                break
            improved = False
            for i in range(N):
                r_max = max_radius_at(x[i], y[i], x, y, r, w, h, i)
                if r_max > r[i] + 1e-10:
                    # Binary search for exact max
                    lo, hi = r[i], r_max
                    for _ in range(30):
                        mid = (lo + hi) / 2.0
                        r[i] = mid
                        if is_valid(x, y, r, w, h, tol=1e-10):
                            lo = mid
                        else:
                            hi = mid
                    r[i] = lo
                    improved = True
            if not improved:
                break

        return x, y, r

    def local_search(x, y, r, w, h, time_budget=10.0):
        """Local search: move each circle to improve its max radius."""
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        r = r.copy().astype(float)
        start = time.time()

        n_angles = 16
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        steps = [0.08, 0.04, 0.02, 0.01, 0.005, 0.002]

        for rnd in range(15):
            if time.time() - start > time_budget:
                break
            improved = False
            for i in range(N):
                if time.time() - start > time_budget:
                    break
                best_move = None
                best_r = r[i]

                for angle in angles:
                    for step in steps:
                        nx = x[i] + step * np.cos(angle)
                        ny = y[i] + step * np.sin(angle)
                        if nx < 0 or nx > w or ny < 0 or ny > h:
                            continue
                        # Check if position is valid with current r[i]
                        valid = True
                        for j in range(N):
                            if i == j:
                                continue
                            dx = nx - x[j]
                            dy = ny - y[j]
                            d2 = dx * dx + dy * dy
                            rs = r[i] + r[j]
                            if d2 < rs * rs - 1e-9:
                                valid = False
                                break
                        if not valid:
                            continue
                        r_new = max_radius_at(nx, ny, x, y, r, w, h, i)
                        if r_new > best_r + 1e-8:
                            best_r = r_new
                            best_move = (nx, ny)

                if best_move is not None:
                    x[i] = best_move[0]
                    y[i] = best_move[1]
                    r[i] = best_r
                    improved = True

            if not improved:
                break

        return x, y, r

    def slsqp_optimize(x, y, r, w, h, time_budget=10.0):
        """SLSQP optimization with explicit constraints."""
        z0 = np.empty(3 * N + 1)
        z0[0::3][:N] = x
        z0[1::3][:N] = y
        z0[2::3][:N] = r
        z0[-1] = w

        def obj(zv):
            return -np.sum(zv[2::3][:N])

        constraints = []
        bounds = []

        for i in range(N):
            # x bounds
            bounds.append((0.001, 1.999))
            # y bounds
            bounds.append((0.001, 1.999))
            # r bounds
            bounds.append((0.001, 0.5))

        bounds.append((0.3, 1.7))  # w

        # Wall constraints: x_i >= r_i, x_i + r_i <= w, y_i >= r_i, y_i + r_i <= h
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3 * i] - p[3 * i + 2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3 * i] - p[3 * i + 2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3 * i + 1] - p[3 * i + 2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3 * i + 1] - p[3 * i + 2]})

        # Overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda p, i=i, j=j: (p[3 * i] - p[3 * j]) ** 2 + (p[3 * i + 1] - p[3 * j + 1]) ** 2 - (p[3 * i + 2] + p[3 * j + 2]) ** 2
                })

        maxiter = int(min(300, time_budget * 60))
        try:
            res = minimize(obj, z0, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
            zo = res.x
            xo = zo[0::3][:N]
            yo = zo[1::3][:N]
            ro = zo[2::3][:N]
            wo = zo[-1]
            ho = 2.0 - wo
            if is_valid(xo, yo, ro, wo, ho, tol=1e-9):
                return xo, yo, ro, wo
        except Exception:
            pass
        return None

    def lbfgsb_optimize(x, y, r, w, h, time_budget=8.0):
        """L-BFGS-B with penalty objective."""
        z0 = np.empty(3 * N + 1)
        z0[0::3][:N] = x
        z0[1::3][:N] = y
        z0[2::3][:N] = r
        z0[-1] = w

        def objective(zv):
            xv = zv[0::3][:N]
            yv = zv[1::3][:N]
            rv = zv[2::3][:N]
            wv = zv[-1]
            hv = 2.0 - wv
            if wv < 0.2 or hv < 0.2:
                return 1e6
            obj = -np.sum(rv)
            dx = xv[iu[0]] - xv[iu[1]]
            dy = yv[iu[0]] - yv[iu[1]]
            d2 = dx * dx + dy * dy
            r_sum = rv[iu[0]] + rv[iu[1]]
            d = np.sqrt(np.maximum(d2, 1e-12))
            overlap = np.maximum(r_sum - d, 0.0)
            obj += 500.0 * np.sum(overlap * overlap)
            lv = np.maximum(rv - xv, 0.0)
            rv2 = np.maximum(xv + rv - wv, 0.0)
            bv = np.maximum(rv - yv, 0.0)
            tv = np.maximum(yv + rv - hv, 0.0)
            obj += 500.0 * (np.sum(lv * lv) + np.sum(rv2 * rv2) + np.sum(bv * bv) + np.sum(tv * tv))
            return obj

        maxiter = int(min(200, time_budget * 30))
        try:
            res = minimize(objective, z0, method='L-BFGS-B',
                           bounds=[(0.001, 1.999)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'gtol': 1e-10, 'ftol': 1e-14})
            zo = res.x
            xo = zo[0::3][:N]
            yo = zo[1::3][:N]
            ro = zo[2::3][:N]
            wo = zo[-1]
            ho = 2.0 - wo
            if is_valid(xo, yo, ro, wo, ho, tol=1e-9):
                return xo, yo, ro, wo
        except Exception:
            pass
        return None

    def process_configuration(x0, y0, r0, w0, h0, time_budget=20.0):
        """Full pipeline for one initial configuration."""
        x, y, r = force_relax(x0, y0, r0, w0, h0, max_iter=1500)
        x, y, r = expand_all_radii(x, y, r, w0, h0, time_budget=3.0)
        x, y, r = local_search(x, y, r, w0, h0, time_budget=5.0)
        x, y, r = expand_all_radii(x, y, r, w0, h0, time_budget=3.0)

        val = float(np.sum(r))
        result = (x.copy(), y.copy(), r.copy(), w0)

        # SLSQP refinement
        if time.time() < deadline - 25:
            res_sl = slsqp_optimize(x, y, r, w0, h0, time_budget=min(8.0, deadline - time.time() - 20))
            if res_sl is not None:
                xs, ys, rs, ws = res_sl
                xs, ys, rs = expand_all_radii(xs, ys, rs, ws, 2.0 - ws, time_budget=2.0)
                if is_valid(xs, ys, rs, ws, 2.0 - ws):
                    val_s = float(np.sum(rs))
                    if val_s > val:
                        val = val_s
                        result = (xs, ys, rs, ws)

        # L-BFGS-B
        if time.time() < deadline - 15:
            res_lb = lbfgsb_optimize(x, y, r, w0, h0, time_budget=min(6.0, deadline - time.time() - 12))
            if res_lb is not None:
                xs, ys, rs, ws = res_lb
                xs, ys, rs = expand_all_radii(xs, ys, rs, ws, 2.0 - ws, time_budget=2.0)
                if is_valid(xs, ys, rs, ws, 2.0 - ws):
                    val_l = float(np.sum(rs))
                    if val_l > val:
                        val = val_l
                        result = (xs, ys, rs, ws)

        return val, result

    def get_initial_guesses():
        guesses = []

        # Grid-based starts with various aspect ratios
        aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.75, 1.3, 0.7, 1.4, 0.65, 1.5, 0.6]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.35:
                continue
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            # 5x5 grid (last row partial)
            for i in range(N):
                row, col = divmod(i, 5)
                x[i] = (col + 0.5) / 5.0 * w0
                y[i] = (row + 0.5) / 5.0 * h0
                r[i] = 0.4 * min(w0 / 5.0, h0 / 5.0)
            guesses.append((x, y, r, w0))

        # Random starts
        for seed_val in range(12):
            rng = np.random.default_rng(seed_val * 7919 + 31)
            w0 = rng.uniform(0.55, 1.45)
            h0 = 2.0 - w0
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                r0 = rng.uniform(0.03, 0.08)
                x[i] = rng.uniform(r0, w0 - r0)
                y[i] = rng.uniform(r0, h0 - r0)
                r[i] = r0
            guesses.append((x, y, r, w0))

        return guesses

    guesses = get_initial_guesses()

    for idx, (x0, y0, r0, w0) in enumerate(guesses):
        if time.time() > deadline - 35:
            break
        h0 = 2.0 - w0
        time_budget = min(15.0, deadline - time.time() - 30)
        val, result = process_configuration(x0, y0, r0, w0, h0, time_budget)
        if val > best_value:
            best_value = val
            best_result = result

    # Phase 2: Polish best result
    if best_result is not None and time.time() < deadline - 10:
        x, y, r, w = best_result
        h = 2.0 - w

        # Multiple expansion + local search rounds
        for _ in range(3):
            if time.time() > deadline - 8:
                break
            x, y, r = expand_all_radii(x, y, r, w, h, time_budget=2.0)
            x, y, r = local_search(x, y, r, w, h, time_budget=3.0)
            x, y, r = expand_all_radii(x, y, r, w, h, time_budget=2.0)
            if is_valid(x, y, r, w, h):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    best_result = (x.copy(), y.copy(), r.copy(), w)

        # Final SLSQP
        if time.time() < deadline - 5:
            res_sl = slsqp_optimize(x, y, r, w, h, time_budget=min(4.0, deadline - time.time() - 3))
            if res_sl is not None:
                xs, ys, rs, ws = res_sl
                xs, ys, rs = expand_all_radii(xs, ys, rs, ws, 2.0 - ws, time_budget=1.0)
                if is_valid(xs, ys, rs, ws, 2.0 - ws):
                    val_s = float(np.sum(rs))
                    if val_s > best_value:
                        best_value = val_s
                        best_result = (xs, ys, rs, ws)

    # Phase 3: Try aspect ratio variations of best
    if best_result is not None and time.time() < deadline - 8:
        x, y, r, w_orig = best_result
        h_orig = 2.0 - w_orig

        for scale in [0.92, 0.95, 1.05, 1.08]:
            if time.time() > deadline - 5:
                break
            w_try = w_orig * scale
            if w_try < 0.35 or w_try > 1.65:
                continue
            h_try = 2.0 - w_try
            sx = w_try / w_orig
            sy = h_try / h_orig
            xs = x * sx
            ys = y * sy
            rs = r * min(sx, sy) * 0.95

            xs, ys, rs = force_relax(xs, ys, rs, w_try, h_try, max_iter=500)
            xs, ys, rs = expand_all_radii(xs, ys, rs, w_try, h_try, time_budget=2.0)
            if is_valid(xs, ys, rs, w_try, h_try):
                val = float(np.sum(rs))
                if val > best_value:
                    best_value = val
                    best_result = (xs.copy(), ys.copy(), rs.copy(), w_try)

    if best_result is None:
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, 5)
            x[i] = (col + 0.5) / 5.0
            y[i] = (row + 0.5) / 5.0
            r[i] = 0.09
        return np.column_stack((x, y, r))

    x, y, r, w = best_result
    h = 2.0 - w

    # Final safety shrink
    for scale in [1.0, 0.99999, 0.9999, 0.999]:
        r_trial = r * scale
        if is_valid(x, y, r_trial, w, h, tol=1e-9):
            r = r_trial
            break

    return np.column_stack((x, y, r))
