import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def is_valid(x, y, r, w, tol=1e-7):
        h = 2.0 - w
        if w <= tol or h <= tol:
            return False
        for i in range(N):
            if r[i] <= tol:
                return False
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                    return False
        return True

    def is_valid_equal_r(x, y, r, w, tol=1e-7):
        h = 2.0 - w
        if w <= tol or h <= tol or r <= tol:
            return False
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        iu = np.triu_indices(N, k=1)
        if np.any(d_sq[iu] < (2 * r) ** 2 - tol):
            return False
        return True

    def sum_radii(r):
        return float(np.sum(r))

    def grid_init(w0, n=N):
        h0 = 2.0 - w0
        best_val = -1
        best_x = best_y = best_r = None

        for rows in range(3, 10):
            for cols in range(3, 10):
                if rows * cols < n:
                    continue
                if rows * cols > n + 10:
                    continue
                cell_w = w0 / cols
                cell_h = h0 / rows
                r_cell = min(cell_w, cell_h) / 2.0 * 0.97

                if r_cell < 0.012:
                    continue

                x = np.zeros(n)
                y = np.zeros(n)
                r = np.full(n, r_cell)

                idx = 0
                for ri in range(rows):
                    for ci in range(cols):
                        if idx >= n:
                            break
                        cx = (ci + 0.5) * cell_w
                        cy = (ri + 0.5) * cell_h
                        x[idx] = cx
                        y[idx] = cy
                        idx += 1

                if idx < n:
                    for i in range(idx, n):
                        x[i] = w0 / 2.0
                        y[i] = h0 / 2.0
                        r[i] = r_cell * 0.5

                val = sum_radii(r)
                if val > best_val:
                    best_val = val
                    best_x = x.copy()
                    best_y = y.copy()
                    best_r = r.copy()

        if best_x is None:
            cols = 5
            rows = 5
            cell_w = w0 / cols
            cell_h = h0 / rows
            r0 = min(cell_w, cell_h) / 2.0 * 0.9
            x = np.zeros(n)
            y = np.zeros(n)
            r = np.full(n, r0)
            for i in range(n):
                ri, ci = divmod(i, cols)
                x[i] = (ci + 0.5) * cell_w
                y[i] = (ri + 0.5) * cell_h
            best_x, best_y, best_r = x, y, r

        return best_x, best_y, best_r

    def force_relax(x, y, r, w, max_iter=600, tol=1e-10):
        h = 2.0 - w
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        dt = 0.004
        min_dt = 0.00005

        for it in range(max_iter):
            fx = np.zeros(N)
            fy = np.zeros(N)

            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 300
                if x[i] > w - r[i]:
                    fx[i] -= (x[i] - (w - r[i])) * 300
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 300
                if y[i] > h - r[i]:
                    fy[i] -= (y[i] - (h - r[i])) * 300

            max_overlap = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy) + 1e-12
                    r_sum = r[i] + r[j]
                    overlap = r_sum - d
                    if overlap > 0:
                        ux = dx / d
                        uy = dy / d
                        f = overlap * 50.0
                        fx[i] += ux * f
                        fy[i] += uy * f
                        fx[j] -= ux * f
                        fy[j] -= uy * f
                        max_overlap = max(max_overlap, overlap)

            for i in range(N):
                if x[i] < r[i]:
                    max_overlap = max(max_overlap, r[i] - x[i])
                if x[i] > w - r[i]:
                    max_overlap = max(max_overlap, x[i] - (w - r[i]))
                if y[i] < r[i]:
                    max_overlap = max(max_overlap, r[i] - y[i])
                if y[i] > h - r[i]:
                    max_overlap = max(max_overlap, y[i] - (h - r[i]))

            if max_overlap < tol:
                break

            force_mag = np.max(np.sqrt(fx * fx + fy * fy))
            if force_mag > 1.5:
                dt = max(min_dt, dt * 0.5)
            elif force_mag < 0.03:
                dt = min(0.01, dt * 1.2)

            x += dt * fx
            y += dt * fy

            for i in range(N):
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])

        return x, y

    def equal_r_objective(z, r, w):
        x = z[:N]
        y = z[N:2*N]
        h = 2.0 - w
        pen = 0.0
        x_low = np.maximum(r - x, 0.0)
        x_high = np.maximum(x - (w - r), 0.0)
        y_low = np.maximum(r - y, 0.0)
        y_high = np.maximum(y - (h - r), 0.0)
        pen += np.sum(x_low ** 2) + np.sum(x_high ** 2)
        pen += np.sum(y_low ** 2) + np.sum(y_high ** 2)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        iu = np.triu_indices(N, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-15))
        overlap = np.maximum(2 * r - d_pairs, 0.0)
        pen += np.sum(overlap ** 2)
        return pen

    def find_positions_for_radius(r, w, x_init, y_init, max_iter=300):
        h = 2.0 - w
        z0 = np.concatenate([x_init, y_init])
        bounds = [(r, w - r)] * N + [(r, h - r)] * N
        try:
            res = minimize(equal_r_objective, z0, args=(r, w), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-15, 'maxls': 60})
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            return x_opt, y_opt, res.fun
        except Exception:
            return x_init.copy(), y_init.copy(), np.inf

    def binary_search_max_radius(w, x_init, y_init, r_low, r_high, max_iter=15, tol=1e-7):
        best_x, best_y, best_r = x_init.copy(), y_init.copy(), r_low
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            r_mid = (r_low + r_high) / 2.0
            if r_mid - r_low < tol:
                break
            x_opt, y_opt, obj_val = find_positions_for_radius(r_mid, w, x_init, y_init, max_iter=200)
            if obj_val < 1e-9 and is_valid_equal_r(x_opt, y_opt, r_mid, w):
                best_x, best_y, best_r = x_opt.copy(), y_opt.copy(), r_mid
                r_low = r_mid
            else:
                r_high = r_mid
        return best_x, best_y, best_r

    def smooth_objective(z, penalty_weight=800.0):
        x = z[:N]
        y = z[N:2*N]
        r = z[2*N:3*N]
        w = z[3*N]
        h = 2.0 - w

        if w < 0.2 or h < 0.2:
            return 1e8

        r_clipped = np.clip(r, 0.001, min(w, h) / 2.0 - 0.001)
        obj = -np.sum(r_clipped)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r_clipped[:, None] + r_clipped[None, :]

        iu = np.triu_indices(N, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-15))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        overlap_penalty = penalty_weight * np.sum(overlap ** 2)

        x_low_viol = np.maximum(r_clipped - x, 0.0)
        x_high_viol = np.maximum(x - (w - r_clipped), 0.0)
        y_low_viol = np.maximum(r_clipped - y, 0.0)
        y_high_viol = np.maximum(y - (h - r_clipped), 0.0)

        wall_penalty = penalty_weight * (np.sum(x_low_viol ** 2) + np.sum(x_high_viol ** 2) +
                                         np.sum(y_low_viol ** 2) + np.sum(y_high_viol ** 2))

        small_r_penalty = penalty_weight * np.sum(np.maximum(0.01 - r_clipped, 0.0) ** 2)

        return obj + overlap_penalty + wall_penalty + small_r_penalty

    def smooth_slsqp(x_init, y_init, r_init, w_init, time_limit=18.0):
        z0 = np.concatenate([x_init, y_init, r_init, [w_init]])
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return None
        maxiter = int(min(600, max(100, remaining * 30)))
        h_init = 2.0 - w_init
        bounds = []
        for i in range(N):
            bounds.append((r_init[i] * 0.2, w_init - r_init[i] * 0.2))
        for i in range(N):
            bounds.append((r_init[i] * 0.2, max(r_init[i] * 0.2, h_init - r_init[i] * 0.2)))
        for i in range(N):
            bounds.append((0.01, 0.5))
        bounds.append((0.3, 1.7))
        try:
            res = minimize(smooth_objective, z0, method='SLSQP', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-15, 'maxls': 120})
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            w_opt = res.x[3*N]
            if np.any(r_opt < 0.005):
                return None
            if is_valid(x_opt, y_opt, r_opt, w_opt):
                return x_opt, y_opt, r_opt, w_opt
            x_f, y_f = force_relax(x_opt, y_opt, r_opt, w_opt, max_iter=300)
            if is_valid(x_f, y_f, r_opt, w_opt):
                return x_f, y_f, r_opt, w_opt
        except Exception:
            pass
        return None

    def relax_positions(x, y, r, w, max_iter=400, ftol=1e-15):
        h = 2.0 - w

        def obj(z):
            xz = z[:N]
            yz = z[N:2*N]
            pen = 0.0
            for i in range(N):
                if xz[i] < r[i]:
                    pen += (r[i] - xz[i]) ** 2
                if xz[i] > w - r[i]:
                    pen += (xz[i] - (w - r[i])) ** 2
                if yz[i] < r[i]:
                    pen += (r[i] - yz[i]) ** 2
                if yz[i] > h - r[i]:
                    pen += (yz[i] - (h - r[i])) ** 2
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xz[i] - xz[j]
                    dy = yz[i] - yz[j]
                    d = np.sqrt(dx * dx + dy * dy) + 1e-12
                    ov = r[i] + r[j] - d
                    if ov > 0:
                        pen += ov * ov
            return pen

        z0 = np.concatenate([x, y])
        bounds = [(r[i], w - r[i]) for i in range(N)] + \
                 [(r[i], h - r[i]) for i in range(N)]
        try:
            res = minimize(obj, z0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': ftol, 'maxls': 60})
            return res.x[:N], res.x[N:2*N]
        except Exception:
            return x.copy(), y.copy()

    def try_grow(x, y, r, w, max_rounds=6):
        h = 2.0 - w
        for round_idx in range(max_rounds):
            if (time.time() - start_time) > deadline - 8:
                break
            improved = False
            for i in range(N):
                if (time.time() - start_time) > deadline - 8:
                    break
                for _ in range(60):
                    r_new = r.copy()
                    r_new[i] *= 1.004
                    x_t, y_t = relax_positions(x, y, r_new, w, max_iter=50, ftol=1e-13)
                    if is_valid(x_t, y_t, r_new, w):
                        x, y, r = x_t, y_t, r_new
                        improved = True
                    else:
                        break
            if not improved:
                break
        return x, y, r, w

    def grow_equal_radius(x, y, r, w, max_rounds=12):
        r_cur = r[0]
        for _ in range(max_rounds):
            if time.time() > deadline - 8:
                break
            r_new = r_cur * 1.006
            x_opt, y_opt, obj_val = find_positions_for_radius(r_new, w, x, y, max_iter=150)
            if obj_val < 1e-9 and is_valid_equal_r(x_opt, y_opt, r_new, w):
                x, y, r_cur = x_opt, y_opt, r_new
            else:
                break
        r_arr = np.full(N, r_cur)
        return x, y, r_arr, w

    def polish(x, y, r, w):
        h = 2.0 - w

        def obj(z):
            xz = z[:N]
            yz = z[N:2*N]
            pen = 0.0
            for i in range(N):
                if xz[i] < r[i]:
                    pen += (r[i] - xz[i]) ** 2
                if xz[i] > w - r[i]:
                    pen += (xz[i] - (w - r[i])) ** 2
                if yz[i] < r[i]:
                    pen += (r[i] - yz[i]) ** 2
                if yz[i] > h - r[i]:
                    pen += (yz[i] - (h - r[i])) ** 2
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xz[i] - xz[j]
                    dy = yz[i] - yz[j]
                    d = np.sqrt(dx * dx + dy * dy) + 1e-12
                    ov = r[i] + r[j] - d
                    if ov > 0:
                        pen += ov * ov
            return pen

        z0 = np.concatenate([x, y])
        try:
            res = minimize(obj, z0, method='Nelder-Mead',
                           options={'maxiter': 800, 'xatol': 1e-13, 'fatol': 1e-15})
            x_t, y_t = res.x[:N], res.x[N:2*N]
            if is_valid(x_t, y_t, r, w):
                return x_t, y_t, r, w
        except Exception:
            pass
        return x, y, r, w

    def newton_contact_solve(x_init, y_init, r, w, max_iter=60, tol=1e-13):
        h = 2.0 - w
        x = x_init.copy().astype(float)
        y = y_init.copy().astype(float)

        def active_constraints(x, y):
            constraints = []
            eps = 1e-5
            for i in range(N):
                if abs(x[i] - r[i]) < eps:
                    constraints.append(('wall', i, 0, 'left'))
                if abs(x[i] - (w - r[i])) < eps:
                    constraints.append(('wall', i, 0, 'right'))
                if abs(y[i] - r[i]) < eps:
                    constraints.append(('wall', i, 0, 'bottom'))
                if abs(y[i] - (h - r[i])) < eps:
                    constraints.append(('wall', i, 0, 'top'))
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if abs(d - (r[i] + r[j])) < eps:
                        constraints.append(('circle', i, j, ''))
            return constraints

        for it in range(max_iter):
            cons = active_constraints(x, y)
            if len(cons) == 0:
                break

            n_cons = len(cons)
            res = np.zeros(n_cons)
            J = np.zeros((n_cons, 2 * N))

            for k, (ctype, i, j, side) in enumerate(cons):
                if ctype == 'wall':
                    if side == 'left':
                        res[k] = x[i] - r[i]
                        J[k, i] = 1.0
                    elif side == 'right':
                        res[k] = x[i] - (w - r[i])
                        J[k, i] = 1.0
                    elif side == 'bottom':
                        res[k] = y[i] - r[i]
                        J[k, N + i] = 1.0
                    elif side == 'top':
                        res[k] = y[i] - (h - r[i])
                        J[k, N + i] = 1.0
                else:
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy) + 1e-15
                    res[k] = d - (r[i] + r[j])
                    J[k, i] = dx / d
                    J[k, N + i] = dy / d
                    J[k, j] = -dx / d
                    J[k, N + j] = -dy / d

            if np.max(np.abs(res)) < tol:
                break

            try:
                delta = np.linalg.lstsq(J, -res, rcond=None)[0]
            except np.linalg.LinAlgError:
                break

            alpha = 1.0
            for _ in range(25):
                x_trial = x + alpha * delta[:N]
                y_trial = y + alpha * delta[N:]
                valid = True
                for i in range(N):
                    if x_trial[i] < r[i] - 1e-10 or x_trial[i] > w - r[i] + 1e-10:
                        valid = False
                        break
                    if y_trial[i] < r[i] - 1e-10 or y_trial[i] > h - r[i] + 1e-10:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i + 1, N):
                            dx = x_trial[i] - x_trial[j]
                            dy = y_trial[i] - y_trial[j]
                            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-10:
                                valid = False
                                break
                        if not valid:
                            break
                if valid:
                    x = x_trial
                    y = y_trial
                    break
                alpha *= 0.5

            if alpha < 1e-10:
                break

        return x, y

    def multi_start_perturb(x, y, r, w, rng, n_perturb=5, scale=0.015):
        best_val = sum_radii(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        h = 2.0 - w

        for _ in range(n_perturb):
            if time.time() > deadline - 10:
                break
            x_p = x + rng.normal(0, scale, N)
            y_p = y + rng.normal(0, scale, N)
            x_p = np.clip(x_p, r, w - r)
            y_p = np.clip(y_p, r, h - r)

            x_t, y_t = force_relax(x_p, y_p, r, w, max_iter=400)
            if is_valid(x_t, y_t, r, w):
                val = sum_radii(r)
                if val > best_val:
                    best_val = val
                    best_x, best_y, best_r = x_t.copy(), y_t.copy(), r.copy()

        return best_x, best_y, best_r

    best_value = 0.0
    best_result = None
    rng = np.random.default_rng(123)

    # Phase 1: Multiple aspect ratios with grid initialization and force relaxation
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.7, 0.55, 2.0, 0.5, 1.1, 0.9, 1.25, 0.8, 1.4, 0.7, 1.6, 0.65]

    for w0 in aspect_ratios:
        if time.time() > deadline - 45:
            break
        h0 = 2.0 - w0
        if h0 < 0.35:
            continue

        x0, y0, r0 = grid_init(w0)
        x_t, y_t = force_relax(x0, y0, r0, w0, max_iter=600)
        if is_valid(x_t, y_t, r0, w0):
            val = sum_radii(r0)
            if val > best_value:
                best_value = val
                best_result = (x_t.copy(), y_t.copy(), r0.copy(), w0)

        # Equal-radius binary search
        if time.time() < deadline - 40:
            r_init_val = r0[0] if np.allclose(r0, r0[0]) else min(r0)
            x_eq, y_eq, r_eq = binary_search_max_radius(w0, x_t, y_t, r_init_val * 0.9, r_init_val * 1.2, max_iter=12)
            if r_eq > 0.01:
                r_arr = np.full(N, r_eq)
                val_eq = sum_radii(r_arr)
                if val_eq > best_value:
                    best_value = val_eq
                    best_result = (x_eq.copy(), y_eq.copy(), r_arr.copy(), w0)

    # Phase 2: SLSQP optimization from best
    if best_result is not None and time.time() < deadline - 35:
        x_cur, y_cur, r_cur, w_cur = best_result
        result = smooth_slsqp(x_cur, y_cur, r_cur, w_cur, time_limit=20.0)
        if result is not None:
            x_s, y_s, r_s, w_s = result
            val_s = sum_radii(r_s)
            if val_s > best_value:
                best_value = val_s
                best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)

    # Phase 3: Grow radii
    if best_result is not None and time.time() < deadline - 25:
        x_cur, y_cur, r_cur, w_cur = best_result

        if np.std(r_cur) < 0.005:
            x_eq, y_eq, r_eq_arr, w_eq = grow_equal_radius(x_cur, y_cur, r_cur, w_cur, max_rounds=10)
            if is_valid_equal_r(x_eq, y_eq, r_eq_arr[0], w_eq):
                val_eq = sum_radii(r_eq_arr)
                if val_eq > best_value:
                    best_value = val_eq
                    best_result = (x_eq.copy(), y_eq.copy(), r_eq_arr.copy(), w_eq)

        if time.time() < deadline - 15:
            x_cur, y_cur, r_cur, w_cur = best_result
            x_cur, y_cur, r_cur, w_cur = try_grow(x_cur, y_cur, r_cur, w_cur)
            if is_valid(x_cur, y_cur, r_cur, w_cur):
                val = sum_radii(r_cur)
                if val > best_value:
                    best_value = val
                    best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)

    # Phase 4: Multi-start perturbation
    if best_result is not None and time.time() < deadline - 20:
        x_cur, y_cur, r_cur, w_cur = best_result
        x_p, y_p, r_p = multi_start_perturb(x_cur, y_cur, r_cur, w_cur, rng, n_perturb=5, scale=0.015)
        if is_valid(x_p, y_p, r_p, w_cur):
            val_p = sum_radii(r_p)
            if val_p > best_value:
                best_value = val_p
                best_result = (x_p.copy(), y_p.copy(), r_p.copy(), w_cur)

    # Phase 5: Polish
    if best_result is not None and time.time() < deadline - 18:
        x_cur, y_cur, r_cur, w_cur = best_result
        x_cur, y_cur, r_cur, w_cur = polish(x_cur, y_cur, r_cur, w_cur)
        if is_valid(x_cur, y_cur, r_cur, w_cur):
            val = sum_radii(r_cur)
            if val > best_value:
                best_value = val
                best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)

        if time.time() < deadline - 12:
            x_cur, y_cur, r_cur, w_cur = best_result
            x_new, y_new = newton_contact_solve(x_cur, y_cur, r_cur, w_cur, max_iter=80, tol=1e-13)
            if is_valid(x_new, y_new, r_cur, w_cur):
                val_new = sum_radii(r_cur)
                if val_new > best_value:
                    best_value = val_new
                    best_result = (x_new.copy(), y_new.copy(), r_cur.copy(), w_cur)

    # Phase 6: Second SLSQP pass
    if best_result is not None and time.time() < deadline - 22:
        x_cur, y_cur, r_cur, w_cur = best_result
        result = smooth_slsqp(x_cur, y_cur, r_cur, w_cur, time_limit=15.0)
        if result is not None:
            x_s, y_s, r_s, w_s = result
            val_s = sum_radii(r_s)
            if val_s > best_value:
                best_value = val_s
                best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)

    # Phase 7: Random restarts
    if time.time() < deadline - 28:
        for _ in range(15):
            if time.time() > deadline - 22:
                break
            w0 = rng.uniform(0.5, 1.5)
            h0 = 2.0 - w0
            if h0 < 0.35:
                continue

            x0, y0, r0 = grid_init(w0)
            x0 += rng.normal(0, 0.01, N)
            y0 += rng.normal(0, 0.01, N)

            x_t, y_t = force_relax(x0, y0, r0, w0, max_iter=400)
            if not is_valid(x_t, y_t, r0, w0):
                continue
            val = sum_radii(r0)

            if val > best_value * 0.90:
                if time.time() < deadline - 20:
                    r_init_val = r0[0] if np.allclose(r0, r0[0]) else min(r0)
                    x_eq, y_eq, r_eq = binary_search_max_radius(w0, x_t, y_t, r_init_val * 0.95, r_init_val * 1.15, max_iter=10)
                    if r_eq > 0.01:
                        r_arr = np.full(N, r_eq)
                        val_eq = sum_radii(r_arr)
                        if val_eq > best_value:
                            best_value = val_eq
                            best_result = (x_eq.copy(), y_eq.copy(), r_arr.copy(), w0)
                            continue

                if time.time() < deadline - 15:
                    result = smooth_slsqp(x_t, y_t, r0, w0, time_limit=10.0)
                    if result is not None:
                        x_s, y_s, r_s, w_s = result
                        val_s = sum_radii(r_s)
                        if val_s > best_value:
                            best_value = val_s
                            best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)
                            continue

                if time.time() < deadline - 10:
                    x_g, y_g, r_g, w_g = try_grow(x_t, y_t, r0, w0, max_rounds=4)
                    if is_valid(x_g, y_g, r_g, w_g):
                        val_g = sum_radii(r_g)
                        if val_g > best_value:
                            best_value = val_g
                            best_result = (x_g.copy(), y_g.copy(), r_g.copy(), w_g)

    # Phase 8: Final polish and growth
    if best_result is not None and time.time() < deadline - 10:
        x_cur, y_cur, r_cur, w_cur = best_result
        x_cur, y_cur, r_cur, w_cur = polish(x_cur, y_cur, r_cur, w_cur)
        if is_valid(x_cur, y_cur, r_cur, w_cur):
            val = sum_radii(r_cur)
            if val > best_value:
                best_value = val
                best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)

        if time.time() < deadline - 6:
            x_cur, y_cur, r_cur, w_cur = best_result
            x_new, y_new = newton_contact_solve(x_cur, y_cur, r_cur, w_cur, max_iter=100, tol=1e-14)
            if is_valid(x_new, y_new, r_cur, w_cur):
                val_new = sum_radii(r_cur)
                if val_new > best_value:
                    best_value = val_new
                    best_result = (x_new.copy(), y_new.copy(), r_cur.copy(), w_cur)

        if time.time() < deadline - 5:
            x_cur, y_cur, r_cur, w_cur = best_result
            if np.std(r_cur) < 0.005:
                x_eq, y_eq, r_eq_arr, w_eq = grow_equal_radius(x_cur, y_cur, r_cur, w_cur, max_rounds=8)
                if is_valid_equal_r(x_eq, y_eq, r_eq_arr[0], w_eq):
                    val_eq = sum_radii(r_eq_arr)
                    if val_eq > best_value:
                        best_value = val_eq
                        best_result = (x_eq.copy(), y_eq.copy(), r_eq_arr.copy(), w_eq)
            else:
                x_cur, y_cur, r_cur, w_cur = try_grow(x_cur, y_cur, r_cur, w_cur, max_rounds=4)
                if is_valid(x_cur, y_cur, r_cur, w_cur):
                    val = sum_radii(r_cur)
                    if val > best_value:
                        best_value = val
                        best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)

            if time.time() < deadline - 3:
                x_cur, y_cur, r_cur, w_cur = best_result
                x_fn, y_fn = newton_contact_solve(x_cur, y_cur, r_cur, w_cur, max_iter=50, tol=1e-14)
                if is_valid(x_fn, y_fn, r_cur, w_cur):
                    val_fn = sum_radii(r_cur)
                    if val_fn > best_value:
                        best_value = val_fn
                        best_result = (x_fn.copy(), y_fn.copy(), r_cur.copy(), w_cur)

    if best_result is None:
        w0 = 1.0
        h0 = 1.0
        cols = 5
        rows = 5
        cell_w = w0 / cols
        cell_h = h0 / rows
        r0 = min(cell_w, cell_h) / 2.0 * 0.45
        cells = []
        for i in range(N):
            row, col = divmod(i, cols)
            cx = (col + 0.5) * cell_w
            cy = (row + 0.5) * cell_h
            cells.append((cx, cy, r0))
        return np.array(cells)

    x_final, y_final, r_final, w_final = best_result

    # Safety shrink
    for _ in range(10):
        r_trial = r_final * (1 - 1e-10)
        if is_valid(x_final, y_final, r_trial, w_final):
            r_final = r_trial
        else:
            break

    # Final Newton solve
    if time.time() < deadline - 2:
        x_final, y_final = newton_contact_solve(x_final, y_final, r_final, w_final, max_iter=40, tol=1e-14)
        if not is_valid(x_final, y_final, r_final, w_final):
            x_final, y_final, r_final, w_final = best_result

    return np.column_stack((x_final, y_final, r_final))
