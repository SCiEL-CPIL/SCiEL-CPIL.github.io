import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def is_valid(x, y, r, w, tol=1e-8):
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

    def is_valid_equal_r(x, y, r, w, tol=1e-8):
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

    # --- Vectorized smooth objective with log-sum-exp soft-max (from inspiration) ---
    def smooth_objective_lse(z, beta=50.0, pw=500.0):
        x = z[:N]
        y = z[N:2*N]
        r = z[2*N:3*N]
        w = z[3*N]
        h = 2.0 - w

        if w < 0.25 or h < 0.25 or np.any(r < 0.001):
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

        overlap_vals = r_sum_pairs - d_pairs
        # Log-sum-exp soft-max
        shifted = beta * overlap_vals
        shifted = np.clip(shifted, -500, 500)
        max_s = np.max(shifted)
        lse = max_s + np.log(np.sum(np.exp(shifted - max_s)))
        smooth_max_ov = lse / beta
        hard_ov = np.maximum(overlap_vals, 0.0)
        obj += pw * smooth_max_ov + 100.0 * np.sum(hard_ov * hard_ov)

        # Boundary violations
        x_low_v = r_clipped - x
        x_high_v = x - (w - r_clipped)
        y_low_v = r_clipped - y
        y_high_v = y - (h - r_clipped)
        all_v = np.concatenate([x_low_v, x_high_v, y_low_v, y_high_v])
        shifted_b = beta * all_v
        shifted_b = np.clip(shifted_b, -500, 500)
        max_b = np.max(shifted_b)
        lse_b = max_b + np.log(np.sum(np.exp(shifted_b - max_b)))
        smooth_max_v = lse_b / beta
        hard_v = np.maximum(all_v, 0.0)
        obj += pw * smooth_max_v + 100.0 * np.sum(hard_v * hard_v)

        # Small radius penalty
        obj += 50.0 * np.sum(np.maximum(0.015 - r_clipped, 0.0) ** 2)

        return obj

    # --- Standard quadratic penalty objective (from current) ---
    def smooth_objective_quad(z, penalty_weight=500.0):
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
        obj += penalty_weight * np.sum(overlap ** 2)

        x_low_viol = np.maximum(r_clipped - x, 0.0)
        x_high_viol = np.maximum(x - (w - r_clipped), 0.0)
        y_low_viol = np.maximum(r_clipped - y, 0.0)
        y_high_viol = np.maximum(y - (h - r_clipped), 0.0)
        obj += penalty_weight * (np.sum(x_low_viol ** 2) + np.sum(x_high_viol ** 2) +
                                  np.sum(y_low_viol ** 2) + np.sum(y_high_viol ** 2))
        obj += penalty_weight * np.sum(np.maximum(0.01 - r_clipped, 0.0) ** 2)
        return obj

    # --- Equal-radius position objective ---
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

    # --- Grid initialization ---
    def grid_init(w0, n=N):
        h0 = 2.0 - w0
        best_val = -1
        best_x = best_y = best_r = None

        for rows in range(3, 9):
            for cols in range(3, 9):
                if rows * cols < n:
                    continue
                if rows * cols > n + 8:
                    continue
                cell_w = w0 / cols
                cell_h = h0 / rows
                r_cell = min(cell_w, cell_h) / 2.0 * 0.98
                if r_cell < 0.015:
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

    # --- Hexagonal initialization ---
    def hex_init(w0, n=N):
        h0 = 2.0 - w0
        area = w0 * h0
        r_est = np.sqrt(area / (n * np.pi / 0.9069))
        r_est = min(r_est, 0.15)
        x = np.zeros(n)
        y = np.zeros(n)
        r = np.full(n, r_est)
        dx = 2.0 * r_est * 1.002
        dy = np.sqrt(3) * r_est * 1.002
        row = 0
        col = 0
        idx = 0
        while idx < n:
            cx = dx * (col + 0.5 * (row % 2)) + r_est
            cy = dy * row + r_est
            if cx > w0 - r_est:
                row += 1
                col = 0
                cx = dx * (col + 0.5 * (row % 2)) + r_est
                cy = dy * row + r_est
            if cy > h0 - r_est:
                break
            x[idx] = cx
            y[idx] = cy
            idx += 1
            col += 1
        for i in range(idx, n):
            x[i] = w0 / 2.0
            y[i] = h0 / 2.0
            r[i] = r_est * 0.6
        return x, y, r

    # --- Force relaxation (physical model) ---
    def force_relax(x, y, r, w, max_iter=800, tol=1e-10):
        h = 2.0 - w
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        dt = 0.003
        min_dt = 0.00005

        for it in range(max_iter):
            fx = np.zeros(N)
            fy = np.zeros(N)

            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 200
                if x[i] > w - r[i]:
                    fx[i] -= (x[i] - (w - r[i])) * 200
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 200
                if y[i] > h - r[i]:
                    fy[i] -= (y[i] - (h - r[i])) * 200

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
                        f = overlap * 30.0
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
            if force_mag > 1.0:
                dt = max(min_dt, dt * 0.5)
            elif force_mag < 0.05:
                dt = min(0.008, dt * 1.15)

            x += dt * fx
            y += dt * fy
            for i in range(N):
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])

        return x, y

    # --- Find positions for fixed equal radius ---
    def find_positions_for_radius(r, w, x_init, y_init, max_iter=200):
        h = 2.0 - w
        z0 = np.concatenate([x_init, y_init])
        bounds = [(r, w - r)] * N + [(r, h - r)] * N
        try:
            res = minimize(equal_r_objective, z0, args=(r, w), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-14, 'maxls': 50})
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            return x_opt, y_opt, res.fun
        except Exception:
            return x_init.copy(), y_init.copy(), np.inf

    # --- Binary search for max equal radius ---
    def binary_search_max_radius(w, x_init, y_init, r_low, r_high, max_iter=12, tol=1e-6):
        best_x, best_y, best_r = x_init.copy(), y_init.copy(), r_low
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            r_mid = (r_low + r_high) / 2.0
            if r_mid - r_low < tol:
                break
            x_opt, y_opt, obj_val = find_positions_for_radius(r_mid, w, x_init, y_init, max_iter=150)
            if obj_val < 1e-8 and is_valid_equal_r(x_opt, y_opt, r_mid, w):
                best_x, best_y, best_r = x_opt.copy(), y_opt.copy(), r_mid
                r_low = r_mid
            else:
                r_high = r_mid
        return best_x, best_y, best_r

    # --- SLSQP for equal radius (also optimizes w) ---
    def equal_r_slsqp(x_init, y_init, r_init, w_init, time_limit=15.0):
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return None
        r_target = r_init[0]
        h_init = 2.0 - w_init

        def obj(z):
            x = z[:N]
            y = z[N:2*N]
            w = z[2*N]
            h = 2.0 - w
            if w < 0.3 or h < 0.3:
                return 1e8
            x_low = np.maximum(r_target - x, 0.0)
            x_high = np.maximum(x - (w - r_target), 0.0)
            y_low = np.maximum(r_target - y, 0.0)
            y_high = np.maximum(y - (h - r_target), 0.0)
            wall_pen = 500.0 * (np.sum(x_low ** 2) + np.sum(x_high ** 2) +
                               np.sum(y_low ** 2) + np.sum(y_high ** 2))
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            iu = np.triu_indices(N, k=1)
            d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-15))
            overlap = np.maximum(2 * r_target - d_pairs, 0.0)
            overlap_pen = 500.0 * np.sum(overlap ** 2)
            space_reward = 0.1 * (w * (2.0 - w))
            return -space_reward + wall_pen + overlap_pen

        z0 = np.concatenate([x_init, y_init, [w_init]])
        bounds = [(r_target, w_init - r_target)] * N + \
                 [(r_target, h_init - r_target)] * N + [(0.3, 1.7)]
        try:
            maxiter = int(min(300, max(50, remaining * 20)))
            res = minimize(obj, z0, method='SLSQP', bounds=bounds,
                          options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 80})
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            w_opt = float(np.clip(res.x[2*N], 0.3, 1.7))
            if is_valid_equal_r(x_opt, y_opt, r_target, w_opt):
                r_arr = np.full(N, r_target)
                return x_opt, y_opt, r_arr, w_opt
            x_f, y_f = force_relax(x_opt, y_opt, np.full(N, r_target), w_opt, max_iter=100)
            if is_valid_equal_r(x_f, y_f, r_target, w_opt):
                r_arr = np.full(N, r_target)
                return x_f, y_f, r_arr, w_opt
        except Exception:
            pass
        return None

    # --- SLSQP for variable radii with LSE objective ---
    def smooth_slsqp_lse(x_init, y_init, r_init, w_init, time_limit=18.0):
        z0 = np.concatenate([x_init, y_init, r_init, [w_init]])
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return None
        maxiter = int(min(500, max(100, remaining * 25)))
        h_init = 2.0 - w_init
        bounds = []
        for i in range(N):
            bounds.append((r_init[i] * 0.3, w_init - r_init[i] * 0.3))
        for i in range(N):
            bounds.append((r_init[i] * 0.3, max(r_init[i] * 0.3, h_init - r_init[i] * 0.3)))
        for i in range(N):
            bounds.append((0.01, 0.45))
        bounds.append((0.35, 1.65))
        try:
            res = minimize(smooth_objective_lse, z0, args=(50.0, 500.0), method='SLSQP', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            w_opt = res.x[3*N]
            if np.any(r_opt < 0.005):
                return None
            if is_valid(x_opt, y_opt, r_opt, w_opt):
                return x_opt, y_opt, r_opt, w_opt
            x_f, y_f = force_relax(x_opt, y_opt, r_opt, w_opt, max_iter=200)
            if is_valid(x_f, y_f, r_opt, w_opt):
                return x_f, y_f, r_opt, w_opt
        except Exception:
            pass
        return None

    # --- SLSQP for variable radii with quadratic objective ---
    def smooth_slsqp_quad(x_init, y_init, r_init, w_init, time_limit=18.0):
        z0 = np.concatenate([x_init, y_init, r_init, [w_init]])
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return None
        maxiter = int(min(500, max(100, remaining * 25)))
        h_init = 2.0 - w_init
        bounds = []
        for i in range(N):
            bounds.append((r_init[i] * 0.3, w_init - r_init[i] * 0.3))
        for i in range(N):
            bounds.append((r_init[i] * 0.3, max(r_init[i] * 0.3, h_init - r_init[i] * 0.3)))
        for i in range(N):
            bounds.append((0.01, 0.45))
        bounds.append((0.35, 1.65))
        try:
            res = minimize(smooth_objective_quad, z0, args=(500.0,), method='SLSQP', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            w_opt = res.x[3*N]
            if np.any(r_opt < 0.005):
                return None
            if is_valid(x_opt, y_opt, r_opt, w_opt):
                return x_opt, y_opt, r_opt, w_opt
            x_f, y_f = force_relax(x_opt, y_opt, r_opt, w_opt, max_iter=200)
            if is_valid(x_f, y_f, r_opt, w_opt):
                return x_f, y_f, r_opt, w_opt
        except Exception:
            pass
        return None

    # --- Grow equal radius ---
    def grow_equal_radius(x, y, r, w, max_rounds=10):
        r_cur = r[0]
        for _ in range(max_rounds):
            if time.time() > deadline - 8:
                break
            r_new = r_cur * 1.005
            x_opt, y_opt, obj_val = find_positions_for_radius(r_new, w, x, y, max_iter=100)
            if obj_val < 1e-8 and is_valid_equal_r(x_opt, y_opt, r_new, w):
                x, y, r_cur = x_opt, y_opt, r_new
            else:
                break
        r_arr = np.full(N, r_cur)
        return x, y, r_arr, w

    # --- Relax positions for given radii ---
    def relax_positions(x, y, r, w, max_iter=300, ftol=1e-14):
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
                           options={'maxiter': max_iter, 'ftol': ftol, 'maxls': 50})
            return res.x[:N], res.x[N:2*N]
        except Exception:
            return x.copy(), y.copy()

    # --- Try to grow individual radii ---
    def try_grow(x, y, r, w, max_rounds=5):
        h = 2.0 - w
        for round_idx in range(max_rounds):
            if (time.time() - start_time) > deadline - 8:
                break
            improved = False
            for i in range(N):
                if (time.time() - start_time) > deadline - 8:
                    break
                for _ in range(50):
                    r_new = r.copy()
                    r_new[i] *= 1.003
                    x_t, y_t = relax_positions(x, y, r_new, w, max_iter=40, ftol=1e-12)
                    if is_valid(x_t, y_t, r_new, w):
                        x, y, r = x_t, y_t, r_new
                        improved = True
                    else:
                        break
            if not improved:
                break
        return x, y, r, w

    # --- Nelder-Mead polish ---
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
                           options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14})
            x_t, y_t = res.x[:N], res.x[N:2*N]
            if is_valid(x_t, y_t, r, w):
                return x_t, y_t, r, w
        except Exception:
            pass
        return x, y, r, w

    # --- Newton contact solver ---
    def newton_contact_solve(x_init, y_init, r, w, max_iter=50, tol=1e-12):
        h = 2.0 - w
        x = x_init.copy().astype(float)
        y = y_init.copy().astype(float)

        def active_constraints(x, y):
            constraints = []
            eps = 1e-6
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

        def residual(x, y):
            cons = active_constraints(x, y)
            res = np.zeros(len(cons))
            for k, (ctype, i, j, side) in enumerate(cons):
                if ctype == 'wall':
                    if side == 'left':
                        res[k] = x[i] - r[i]
                    elif side == 'right':
                        res[k] = x[i] - (w - r[i])
                    elif side == 'bottom':
                        res[k] = y[i] - r[i]
                    elif side == 'top':
                        res[k] = y[i] - (h - r[i])
                else:
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy) + 1e-15
                    res[k] = d - (r[i] + r[j])
            return res, cons

        def jacobian(x, y, cons):
            n_cons = len(cons)
            J = np.zeros((n_cons, 2 * N))
            for k, (ctype, i, j, side) in enumerate(cons):
                if ctype == 'wall':
                    if side == 'left' or side == 'right':
                        J[k, i] = 1.0
                    else:
                        J[k, N + i] = 1.0
                else:
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy) + 1e-15
                    J[k, i] = dx / d
                    J[k, N + i] = dy / d
                    J[k, j] = -dx / d
                    J[k, N + j] = -dy / d
            return J

        for it in range(max_iter):
            res, cons = residual(x, y)
            if len(cons) == 0:
                break
            if np.max(np.abs(res)) < tol:
                break
            J = jacobian(x, y, cons)
            try:
                delta = np.linalg.lstsq(J, -res, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            alpha = 1.0
            for _ in range(20):
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
                            dy = x_trial[i] - x_trial[j]
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
            if alpha < 1e-8:
                break
        return x, y

    # --- Multi-start perturbation ---
    def multi_start_perturb(x, y, r, w, rng, n_perturb=3, scale=0.01):
        best_val = sum_radii(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        for _ in range(n_perturb):
            if time.time() > deadline - 10:
                break
            x_p = x + rng.normal(0, scale, N)
            y_p = y + rng.normal(0, scale, N)
            x_p = np.clip(x_p, r, w - r)
            h = 2.0 - w
            y_p = np.clip(y_p, r, h - r)
            x_t, y_t = force_relax(x_p, y_p, r, w, max_iter=300)
            if is_valid(x_t, y_t, r, w):
                val = sum_radii(r)
                if val > best_val:
                    best_val = val
                    best_x, best_y, best_r = x_t.copy(), y_t.copy(), r.copy()
        return best_x, best_y, best_r

    # ==================== MAIN SEARCH ====================
    best_value = 0.0
    best_result = None
    rng = np.random.default_rng(42)

    # Phase 1: Multiple aspect ratios with grid and hex initialization
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.7, 0.55, 2.0, 0.5, 1.1, 0.9, 1.25, 0.8, 1.4, 0.7, 1.6, 0.65]

    for w0 in aspect_ratios:
        if time.time() > deadline - 45:
            break
        h0 = 2.0 - w0
        if h0 < 0.35:
            continue

        # Grid init
        x0, y0, r0 = grid_init(w0)
        x_t, y_t = force_relax(x0, y0, r0, w0, max_iter=500)
        if is_valid(x_t, y_t, r0, w0):
            val = sum_radii(r0)
            if val > best_value:
                best_value = val
                best_result = (x_t.copy(), y_t.copy(), r0.copy(), w0)

        # Hex init
        x0h, y0h, r0h = hex_init(w0)
        x_th, y_th = force_relax(x0h, y0h, r0h, w0, max_iter=500)
        if is_valid(x_th, y_th, r0h, w0):
            val_h = sum_radii(r0h)
            if val_h > best_value:
                best_value = val_h
                best_result = (x_th.copy(), y_th.copy(), r0h.copy(), w0)

        # Equal-radius binary search
        if time.time() < deadline - 40:
            r_init_val = r0[0] if np.allclose(r0, r0[0]) else min(r0)
            x_eq, y_eq, r_eq = binary_search_max_radius(w0, x_t, y_t, r_init_val * 0.9, r_init_val * 1.15, max_iter=10)
            if r_eq > 0.01:
                r_arr = np.full(N, r_eq)
                val_eq = sum_radii(r_arr)
                if val_eq > best_value:
                    best_value = val_eq
                    best_result = (x_eq.copy(), y_eq.copy(), r_arr.copy(), w0)

    # Phase 2: LSE and quadratic SLSQP from best
    if best_result is not None and time.time() < deadline - 35:
        x_cur, y_cur, r_cur, w_cur = best_result

        # LSE SLSQP (new from inspiration)
        result_lse = smooth_slsqp_lse(x_cur, y_cur, r_cur, w_cur, time_limit=18.0)
        if result_lse is not None:
            x_s, y_s, r_s, w_s = result_lse
            val_s = sum_radii(r_s)
            if val_s > best_value:
                best_value = val_s
                best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)

        # Quadratic SLSQP
        if time.time() < deadline - 20:
            x_cur, y_cur, r_cur, w_cur = best_result
            result_q = smooth_slsqp_quad(x_cur, y_cur, r_cur, w_cur, time_limit=12.0)
            if result_q is not None:
                x_s, y_s, r_s, w_s = result_q
                val_s = sum_radii(r_s)
                if val_s > best_value:
                    best_value = val_s
                    best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)

        # Equal-radius SLSQP
        if time.time() < deadline - 15:
            x_cur, y_cur, r_cur, w_cur = best_result
            result_eq = equal_r_slsqp(x_cur, y_cur, r_cur, w_cur, time_limit=12.0)
            if result_eq is not None:
                x_s, y_s, r_s, w_s = result_eq
                val_s = sum_radii(r_s)
                if val_s > best_value:
                    best_value = val_s
                    best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)

    # Phase 3: Grow radii
    if best_result is not None and time.time() < deadline - 25:
        x_cur, y_cur, r_cur, w_cur = best_result

        if np.std(r_cur) < 0.005:
            x_eq, y_eq, r_eq_arr, w_eq = grow_equal_radius(x_cur, y_cur, r_cur, w_cur, max_rounds=8)
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
    if best_result is not None and time.time() < deadline - 18:
        x_cur, y_cur, r_cur, w_cur = best_result
        x_p, y_p, r_p = multi_start_perturb(x_cur, y_cur, r_cur, w_cur, rng, n_perturb=4, scale=0.012)
        if is_valid(x_p, y_p, r_p, w_cur):
            val_p = sum_radii(r_p)
            if val_p > best_value:
                best_value = val_p
                best_result = (x_p.copy(), y_p.copy(), r_p.copy(), w_cur)

    # Phase 5: Polish
    if best_result is not None and time.time() < deadline - 15:
        x_cur, y_cur, r_cur, w_cur = best_result
        x_cur, y_cur, r_cur, w_cur = polish(x_cur, y_cur, r_cur, w_cur)
        if is_valid(x_cur, y_cur, r_cur, w_cur):
            val = sum_radii(r_cur)
            if val > best_value:
                best_value = val
                best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)

        if time.time() < deadline - 10:
            x_cur, y_cur, r_cur, w_cur = best_result
            x_new, y_new = newton_contact_solve(x_cur, y_cur, r_cur, w_cur, max_iter=80, tol=1e-12)
            if is_valid(x_new, y_new, r_cur, w_cur):
                val_new = sum_radii(r_cur)
                if val_new > best_value:
                    best_value = val_new
                    best_result = (x_new.copy(), y_new.copy(), r_cur.copy(), w_cur)

    # Phase 6: Second optimization pass with both objectives
    if best_result is not None and time.time() < deadline - 30:
        x_cur, y_cur, r_cur, w_cur = best_result

        result_lse2 = smooth_slsqp_lse(x_cur, y_cur, r_cur, w_cur, time_limit=15.0)
        if result_lse2 is not None:
            x_s, y_s, r_s, w_s = result_lse2
            val_s = sum_radii(r_s)
            if val_s > best_value:
                best_value = val_s
                best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)

        if time.time() < deadline - 15:
            x_cur, y_cur, r_cur, w_cur = best_result
            result_q2 = smooth_slsqp_quad(x_cur, y_cur, r_cur, w_cur, time_limit=10.0)
            if result_q2 is not None:
                x_s, y_s, r_s, w_s = result_q2
                val_s = sum_radii(r_s)
                if val_s > best_value:
                    best_value = val_s
                    best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)

        if time.time() < deadline - 10:
            x_cur, y_cur, r_cur, w_cur = best_result
            result_eq2 = equal_r_slsqp(x_cur, y_cur, r_cur, w_cur, time_limit=8.0)
            if result_eq2 is not None:
                x_s, y_s, r_s, w_s = result_eq2
                val_s = sum_radii(r_s)
                if val_s > best_value:
                    best_value = val_s
                    best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)

    # Phase 7: Random restarts
    if time.time() < deadline - 30:
        for _ in range(12):
            if time.time() > deadline - 22:
                break
            w0 = rng.uniform(0.55, 1.45)
            h0 = 2.0 - w0
            if h0 < 0.35:
                continue

            if rng.random() < 0.5:
                x0, y0, r0 = grid_init(w0)
            else:
                x0, y0, r0 = hex_init(w0)

            x0 += rng.normal(0, 0.008, N)
            y0 += rng.normal(0, 0.008, N)

            x_t, y_t = force_relax(x0, y0, r0, w0, max_iter=300)
            if not is_valid(x_t, y_t, r0, w0):
                continue
            val = sum_radii(r0)

            if val > best_value * 0.92:
                if time.time() < deadline - 20:
                    r_init_val = r0[0] if np.allclose(r0, r0[0]) else min(r0)
                    x_eq, y_eq, r_eq = binary_search_max_radius(w0, x_t, y_t, r_init_val * 0.95, r_init_val * 1.1, max_iter=8)
                    if r_eq > 0.01:
                        r_arr = np.full(N, r_eq)
                        val_eq = sum_radii(r_arr)
                        if val_eq > best_value:
                            best_value = val_eq
                            best_result = (x_eq.copy(), y_eq.copy(), r_arr.copy(), w0)
                            continue

                if time.time() < deadline - 15:
                    result = smooth_slsqp_lse(x_t, y_t, r0, w0, time_limit=8.0)
                    if result is not None:
                        x_s, y_s, r_s, w_s = result
                        val_s = sum_radii(r_s)
                        if val_s > best_value:
                            best_value = val_s
                            best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)
                            continue

                if time.time() < deadline - 10:
                    x_g, y_g, r_g, w_g = try_grow(x_t, y_t, r0, w0, max_rounds=3)
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
            x_hp, y_hp = newton_contact_solve(x_cur, y_cur, r_cur, w_cur, max_iter=150, tol=1e-14)
            if is_valid(x_hp, y_hp, r_cur, w_cur):
                val_hp = sum_radii(r_cur)
                if val_hp > best_value:
                    best_value = val_hp
                    best_result = (x_hp.copy(), y_hp.copy(), r_cur.copy(), w_cur)

        if time.time() < deadline - 4:
            x_cur, y_cur, r_cur, w_cur = best_result
            if np.std(r_cur) < 0.005:
                x_eq, y_eq, r_eq_arr, w_eq = grow_equal_radius(x_cur, y_cur, r_cur, w_cur, max_rounds=5)
                if is_valid_equal_r(x_eq, y_eq, r_eq_arr[0], w_eq):
                    val_eq = sum_radii(r_eq_arr)
                    if val_eq > best_value:
                        best_value = val_eq
                        best_result = (x_eq.copy(), y_eq.copy(), r_eq_arr.copy(), w_eq)
            else:
                x_cur, y_cur, r_cur, w_cur = try_grow(x_cur, y_cur, r_cur, w_cur, max_rounds=3)
                if is_valid(x_cur, y_cur, r_cur, w_cur):
                    val = sum_radii(r_cur)
                    if val > best_value:
                        best_value = val
                        best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)

            if time.time() < deadline - 2:
                x_cur, y_cur, r_cur, w_cur = best_result
                x_fn, y_fn = newton_contact_solve(x_cur, y_cur, r_cur, w_cur, max_iter=60, tol=1e-13)
                if is_valid(x_fn, y_fn, r_cur, w_cur):
                    val_fn = sum_radii(r_cur)
                    if val_fn > best_value:
                        best_value = val_fn
                        best_result = (x_fn.copy(), y_fn.copy(), r_cur.copy(), w_cur)

    # Fallback
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

    # Final Newton
    if time.time() < deadline - 1:
        x_final, y_final = newton_contact_solve(x_final, y_final, r_final, w_final, max_iter=40, tol=1e-13)
        if not is_valid(x_final, y_final, r_final, w_final):
            x_final, y_final, r_final, w_final = best_result

    return np.column_stack((x_final, y_final, r_final))
