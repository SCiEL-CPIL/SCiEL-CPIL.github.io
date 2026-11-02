import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def check_time():
        return time.time() > deadline

    def time_remaining():
        return deadline - time.time()

    def is_valid_config(w, h, r, xs, ys, tol=1e-8):
        if w <= 0 or h <= 0 or r <= 0:
            return False
        if w + h > 4.0 + 1e-7:
            return False
        if np.any(xs < r - tol) or np.any(xs > w - r + tol):
            return False
        if np.any(ys < r - tol) or np.any(ys > h - r + tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                if dx * dx + dy * dy < 4.0 * r * r - tol:
                    return False
        return True

    def pack_hexagonal(rows):
        if sum(rows) != N:
            return None
        pts = []
        for k, n_circles in enumerate(rows):
            y_k = k * np.sqrt(3)
            offset = 1.0 if k % 2 == 1 else 0.0
            for i in range(n_circles):
                x_k = i * 2.0 + offset
                pts.append((x_k, y_k))
        pts = np.array(pts)
        if len(pts) != N:
            return None
        center = pts.mean(axis=0)
        pts -= center
        xs = pts[:, 0]
        ys = pts[:, 1]
        w_centers = xs.max() - xs.min()
        h_centers = ys.max() - ys.min()
        w_total = w_centers + 2.0
        h_total = h_centers + 2.0
        scale = 2.0 / (w_total + h_total)
        r = scale
        w = scale * w_total
        h = scale * h_total
        xs_final = pts[:, 0] * scale + w / 2.0
        ys_final = pts[:, 1] * scale + h / 2.0
        return w, h, r, xs_final, ys_final

    def pack_grid(rows, cols):
        pts = []
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                pts.append((i * 2.0, j * 2.0))
                count += 1
            if count >= N:
                break
        pts = np.array(pts[:N])
        center = pts.mean(axis=0)
        pts -= center
        xs = pts[:, 0]
        ys = pts[:, 1]
        w_centers = xs.max() - xs.min()
        h_centers = ys.max() - ys.min()
        w_total = w_centers + 2.0
        h_total = h_centers + 2.0
        scale = 2.0 / (w_total + h_total)
        r = scale
        w = scale * w_total
        h = scale * h_total
        xs_final = pts[:, 0] * scale + w / 2.0
        ys_final = pts[:, 1] * scale + h / 2.0
        return w, h, r, xs_final, ys_final

    def physics_simulate(w, h, r, xs, ys, max_steps=100):
        if check_time():
            return w, h, r, xs, ys
        x = xs.copy()
        y = ys.copy()

        def relax(x, y, r_val, w_val, h_val, steps=50):
            for _ in range(steps):
                if check_time():
                    break
                fx = np.zeros(N)
                fy = np.zeros(N)
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist_sq = dx * dx + dy * dy
                        min_dist = 2.0 * r_val
                        if dist_sq < min_dist * min_dist:
                            dist = np.sqrt(dist_sq)
                            if dist > 1e-6:
                                overlap = min_dist - dist
                                force = 10.0 * overlap
                                fx[i] += (dx / dist) * force
                                fy[i] += (dy / dist) * force
                                fx[j] -= (dx / dist) * force
                                fy[j] -= (dy / dist) * force
                            else:
                                fx[i] += np.random.uniform(-1, 1)
                                fy[i] += np.random.uniform(-1, 1)
                                fx[j] -= np.random.uniform(-1, 1)
                                fy[j] -= np.random.uniform(-1, 1)
                for i in range(N):
                    if x[i] < r_val:
                        fx[i] += 10.0 * (r_val - x[i])
                    if x[i] > w_val - r_val:
                        fx[i] -= 10.0 * (x[i] - (w_val - r_val))
                    if y[i] < r_val:
                        fy[i] += 10.0 * (r_val - y[i])
                    if y[i] > h_val - r_val:
                        fy[i] -= 10.0 * (y[i] - (h_val - r_val))
                x += 0.01 * fx
                y += 0.01 * fy
                x = np.clip(x, r_val, w_val - r_val)
                y = np.clip(y, r_val, h_val - r_val)
            return x, y

        if not is_valid_config(x, y, r, w, h):
            x, y = relax(x, y, r, w, h, steps=80)
            if not is_valid_config(x, y, r, w, h):
                return w, h, r, xs, ys

        max_r_possible = min(w, h) / 2.0
        r_try = r
        delta_r = 0.001
        improved = False
        while r_try < max_r_possible - 1e-4 and not check_time():
            x_new, y_new = relax(x, y, r_try + delta_r, w, h, steps=15)
            if is_valid_config(x_new, y_new, r_try + delta_r, w, h):
                r_try += delta_r
                x, y = x_new, y_new
                improved = True
            else:
                break
        if improved:
            return w, h, r_try, x, y
        return w, h, r, x, y

    def refine_smooth(w, h, r, xs, ys, is_polish=False):
        if check_time():
            return w, h, r, xs, ys

        w, h, r, xs, ys = physics_simulate(w, h, r, xs, ys)

        v0 = np.zeros(2 + 2 * N)
        v0[0] = w
        v0[1] = r
        v0[2:2 + N] = xs
        v0[2 + N:2 + 2 * N] = ys

        beta = 50.0 if not is_polish else 200.0
        penalty_weight = 10.0 if not is_polish else 50.0

        idx_i, idx_j = np.triu_indices(N, k=1)

        def softplus(x, beta_val=50.0):
            return np.where(x > 1.0 / beta_val, x, np.log1p(np.exp(beta_val * x)) / beta_val)

        def objective(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            if w_val < 0.1 or w_val > 1.9 or r_val < 0.01 or r_val > 1.0:
                return 1e6
            xs_v = v[2:2 + N]
            ys_v = v[2 + N:2 + 2 * N]
            penalty = 0.0
            viol_x_left = r_val - xs_v
            viol_x_right = xs_v - (w_val - r_val)
            viol_y_bottom = r_val - ys_v
            viol_y_top = ys_v - (h_val - r_val)
            penalty += np.sum(softplus(viol_x_left, beta))
            penalty += np.sum(softplus(viol_x_right, beta))
            penalty += np.sum(softplus(viol_y_bottom, beta))
            penalty += np.sum(softplus(viol_y_top, beta))
            dx = xs_v[idx_i] - xs_v[idx_j]
            dy = ys_v[idx_i] - ys_v[idx_j]
            dist = np.sqrt(dx * dx + dy * dy + 1e-12)
            viol = 2.0 * r_val - dist
            penalty += np.sum(softplus(viol, beta))
            return -r_val + penalty_weight * penalty

        def grad_objective(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            xs_v = v[2:2 + N]
            ys_v = v[2 + N:2 + 2 * N]
            grad = np.zeros_like(v)
            grad[1] = -1.0

            def d_softplus(x, beta_val=50.0):
                z = beta_val * x
                sig = np.where(z > 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
                return sig * beta_val

            viol_x_left = r_val - xs_v
            viol_x_right = xs_v - (w_val - r_val)
            viol_y_bottom = r_val - ys_v
            viol_y_top = ys_v - (h_val - r_val)
            ds_xl = d_softplus(viol_x_left, beta)
            ds_xr = d_softplus(viol_x_right, beta)
            ds_yb = d_softplus(viol_y_bottom, beta)
            ds_yt = d_softplus(viol_y_top, beta)
            grad[2:2 + N] -= penalty_weight * ds_xl
            grad[2:2 + N] += penalty_weight * ds_xr
            grad[2 + N:2 + 2 * N] -= penalty_weight * ds_yb
            grad[2 + N:2 + 2 * N] += penalty_weight * ds_yt
            grad[1] += penalty_weight * (np.sum(ds_xl) + np.sum(ds_yb))
            grad[0] -= penalty_weight * np.sum(ds_xr)
            dx = xs_v[idx_i] - xs_v[idx_j]
            dy = ys_v[idx_i] - ys_v[idx_j]
            dist = np.sqrt(dx * dx + dy * dy + 1e-12)
            viol = 2.0 * r_val - dist
            ds = d_softplus(viol, beta)
            grad_xi = penalty_weight * ds * (dx / dist)
            grad_xj = -penalty_weight * ds * (dx / dist)
            grad_yi = penalty_weight * ds * (dy / dist)
            grad_yj = -penalty_weight * ds * (dy / dist)
            np.add.at(grad[2:2 + N], idx_i, -grad_xi)
            np.add.at(grad[2:2 + N], idx_j, grad_xj)
            np.add.at(grad[2 + N:2 + 2 * N], idx_i, -grad_yi)
            np.add.at(grad[2 + N:2 + 2 * N], idx_j, grad_yj)
            grad[1] += penalty_weight * np.sum(ds * 2.0)
            return grad

        bounds = [(0.2, 1.8), (0.01, 1.0)] + [(0.0, 2.0)] * (2 * N)

        try:
            if is_polish:
                options = {'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-14, 'disp': False, 'maxls': 50}
            else:
                options = {'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10, 'disp': False, 'maxls': 50}

            res = minimize(objective, v0, method='L-BFGS-B', jac=grad_objective, bounds=bounds, options=options)
            if res.fun < -r + 1e-6:
                w_res = res.x[0]
                r_res = res.x[1]
                h_res = 2.0 - w_res
                xs_res = res.x[2:2 + N]
                ys_res = res.x[2 + N:2 + 2 * N]
                if is_valid_config(w_res, h_res, r_res, xs_res, ys_res):
                    return w_res, h_res, r_res, xs_res, ys_res
        except Exception:
            pass
        return w, h, r, xs, ys

    def refine_slsqp(w, h, r, xs, ys, is_polish=False):
        if check_time():
            return w, h, r, xs, ys

        v0 = np.zeros(2 + 2 * N)
        v0[0] = w
        v0[1] = r
        v0[2:2 + N] = xs
        v0[2 + N:2 + 2 * N] = ys

        def objective(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            if w_val <= 0.1 or h_val <= 0.1 or r_val <= 0:
                return 1e6
            return -r_val

        def constraints(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            xs_v = v[2:2 + N]
            ys_v = v[2 + N:2 + 2 * N]
            cons = []
            cons.append({'type': 'ineq', 'fun': lambda v: v[0] - 0.2})
            cons.append({'type': 'ineq', 'fun': lambda v: 2.0 - v[0] - 0.2})
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2 + i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[2 + i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2 + N + i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[0]) - v[2 + N + i] - v[1]})
            for i in range(N):
                for j in range(i + 1, N):
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda v, i=i, j=j: ((v[2 + i] - v[2 + j]) ** 2 + (v[2 + N + i] - v[2 + N + j]) ** 2) - 4 * (v[1] ** 2)
                    })
            return cons

        bounds = [(0.2, 1.8), (0.01, 1.0)] + [(0.0, 2.0)] * (2 * N)

        try:
            if is_polish:
                options = {'maxiter': 300, 'ftol': 1e-12, 'disp': False}
            else:
                options = {'maxiter': 100, 'ftol': 1e-9, 'disp': False}

            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=constraints, options=options)
            if res.success or res.fun < -r:
                w_res = res.x[0]
                r_res = res.x[1]
                h_res = 2.0 - w_res
                xs_res = res.x[2:2 + N]
                ys_res = res.x[2 + N:2 + 2 * N]
                if is_valid_config(w_res, h_res, r_res, xs_res, ys_res):
                    return w_res, h_res, r_res, xs_res, ys_res
        except Exception:
            pass
        return w, h, r, xs, ys

    best_r = 0
    best_config = None

    configs = []
    configs.append(pack_hexagonal([7, 7, 7]))
    configs.append(pack_hexagonal([4, 5, 4, 4, 4]))
    configs.append(pack_hexagonal([3, 4, 4, 4, 4, 2]))
    configs.append(pack_hexagonal([5, 5, 5, 3, 3]))
    configs.append(pack_hexagonal([6, 6, 5, 4]))
    configs.append(pack_hexagonal([8, 7, 6]))
    configs.append(pack_hexagonal([9, 6, 6]))
    configs.append(pack_hexagonal([10, 5, 6]))
    configs.append(pack_hexagonal([11, 5, 5]))
    configs.append(pack_hexagonal([12, 4, 5]))
    configs.append(pack_hexagonal([13, 4, 4]))
    configs.append(pack_hexagonal([14, 4, 3]))
    configs.append(pack_hexagonal([15, 4, 2]))
    configs.append(pack_hexagonal([16, 3, 2]))
    configs.append(pack_hexagonal([17, 3, 1]))
    configs.append(pack_hexagonal([18, 2, 1]))
    configs.append(pack_hexagonal([19, 2]))
    configs.append(pack_hexagonal([20, 1]))
    configs.append(pack_hexagonal([21]))
    configs.append(pack_hexagonal([1, 20]))
    configs.append(pack_hexagonal([1, 10, 10]))
    configs.append(pack_hexagonal([1, 11, 9]))
    configs.append(pack_hexagonal([2, 19]))
    configs.append(pack_hexagonal([2, 10, 9]))
    configs.append(pack_hexagonal([2, 11, 8]))
    configs.append(pack_hexagonal([3, 18]))
    configs.append(pack_hexagonal([3, 10, 8]))
    configs.append(pack_hexagonal([3, 11, 7]))
    configs.append(pack_hexagonal([4, 17]))
    configs.append(pack_hexagonal([4, 10, 7]))
    configs.append(pack_hexagonal([4, 11, 6]))
    configs.append(pack_hexagonal([5, 16]))
    configs.append(pack_hexagonal([5, 10, 6]))
    configs.append(pack_hexagonal([5, 11, 5]))
    configs.append(pack_hexagonal([6, 15]))
    configs.append(pack_hexagonal([6, 10, 5]))
    configs.append(pack_hexagonal([6, 11, 4]))
    configs.append(pack_hexagonal([7, 14]))
    configs.append(pack_hexagonal([7, 10, 4]))
    configs.append(pack_hexagonal([7, 11, 3]))
    configs.append(pack_hexagonal([8, 13]))
    configs.append(pack_hexagonal([8, 10, 3]))
    configs.append(pack_hexagonal([8, 11, 2]))
    configs.append(pack_hexagonal([9, 12]))
    configs.append(pack_hexagonal([9, 10, 2]))
    configs.append(pack_hexagonal([9, 11, 1]))
    configs.append(pack_hexagonal([10, 11]))
    configs.append(pack_hexagonal([10, 10, 1]))

    configs.append(pack_grid(5, 5))
    configs.append(pack_grid(3, 7))
    configs.append(pack_grid(4, 6))
    configs.append(pack_grid(7, 3))
    configs.append(pack_grid(6, 4))
    configs.append(pack_grid(7, 4))
    configs.append(pack_grid(4, 7))
    configs.append(pack_grid(5, 4))
    configs.append(pack_grid(4, 5))
    configs.append(pack_grid(3, 8))
    configs.append(pack_grid(8, 3))

    np.random.seed(42)
    for _ in range(8):
        if check_time():
            break
        w = 1.0
        h = 1.0
        r = 0.08
        xs = np.random.uniform(r, w - r, N)
        ys = np.random.uniform(r, h - r, N)
        configs.append((w, h, r, xs, ys))

    valid_configs = [c for c in configs if c is not None]

    for cfg in valid_configs:
        if check_time():
            break
        w, h, r, xs, ys = cfg
        w_new, h_new, r_new, xs_new, ys_new = refine_smooth(w, h, r, xs, ys, is_polish=False)
        if r_new > best_r:
            best_r = r_new
            best_config = (w_new, h_new, r_new, xs_new, ys_new)

    if best_config is not None and not check_time():
        w, h, r, xs, ys = best_config
        w_new, h_new, r_new, xs_new, ys_new = refine_smooth(w, h, r, xs, ys, is_polish=True)
        if r_new > best_r:
            best_r = r_new
            best_config = (w_new, h_new, r_new, xs_new, ys_new)

    if best_config is not None and not check_time():
        w, h, r, xs, ys = best_config
        w_new, h_new, r_new, xs_new, ys_new = refine_slsqp(w, h, r, xs, ys, is_polish=True)
        if r_new > best_r:
            best_r = r_new
            best_config = (w_new, h_new, r_new, xs_new, ys_new)

    if best_config is not None and not check_time() and time_remaining() > 15.0:
        w, h, r, xs, ys = best_config
        w_new, h_new, r_new, xs_new, ys_new = refine_smooth(w, h, r, xs, ys, is_polish=True)
        if r_new > best_r:
            best_r = r_new
            best_config = (w_new, h_new, r_new, xs_new, ys_new)

    if best_config is None:
        w, h, r, xs, ys = valid_configs[0]
        return np.column_stack([xs, ys, np.full(N, r)])

    w, h, r, xs, ys = best_config
    r_final = r * (1 - 1e-6)
    return np.column_stack([xs, ys, np.full(N, r_final)])
