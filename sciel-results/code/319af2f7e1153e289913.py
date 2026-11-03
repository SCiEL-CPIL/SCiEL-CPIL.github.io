"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def time_left():
        return deadline - time.time()

    best_value = -np.inf
    best_result = None

    def _strictly_valid(circles, w, h, tol=1e-8):
        n = circles.shape[0]
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        if np.any(rs < 1e-9):
            return False
        if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
            return False
        if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
            return False
        dx = xs[:, np.newaxis] - xs[np.newaxis, :]
        dy = ys[:, np.newaxis] - ys[np.newaxis, :]
        d2 = dx * dx + dy * dy
        r_sum = rs[:, np.newaxis] + rs[np.newaxis, :]
        iu = np.triu_indices(n, k=1)
        if np.any(d2[iu] < (r_sum[iu] - tol) ** 2):
            return False
        return True

    def _evaluate(circles, w, h, tol=1e-7):
        if not _strictly_valid(circles, w, h, tol):
            return -np.inf
        return float(np.sum(circles[:, 2]))

    def check_and_update(circles, w, h):
        nonlocal best_value, best_result
        if time_left() < 1.0:
            return
        val = _evaluate(circles, w, h)
        if val > best_value:
            best_value = val
            best_result = circles.copy()

    def _softplus(x, beta=100.0):
        xb = beta * x
        return np.where(xb > 0,
                        x + np.log1p(np.exp(-xb)) / beta,
                        np.log1p(np.exp(xb)) / beta)

    def _softplus_grad(x, beta=100.0):
        return 1.0 / (1.0 + np.exp(-beta * x))

    def _objective_and_grad(x, w, h, beta=100.0, eq_weight=0.05):
        c = x.reshape(-1, 3)
        xs = c[:, 0]
        ys = c[:, 1]
        rs = c[:, 2]

        obj = -np.sum(rs)
        grad = np.zeros_like(x)

        if eq_weight > 0:
            r_mean = np.mean(rs)
            r_dev = rs - r_mean
            obj += eq_weight * np.sum(r_dev * r_dev)
            grad[2::3] += -2.0 * eq_weight * r_dev

        v1 = rs - xs
        obj += np.sum(_softplus(v1, beta)); g1 = _softplus_grad(v1, beta)
        grad[0::3] -= g1; grad[2::3] += g1
        v2 = xs + rs - w
        obj += np.sum(_softplus(v2, beta)); g2 = _softplus_grad(v2, beta)
        grad[0::3] += g2; grad[2::3] += g2
        v3 = rs - ys
        obj += np.sum(_softplus(v3, beta)); g3 = _softplus_grad(v3, beta)
        grad[1::3] -= g3; grad[2::3] += g3
        v4 = ys + rs - h
        obj += np.sum(_softplus(v4, beta)); g4 = _softplus_grad(v4, beta)
        grad[1::3] += g4; grad[2::3] += g4
        v5 = -rs
        obj += np.sum(_softplus(v5, beta)); g5 = _softplus_grad(v5, beta)
        grad[2::3] -= g5

        for i in range(N):
            dx = xs[i] - xs[i + 1:]
            dy = ys[i] - ys[i + 1:]
            d = np.sqrt(dx * dx + dy * dy + 1e-12)
            v = rs[i] + rs[i + 1:] - d
            obj += np.sum(_softplus(v, beta))
            g = _softplus_grad(v, beta)
            grad[3 * i]     += -np.sum(g * dx / d)
            grad[3 * i + 1] += -np.sum(g * dy / d)
            grad[3 * i + 2] += np.sum(g)
            idx = np.arange(i + 1, N)
            grad[3 * idx]     += g * dx / d
            grad[3 * idx + 1] += g * dy / d
            grad[3 * idx + 2] += g

        grad[2::3] += -1.0
        return obj, grad

    def _objective_joint(x, beta=100.0, eq_weight=0.05):
        """Objective with w as free variable; h = 2 - w. x = [w, x0, y0, r0, x1, y1, r1, ...]"""
        w = x[0]
        h = 2.0 - w
        c = x[1:].reshape(-1, 3)
        xs = c[:, 0]
        ys = c[:, 1]
        rs = c[:, 2]

        obj = -np.sum(rs)
        grad = np.zeros_like(x)
        grad_w = 0.0

        if eq_weight > 0:
            r_mean = np.mean(rs)
            r_dev = rs - r_mean
            obj += eq_weight * np.sum(r_dev * r_dev)
            grad[3::3] += -2.0 * eq_weight * r_dev

        v1 = rs - xs
        obj += np.sum(_softplus(v1, beta)); g1 = _softplus_grad(v1, beta)
        grad[1::3] -= g1; grad[3::3] += g1
        v2 = xs + rs - w
        obj += np.sum(_softplus(v2, beta)); g2 = _softplus_grad(v2, beta)
        grad[1::3] += g2; grad[3::3] += g2
        grad_w -= np.sum(g2)
        v3 = rs - ys
        obj += np.sum(_softplus(v3, beta)); g3 = _softplus_grad(v3, beta)
        grad[2::3] -= g3; grad[3::3] += g3
        v4 = ys + rs - h
        obj += np.sum(_softplus(v4, beta)); g4 = _softplus_grad(v4, beta)
        grad[2::3] += g4; grad[3::3] += g4
        grad_w += np.sum(g4)
        v5 = -rs
        obj += np.sum(_softplus(v5, beta)); g5 = _softplus_grad(v5, beta)
        grad[3::3] -= g5

        for i in range(N):
            dx = xs[i] - xs[i + 1:]
            dy = ys[i] - ys[i + 1:]
            d = np.sqrt(dx * dx + dy * dy + 1e-12)
            v = rs[i] + rs[i + 1:] - d
            obj += np.sum(_softplus(v, beta))
            g = _softplus_grad(v, beta)
            grad[1 + 3 * i]     += -np.sum(g * dx / d)
            grad[2 + 3 * i] += -np.sum(g * dy / d)
            grad[3 + 3 * i] += np.sum(g)
            idx = np.arange(i + 1, N)
            grad[1 + 3 * idx]     += g * dx / d
            grad[2 + 3 * idx] += g * dy / d
            grad[3 + 3 * idx] += g

        grad[0] = grad_w
        grad[3::3] += -1.0
        return obj, grad

    def _delaunay_neighbors(centers):
        try:
            tri = Delaunay(centers)
            nbrs = [set() for _ in range(N)]
            for i, j, k in tri.simplices:
                nbrs[i].add(j); nbrs[i].add(k)
                nbrs[j].add(i); nbrs[j].add(k)
                nbrs[k].add(i); nbrs[k].add(j)
            return [sorted(s) for s in nbrs]
        except Exception:
            return [list(range(N)) for _ in range(N)]

    def _fast_valid(circles, w, h, neighbors, tol=1e-9):
        n = circles.shape[0]
        xs = circles[:, 0]; ys = circles[:, 1]; rs = circles[:, 2]
        if np.any(rs < 1e-9):
            return False
        if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
            return False
        if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
            return False
        for i in range(n):
            for j in neighbors[i]:
                if j <= i:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d2 = dx * dx + dy * dy
                need = rs[i] + rs[j] - tol
                if d2 < need * need:
                    return False
        return True

    def _hex_init(w, h, seed):
        rng = random.Random(seed)
        row_configs = [
            [3, 4, 4, 4, 3, 3],
            [4, 4, 4, 4, 3, 2],
            [2, 4, 4, 4, 4, 3],
            [3, 4, 4, 4, 4, 2],
            [4, 3, 4, 4, 4, 2],
            [2, 3, 4, 4, 4, 4],
            [3, 3, 4, 4, 4, 3],
            [4, 4, 3, 4, 4, 2],
            [2, 4, 3, 4, 4, 4],
            [3, 4, 3, 4, 4, 3],
            [5, 4, 4, 4, 4],
            [4, 5, 4, 4, 4],
            [3, 5, 4, 4, 5],
            [4, 4, 5, 4, 4],
            [2, 5, 4, 5, 5],
            [5, 5, 5, 5, 1],
            [1, 5, 5, 5, 5],
            [3, 4, 5, 4, 5],
            [5, 4, 5, 4, 3],
            [4, 4, 4, 4, 4, 1],
            [1, 4, 4, 4, 4, 4],
            [3, 4, 4, 4, 4, 2],
            [2, 4, 4, 4, 4, 3],
            [4, 4, 4, 3, 4, 2],
            [3, 3, 3, 4, 4, 4],
            [4, 4, 4, 3, 3, 3],
        ]
        best = None
        best_val = -np.inf
        for row_counts in row_configs:
            if sum(row_counts) != N:
                continue
            n_rows = len(row_counts)
            max_cols = max(row_counts)
            est_r_h = w / (max_cols + 0.5)
            est_r_v = h / (n_rows * np.sqrt(3) / 2 + 1.0)
            r_est = min(est_r_h, est_r_v) * 0.95
            if r_est < 0.01:
                continue
            circles = []
            for row_idx, n_cols in enumerate(row_counts):
                y = (row_idx + 0.5) * r_est * np.sqrt(3)
                x_offset = r_est if row_idx % 2 == 1 else 0.0
                row_width = n_cols * 2 * r_est
                center_x = w / 2
                for col_idx in range(n_cols):
                    x = center_x - row_width / 2 + x_offset + (col_idx + 0.5) * 2 * r_est
                    circles.append([x, y, r_est])
            circles = np.array(circles)
            valid = True
            for i in range(N):
                xi, yi, ri = circles[i]
                if ri < 1e-9 or xi - ri < -1e-6 or xi + ri > w + 1e-6 or yi - ri < -1e-6 or yi + ri > h + 1e-6:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        xi, yi, ri = circles[i]
                        xj, yj, rj = circles[j]
                        dist_sq = (xi - xj) ** 2 + (yi - yj) ** 2
                        min_dist = ri + rj
                        if dist_sq < (min_dist - 1e-7) ** 2:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                val = float(np.sum(circles[:, 2]))
                if val > best_val:
                    best_val = val
                    best = circles.copy()
        if best is None:
            a = int(np.ceil(np.sqrt(N * w / h)))
            b = int(np.ceil(N / a))
            while a * b < N:
                b += 1
            circles = []
            for i in range(N):
                col = i % a
                row = i // a
                r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
                x = (col + 0.5) * w / a
                y = (row + 0.5) * h / b
                circles.append([x, y, r])
            best = np.array(circles)
        return best

    def _grid_init(w, h, seed):
        rng = random.Random(seed)
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        circles = []
        for i in range(N):
            col = i % a
            row = i // a
            r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
            x = (col + 0.5) * w / a
            y = (row + 0.5) * h / b
            circles.append([x, y, r])
        return np.array(circles)

    def _random_init(w, h, seed):
        rng = np.random.RandomState(seed)
        r0 = min(w, h) / 2.0 * 0.25
        circles = np.zeros((N, 3))
        circles[:, 0] = rng.uniform(r0, w - r0, N)
        circles[:, 1] = rng.uniform(r0, h - r0, N)
        circles[:, 2] = r0 * rng.uniform(0.5, 1.0, N)
        return circles

    def _polish(circles, w, h, beta_start=10.0, max_iter=30):
        x = circles.flatten()
        best_x = x.copy()
        best_val = -np.inf

        if _strictly_valid(circles, w, h):
            best_val = np.sum(circles[:, 2])

        betas = [beta_start, beta_start * 10, beta_start * 100, beta_start * 1000]

        for beta in betas:
            if time_left() < 3.0:
                break

            bounds = []
            r_max = min(w, h) / 2.0
            for i in range(N):
                bounds.append((0, w))
                bounds.append((0, h))
                bounds.append((1e-4, r_max))

            try:
                res = minimize(_objective_and_grad, x, method='L-BFGS-B', bounds=bounds,
                               jac=True, args=(w, h, beta, 0.05),
                               options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10})
                if time_left() < 2.0:
                    break
                new_circles = res.x.reshape(-1, 3)
                if _strictly_valid(new_circles, w, h):
                    val = np.sum(new_circles[:, 2])
                    if val > best_val:
                        best_val = val
                        best_x = res.x.copy()
                x = res.x
            except Exception:
                pass

        return best_x.reshape(-1, 3)

    def _contact_graph_polish(circles, w, h):
        if time_left() < 10.0:
            return circles

        x0 = circles.flatten()
        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-4, r_max))

        stages = [
            (300.0, 0.005, 2000),
            (800.0, 0.002, 3000),
            (2000.0, 0.001, 5000),
            (5000.0, 0.0005, 5000),
        ]

        best_x = x0.copy()
        best_val = -np.inf
        if _strictly_valid(circles, w, h):
            best_val = float(np.sum(circles[:, 2]))

        for beta, eq_w, iters in stages:
            if time_left() < 5.0:
                break
            try:
                res = minimize(_objective_and_grad, x0, method='L-BFGS-B', bounds=bounds,
                               jac=True, args=(w, h, beta, eq_w),
                               options={'maxiter': iters, 'ftol': 1e-14, 'gtol': 1e-12,
                                        'maxls': 80, 'maxfun': 50000})
                if time_left() < 3.0:
                    break
                new_circles = res.x.reshape(-1, 3)
                if _strictly_valid(new_circles, w, h):
                    val = float(np.sum(new_circles[:, 2]))
                    if val > best_val:
                        best_val = val
                        best_x = res.x.copy()
                x0 = res.x
            except Exception:
                pass

        return best_x.reshape(-1, 3)

    def _physics_refine(circles, w, h, n_steps=2000, seed=42):
        if time_left() < 8.0:
            return circles

        rng = np.random.RandomState(seed)
        xs = circles[:, 0].copy().astype(float)
        ys = circles[:, 1].copy().astype(float)
        rs = circles[:, 2].copy().astype(float)

        temp_start = 0.02
        temp_end = 0.001
        n_total = n_steps

        for step in range(n_total):
            if time_left() < 3.0:
                break

            t = step / max(n_total - 1, 1)
            temp = temp_start * (temp_end / temp_start) ** t

            forces_x = np.zeros(N)
            forces_y = np.zeros(N)
            forces_r = np.zeros(N)

            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    d = np.sqrt(dx * dx + dy * dy + 1e-12)
                    overlap = rs[i] + rs[j] - d
                    if overlap > 0:
                        f = overlap * 50.0
                        fx = f * dx / d
                        fy = f * dy / d
                        forces_x[i] += fx
                        forces_y[i] += fy
                        forces_x[j] -= fx
                        forces_y[j] -= fy
                        forces_r[i] -= overlap * 0.5
                        forces_r[j] -= overlap * 0.5

            for i in range(N):
                pen = rs[i] - xs[i]
                if pen > 0:
                    forces_x[i] += pen * 80.0
                    forces_r[i] -= pen * 0.3
                pen = xs[i] + rs[i] - w
                if pen > 0:
                    forces_x[i] -= pen * 80.0
                    forces_r[i] -= pen * 0.3
                pen = rs[i] - ys[i]
                if pen > 0:
                    forces_y[i] += pen * 80.0
                    forces_r[i] -= pen * 0.3
                pen = ys[i] + rs[i] - h
                if pen > 0:
                    forces_y[i] -= pen * 80.0
                    forces_r[i] -= pen * 0.3

            forces_r += 0.01

            damping = 0.7
            for i in range(N):
                xs[i] += damping * forces_x[i] * 0.01 + temp * rng.randn() * 0.005
                ys[i] += damping * forces_y[i] * 0.01 + temp * rng.randn() * 0.005
                rs[i] += damping * forces_r[i] * 0.01 + temp * rng.randn() * 0.002

                xs[i] = max(rs[i] + 1e-9, min(w - rs[i] - 1e-9, xs[i]))
                ys[i] = max(rs[i] + 1e-9, min(h - rs[i] - 1e-9, ys[i]))
                rs[i] = max(1e-4, min(min(w, h) / 2.0, rs[i]))

        new_circles = np.column_stack([xs, ys, rs])
        return new_circles

    def _solve_lp_radii(centers, w, h):
        """Solve LP for optimal radii given fixed centers."""
        xs = centers[:, 0]
        ys = centers[:, 1]
        
        c_obj = -np.ones(N)
        
        A_ub = []
        b_ub = []
        
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                row = np.zeros(N)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(d)
            
        for i in range(N):
            row = np.zeros(N)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(xs[i])
            
            A_ub.append(row.copy())
            b_ub.append(w - xs[i])
            
            A_ub.append(row.copy())
            b_ub.append(ys[i])
            
            A_ub.append(row.copy())
            b_ub.append(h - ys[i])
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        bounds = [(0, None) for _ in range(N)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                radii = res.x
                if np.any(radii < 1e-9):
                    return None
                circles = np.column_stack([xs, ys, radii])
                if _strictly_valid(circles, w, h):
                    return circles
        except Exception:
            pass
        return None

    def _lp_refine(circles, w, h, n_perturb=50):
        """Use LP to find optimal radii for perturbed center configurations."""
        if time_left() < 10.0:
            return circles
        
        best_circles = circles
        best_val = _evaluate(circles, w, h)
        rng = np.random.RandomState(42)
        
        for _ in range(n_perturb):
            if time_left() < 3.0:
                break
            
            # Perturb centers slightly
            perturbed = circles.copy()
            perturbed[:, 0] += rng.uniform(-0.02, 0.02, N)
            perturbed[:, 1] += rng.uniform(-0.02, 0.02, N)
            
            # Clip to valid range
            for i in range(N):
                perturbed[i, 0] = max(0.01, min(w - 0.01, perturbed[i, 0]))
                perturbed[i, 1] = max(0.01, min(h - 0.01, perturbed[i, 1]))
            
            lp_result = _solve_lp_radii(perturbed, w, h)
            if lp_result is not None:
                val = _evaluate(lp_result, w, h)
                if val > best_val:
                    best_val = val
                    best_circles = lp_result
        
        return best_circles

    def _joint_optimize(circles, w, h, beta_start=50.0, max_iter=200):
        """Jointly optimize w and circle positions/radii."""
        if time_left() < 10.0:
            return circles, w, h

        x0 = np.concatenate([[w], circles.flatten()])
        r_max = min(w, h) / 2.0
        bounds = [(0.3, 1.7)]
        for i in range(N):
            bounds.append((0, w * 1.1))
            bounds.append((0, h * 1.1))
            bounds.append((1e-4, r_max))

        best_x = x0.copy()
        best_val = _evaluate(circles, w, h)

        for beta in [beta_start, beta_start * 10, beta_start * 100]:
            if time_left() < 5.0:
                break
            try:
                res = minimize(_objective_joint, x0, method='L-BFGS-B', bounds=bounds,
                               jac=True, args=(beta, 0.05),
                               options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
                if time_left() < 3.0:
                    break
                new_w = res.x[0]
                new_h = 2.0 - new_w
                new_circles = res.x[1:].reshape(-1, 3)
                if _strictly_valid(new_circles, new_w, new_h):
                    val = float(np.sum(new_circles[:, 2]))
                    if val > best_val:
                        best_val = val
                        best_x = res.x.copy()
                x0 = res.x
            except Exception:
                pass

        new_w = best_x[0]
        new_h = 2.0 - new_w
        new_circles = best_x[1:].reshape(-1, 3)
        return new_circles, new_w, new_h

    def _final_polish(circles, w, h):
        """High-precision final polish with very high beta."""
        if time_left() < 8.0:
            return circles

        x = circles.flatten()
        r_max = min(w, h) / 2.0
        bounds = []
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-4, r_max))

        best_x = x.copy()
        best_val = -np.inf
        if _strictly_valid(circles, w, h):
            best_val = float(np.sum(circles[:, 2]))

        for beta in [5000.0, 10000.0]:
            if time_left() < 4.0:
                break
            try:
                res = minimize(_objective_and_grad, x, method='L-BFGS-B', bounds=bounds,
                               jac=True, args=(w, h, beta, 0.01),
                               options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-13,
                                        'maxls': 100, 'maxfun': 100000})
                if time_left() < 2.0:
                    break
                new_circles = res.x.reshape(-1, 3)
                if _strictly_valid(new_circles, w, h):
                    val = float(np.sum(new_circles[:, 2]))
                    if val > best_val:
                        best_val = val
                        best_x = res.x.copy()
                x = res.x
            except Exception:
                pass

        return best_x.reshape(-1, 3)

    def _aspect_refine(circles, w, h):
        if time_left() < 15.0:
            return circles, w, h

        best_circles = circles
        best_val = _evaluate(circles, w, h)
        best_w, best_h = w, h

        for dw in [-0.05, -0.03, -0.02, 0.02, 0.03, 0.05]:
            if time_left() < 5.0:
                break
            new_w = w + dw
            new_h = 2.0 - new_w
            if new_w < 0.3 or new_h < 0.3:
                continue

            scaled = circles.copy()
            scaled[:, 0] *= new_w / w
            scaled[:, 1] *= new_h / h
            polished = _polish(scaled, new_w, new_h, beta_start=50.0, max_iter=20)
            val = _evaluate(polished, new_w, new_h)
            if val > best_val:
                best_val = val
                best_circles = polished
                best_w, best_h = new_w, new_h

        return best_circles, best_w, best_h

    # Main loop over aspect ratios
    widths = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 0.75, 1.25, 0.85, 1.15]

    for w in widths:
        if time_left() < 15.0:
            break
        h = 2.0 - w

        inits = []
        hex_circles = _hex_init(w, h, seed=42)
        inits.append(hex_circles)
        grid_circles = _grid_init(w, h, seed=42)
        inits.append(grid_circles)
        rand_circles = _random_init(w, h, seed=42)
        inits.append(rand_circles)
        if time_left() > 20.0:
            rand_circles2 = _random_init(w, h, seed=123)
            inits.append(rand_circles2)

        for init_circles in inits:
            if time_left() < 12.0:
                break

            polished = _polish(init_circles, w, h, beta_start=10.0, max_iter=30)
            check_and_update(polished, w, h)

            if time_left() > 10.0:
                phys = _physics_refine(polished, w, h, n_steps=1500, seed=42)
                check_and_update(phys, w, h)

            if time_left() > 12.0:
                if _evaluate(phys, w, h) > _evaluate(polished, w, h):
                    start_for_cg = phys
                else:
                    start_for_cg = polished
                cg_polished = _contact_graph_polish(start_for_cg, w, h)
                check_and_update(cg_polished, w, h)

            # LP refinement
            if time_left() > 10.0:
                lp_result = _lp_refine(cg_polished, w, h, n_perturb=30)
                check_and_update(lp_result, w, h)
                if _evaluate(lp_result, w, h) > _evaluate(cg_polished, w, h):
                    cg_polished = lp_result

            if time_left() > 15.0:
                joint_c, joint_w, joint_h = _joint_optimize(cg_polished, w, h, beta_start=50.0, max_iter=200)
                check_and_update(joint_c, joint_w, joint_h)
                if joint_w != w or joint_h != h:
                    cg_polished = joint_c
                    w, h = joint_w, joint_h

            if time_left() > 10.0:
                refined, new_w, new_h = _aspect_refine(cg_polished, w, h)
                if new_w != w or new_h != h:
                    check_and_update(refined, new_w, new_h)
                    cg_polished = refined
                    w, h = new_w, new_h

            if time_left() > 8.0:
                final = _final_polish(cg_polished, w, h)
                check_and_update(final, w, h)

    if best_result is None:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.1 - 1e-4
        best_result = np.zeros((N, 3))
        for i in range(a):
            for j in range(b):
                idx = i * b + j
                best_result[idx, 0] = (i + 0.5) * w / a
                best_result[idx, 1] = (j + 0.5) * h / b
                best_result[idx, 2] = r

    return best_result
# EVOLVE-BLOCK-END
