import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    best_value = -np.inf
    best_result = None

    def check_time():
        return time.time() > deadline

    def check_and_update(x, w, h):
        nonlocal best_value, best_result
        if check_time():
            return
        # x is [x1,y1,r1, x2,y2,r2, ...]
        valid = True
        for i in range(N):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            if ri < 1e-9 or xi - ri < -1e-6 or xi + ri > w + 1e-6 or yi - ri < -1e-6 or yi + ri > h + 1e-6:
                valid = False
                break
        if not valid:
            return
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                dist_sq = (xi-xj)**2 + (yi-yj)**2
                min_dist = ri + rj
                if dist_sq < (min_dist - 1e-7)**2:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            val = np.sum(x[2::3])
            if val > best_value:
                best_value = val
                best_result = x.copy()

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
            r_var = np.mean((rs - r_mean) ** 2)
            obj += eq_weight * r_var
            grad[2::3] += eq_weight * 2.0 * (rs - r_mean) / N

        v1 = rs - xs
        obj += np.sum(_softplus(v1, beta))
        g1 = _softplus_grad(v1, beta)
        grad[0::3] -= g1
        grad[2::3] += g1

        v2 = xs + rs - w
        obj += np.sum(_softplus(v2, beta))
        g2 = _softplus_grad(v2, beta)
        grad[0::3] += g2
        grad[2::3] += g2

        v3 = rs - ys
        obj += np.sum(_softplus(v3, beta))
        g3 = _softplus_grad(v3, beta)
        grad[1::3] -= g3
        grad[2::3] += g3

        v4 = ys + rs - h
        obj += np.sum(_softplus(v4, beta))
        g4 = _softplus_grad(v4, beta)
        grad[1::3] += g4
        grad[2::3] += g4

        v5 = -rs
        obj += np.sum(_softplus(v5, beta))
        g5 = _softplus_grad(v5, beta)
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
            [7, 7, 7],
            [4, 5, 4, 4, 4],
            [3, 4, 4, 4, 4, 2],
            [5, 5, 5, 3, 3],
            [6, 6, 5, 4],
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
                    for j in range(i+1, N):
                        xi, yi, ri = circles[i]
                        xj, yj, rj = circles[j]
                        dist_sq = (xi-xj)**2 + (yi-yj)**2
                        min_dist = ri + rj
                        if dist_sq < (min_dist - 1e-7)**2:
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
        r_est = 0.08
        circles = []
        for _ in range(N):
            x = rng.uniform(r_est, w - r_est)
            y = rng.uniform(r_est, h - r_est)
            circles.append([x, y, r_est])
        return np.array(circles)

    def _softplus_optimize(x0, w, h, max_iters=200, beta_start=50.0, beta_end=500.0, eq_weight_start=0.1, eq_weight_end=0.01):
        """
        Multi-phase softplus optimization with increasing beta and decreasing equalization.
        Uses L-BFGS-B with analytical gradients.
        """
        x = x0.copy()
        n_phases = 4
        for phase in range(n_phases):
            if check_time():
                break
            t = phase / max(n_phases - 1, 1)
            beta = beta_start * (beta_end / beta_start) ** t
            eq_w = eq_weight_start * (eq_weight_end / eq_weight_start) ** t
            
            iters = max_iters // n_phases
            
            def obj_func(xv):
                return _objective_and_grad(xv, w, h, beta, eq_w)[0]
            
            def obj_grad(xv):
                return _objective_and_grad(xv, w, h, beta, eq_w)[1]
            
            # Bounds: r >= 0, x in [0, w], y in [0, h]
            bounds = []
            for i in range(N):
                bounds.append((0, w))  # x
                bounds.append((0, h))  # y
                bounds.append((1e-6, None))  # r
            
            try:
                res = minimize(obj_func, x, jac=obj_grad, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': iters, 'ftol': 1e-12, 'gtol': 1e-8})
                x = res.x
                check_and_update(x, w, h)
            except Exception:
                pass
        
        return x

    # Main search loop
    # Try different aspect ratios (w + h = 2)
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5]
    
    for ar in aspect_ratios:
        if check_time():
            break
        
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 - w
        
        # Try multiple initializations
        inits = []
        inits.append(('hex', _hex_init(w, h, seed=42)))
        inits.append(('grid', _grid_init(w, h, seed=42)))
        inits.append(('random', _random_init(w, h, seed=123)))
        inits.append(('random', _random_init(w, h, seed=456)))
        
        for init_type, x0 in inits:
            if check_time():
                break
            
            x_opt = _softplus_optimize(x0, w, h, max_iters=300)
            check_and_update(x_opt, w, h)
    
    # Final polish with SLSQP on the best result if time permits
    if best_result is not None and not check_time():
        # Try SLSQP refinement on the best found
        w_best = 1.0  # We need to track which w,h was best
        # Actually, let's just return the best result we found
        pass

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        w = 1.0
        h = 1.0
        r = 0.5 / side - 0.01
        for i in range(N):
            col = i % side
            row = i // side
            x = (col + 0.5) * w / side
            y = (row + 0.5) * h / side
            cells.append([x, y, r])
        return np.array(cells)
    
    # Convert best_result from [x1,y1,r1,...] to (N,3) array
    return best_result.reshape(N, 3)
