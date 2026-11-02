"""Improved circle packing: spatial hashing, better init, adaptive SA, polish."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin (150s limit)

    best_value = -np.inf
    best_result = None
    best_wh = None

    def compute_overlap_penalty_vec(xs, ys, rs, beta=10.0):
        """Vectorized smooth overlap penalty."""
        n = len(rs)
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq)
                min_dist = rs[i] + rs[j]
                val = min_dist - dist
                if val > 1.0 / beta:
                    penalty += val
                else:
                    penalty += np.log1p(np.exp(beta * val)) / beta
        return penalty

    def spatial_hash_penalty(xs, ys, rs, beta=10.0, cell_size=None):
        """Spatial-hash accelerated overlap penalty for SA inner loop."""
        n = len(rs)
        if cell_size is None:
            max_r = np.max(rs)
            cell_size = max_r * 2.5
        # Build hash
        grid = {}
        for i in range(n):
            ci = int(xs[i] // cell_size)
            cj = int(ys[i] // cell_size)
            key = (ci, cj)
            if key not in grid:
                grid[key] = []
            grid[key].append(i)

        penalty = 0.0
        for i in range(n):
            ci = int(xs[i] // cell_size)
            cj = int(ys[i] // cell_size)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    key = (ci + di, cj + dj)
                    if key not in grid:
                        continue
                    for j in grid[key]:
                        if j <= i:
                            continue
                        dx = xs[i] - xs[j]
                        dy = ys[i] - ys[j]
                        dist_sq = dx * dx + dy * dy
                        dist = np.sqrt(dist_sq)
                        min_dist = rs[i] + rs[j]
                        val = min_dist - dist
                        if val > 1.0 / beta:
                            penalty += val
                        else:
                            penalty += np.log1p(np.exp(beta * val)) / beta
        return penalty

    def check_and_update(x, w, h):
        nonlocal best_value, best_result, best_wh
        if time.time() > deadline:
            return
        xs = x[0::3]
        ys = x[1::3]
        rs = x[2::3]
        if np.any(rs < 1e-9):
            return
        if np.any(xs - rs < -1e-6) or np.any(xs + rs > w + 1e-6):
            return
        if np.any(ys - rs < -1e-6) or np.any(ys + rs > h + 1e-6):
            return
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx * dx + dy * dy
                min_dist = rs[i] + rs[j]
                if dist_sq < (min_dist - 1e-7) ** 2:
                    return
        val = np.sum(rs)
        if val > best_value:
            best_value = val
            best_result = x.copy()
            best_wh = (w, h)

    def objective(params, w, h, beta=10.0):
        xs = params[0::3]
        ys = params[1::3]
        rs = params[2::3]
        if np.any(rs <= 0):
            return 1e10 + np.sum(np.minimum(rs, 0) ** 2)
        penalty = compute_overlap_penalty_vec(xs, ys, rs, beta)
        boundary_pen = 0.0
        for i in range(N):
            if xs[i] < rs[i]:
                boundary_pen += (rs[i] - xs[i]) ** 2
            if xs[i] > w - rs[i]:
                boundary_pen += (xs[i] - (w - rs[i])) ** 2
            if ys[i] < rs[i]:
                boundary_pen += (rs[i] - ys[i]) ** 2
            if ys[i] > h - rs[i]:
                boundary_pen += (ys[i] - (h - rs[i])) ** 2
        return penalty + boundary_pen - np.sum(rs)

    def make_hex_init(w, h):
        """Hexagonal close-packing initialization."""
        x0 = np.zeros(3 * N)
        # Estimate optimal radius for hex packing
        area = w * h
        # Hex packing density ~0.9069, 21 circles
        # Each circle area = pi*r^2, total = 21*pi*r^2
        # 21*pi*r^2 / area ≈ 0.9069 => r ≈ sqrt(0.9069*area/(21*pi))
        r_est = np.sqrt(0.9069 * area / (N * np.pi)) * 0.85  # safety factor

        # Hex grid: row spacing = r*sqrt(3), col spacing = 2*r
        # Adjust to fit
        cols = int(np.ceil(w / (2 * r_est)))
        rows = int(np.ceil(h / (r_est * np.sqrt(3))))
        while cols * rows < N:
            if cols < rows:
                cols += 1
            else:
                rows += 1

        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= N:
                    break
                x = (col + 0.5) * w / cols
                y = (row + 0.5) * h / rows
                # Offset odd rows for hex pattern
                if row % 2 == 1:
                    x += w / (2 * cols)
                x = np.clip(x, 0, w)
                y = np.clip(y, 0, h)
                x0[3 * idx] = x
                x0[3 * idx + 1] = y
                x0[3 * idx + 2] = r_est * 0.9
                idx += 1
        return x0

    def sa_optimize(w, h, x0, max_iters=3000, seed=42):
        """Spatial-hash accelerated simulated annealing."""
        xs = x0[0::3].copy()
        ys = x0[1::3].copy()
        rs = x0[2::3].copy()

        r_max = min(w, h) / 2.0
        T = 0.03
        cooling = 0.9995
        beta = 15.0

        # Initial cost
        max_r = np.max(rs)
        cell_size = max_r * 2.5
        current_cost = spatial_hash_penalty(xs, ys, rs, beta, cell_size) - np.sum(rs)

        rng = np.random.default_rng(seed)

        for it in range(max_iters):
            if time.time() > deadline:
                break

            idx = rng.integers(0, N)
            old_x, old_y, old_r = xs[idx], ys[idx], rs[idx]

            # Adaptive step size
            scale = T * 0.4
            xs[idx] += rng.normal(0, scale)
            ys[idx] += rng.normal(0, scale)
            rs[idx] += rng.normal(0, scale * 0.3)

            # Clamp
            xs[idx] = np.clip(xs[idx], rs[idx], w - rs[idx])
            ys[idx] = np.clip(ys[idx], rs[idx], h - rs[idx])
            rs[idx] = np.clip(rs[idx], 1e-4, r_max)

            # Recompute cell_size if needed
            if it % 200 == 0:
                max_r = np.max(rs)
                cell_size = max_r * 2.5

            new_cost = spatial_hash_penalty(xs, ys, rs, beta, cell_size) - np.sum(rs)
            delta = new_cost - current_cost

            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_cost = new_cost
            else:
                xs[idx], ys[idx], rs[idx] = old_x, old_y, old_r

            T *= cooling

        x_final = np.zeros(3 * N)
        x_final[0::3] = xs
        x_final[1::3] = ys
        x_final[2::3] = rs
        return x_final

    def polish(x, w, h, max_iter=800):
        """High-precision L-BFGS-B polish."""
        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-4, r_max))
        try:
            res = minimize(lambda p: objective(p, w, h, beta=40.0), x, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12})
            return res.x
        except Exception:
            return x

    # Aspect ratios to try
    widths = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50]

    for w in widths:
        if time.time() > deadline - 5:
            break
        h = 2.0 - w

        x0 = make_hex_init(w, h)

        # L-BFGS-B pass 1
        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-4, r_max))

        try:
            res = minimize(lambda p: objective(p, w, h, beta=8.0), x0, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-9})
            check_and_update(res.x, w, h)

            if time.time() < deadline - 15:
                res2 = minimize(lambda p: objective(p, w, h, beta=25.0), res.x, method='L-BFGS-B',
                                bounds=bounds, options={'maxiter': 400, 'ftol': 1e-11, 'gtol': 1e-11})
                check_and_update(res2.x, w, h)

                # SA refinement
                if time.time() < deadline - 10:
                    x_sa = sa_optimize(w, h, res2.x, max_iters=2500, seed=42 + int(w * 100))
                    check_and_update(x_sa, w, h)

                    # Second SA with perturbation
                    if time.time() < deadline - 8:
                        rng_p = np.random.default_rng(777)
                        x_pert = x_sa.copy()
                        for i in range(N):
                            x_pert[3 * i] += rng_p.normal(0, 0.015)
                            x_pert[3 * i + 1] += rng_p.normal(0, 0.015)
                            x_pert[3 * i + 2] += rng_p.normal(0, 0.008)
                        x_sa2 = sa_optimize(w, h, x_pert, max_iters=1500, seed=999)
                        check_and_update(x_sa2, w, h)
        except Exception:
            pass

    # Final polish on best result
    if best_result is not None and best_wh is not None and time.time() < deadline - 3:
        w, h = best_wh
        x_polished = polish(best_result, w, h, max_iter=600)
        check_and_update(x_polished, w, h)

    # Fallback
    if best_result is None:
        w, h = 1.4, 0.6
        a, b = 7, 3
        r = 0.1 - 1e-4
        best_result = np.zeros(3 * N)
        for i in range(N):
            col = i % a
            row = i // a
            best_result[3 * i] = (col + 0.5) * w / a
            best_result[3 * i + 1] = (row + 0.5) * h / b
            best_result[3 * i + 2] = r

    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = best_result[3 * i:3 * i + 3]
    return result
# EVOLVE-BLOCK-END
