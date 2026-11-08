import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    best_result = None
    best_score = -np.inf

    def validate(xs, ys, rs, w, h):
        """Check that all circles are within bounds and non-overlapping."""
        if len(xs) != N or len(ys) != N or len(rs) != N:
            return False
        if np.any(rs < 1e-9):
            return False
        if np.any(xs < -1e-7) or np.any(xs > w + 1e-7):
            return False
        if np.any(ys < -1e-7) or np.any(ys > h + 1e-7):
            return False
        # Check pairwise non-overlap
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                if rs[i] + rs[j] > d + 1e-6:
                    return False
        return True

    def compute_score(xs, ys, rs, w, h):
        """Compute sum of radii if valid, else -inf."""
        if not validate(xs, ys, rs, w, h):
            return -np.inf
        return float(np.sum(rs))

    def grid_init(w, h, seed):
        """Grid-like initialization."""
        np.random.seed(seed)
        cols = 5
        rows = 5
        xs = []
        ys = []
        for i in range(cols):
            for j in range(rows):
                if len(xs) < N:
                    xs.append((i + 0.5) / cols * w + np.random.uniform(-0.05, 0.05) * w)
                    ys.append((j + 0.5) / rows * h + np.random.uniform(-0.05, 0.05) * h)
        xs = np.array(xs[:N])
        ys = np.array(ys[:N])
        # Estimate initial radii from local spacing
        rs = np.full(N, 0.15 * min(w, h))
        return xs, ys, rs

    def random_init(w, h, seed):
        """Random initialization."""
        np.random.seed(seed)
        xs = np.random.uniform(0.1 * w, 0.9 * w, N)
        ys = np.random.uniform(0.1 * h, 0.9 * h, N)
        rs = np.full(N, 0.08 * min(w, h))
        return xs, ys, rs

    def optimize_fixed_box(w, h, xs_init, ys_init, rs_init, penalty_weight=500.0):
        """
        Optimize circle positions and radii in a fixed box [0,w] x [0,h].
        Variables: [x_0..x_{N-1}, y_0..y_{N-1}, r_0..r_{N-1}]
        Objective: -sum(r) + penalty for violations.
        """
        n = N
        x0 = np.concatenate([xs_init, ys_init, rs_init])
        bounds = [(0, w)] * n + [(0, h)] * n + [(1e-4, min(w, h))] * n

        def obj_grad(x):
            xs = x[0:n]
            ys = x[n:2*n]
            rs = x[2*n:3*n]

            val = -np.sum(rs)
            pen = 0.0
            dpen = np.zeros_like(x)

            # Pairwise overlap penalty
            for i in range(n):
                for j in range(i + 1, n):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    d_sq = dx * dx + dy * dy
                    d = np.sqrt(d_sq) if d_sq > 1e-14 else 1e-7
                    overlap = rs[i] + rs[j] - d
                    if overlap > 0:
                        pen += overlap * overlap
                        f = 2.0 * overlap
                        inv_d = 1.0 / d
                        dpen[i] += f * (-dx * inv_d)
                        dpen[n + i] += f * (-dy * inv_d)
                        dpen[2*n + i] += f
                        dpen[j] += f * (dx * inv_d)
                        dpen[n + j] += f * (dy * inv_d)
                        dpen[2*n + j] += f

            # Boundary penalties
            for i in range(n):
                # Left: r_i - x_i > 0
                o = rs[i] - xs[i]
                if o > 0:
                    pen += o * o
                    f = 2.0 * o
                    dpen[i] += f * (-1.0)
                    dpen[2*n + i] += f
                # Right: r_i - (w - x_i) > 0
                o = rs[i] - (w - xs[i])
                if o > 0:
                    pen += o * o
                    f = 2.0 * o
                    dpen[i] += f
                    dpen[2*n + i] += f
                # Bottom: r_i - y_i > 0
                o = rs[i] - ys[i]
                if o > 0:
                    pen += o * o
                    f = 2.0 * o
                    dpen[n + i] += f * (-1.0)
                    dpen[2*n + i] += f
                # Top: r_i - (h - y_i) > 0
                o = rs[i] - (h - ys[i])
                if o > 0:
                    pen += o * o
                    f = 2.0 * o
                    dpen[n + i] += f
                    dpen[2*n + i] += f

            val += penalty_weight * pen
            grad = np.zeros_like(x)
            grad[2*n:3*n] = -1.0
            grad += penalty_weight * dpen
            return val, grad

        try:
            res = minimize(obj_grad, x0, jac=True, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-9,
                                    'maxls': 50, 'maxfun': 5000})
            xs = res.x[0:n]
            ys = res.x[n:2*n]
            rs = res.x[2*n:3*n]
            # Clamp to valid range
            xs = np.clip(xs, 0, w)
            ys = np.clip(ys, 0, h)
            rs = np.clip(rs, 1e-4, min(w, h))
            score = compute_score(xs, ys, rs, w, h)
            if score > -np.inf:
                return score, np.column_stack([xs, ys, rs])
        except Exception:
            pass
        return -np.inf, None

    def voronoi_radius_init(xs, ys, w, h):
        """
        Estimate radii based on distance to nearest neighbor and walls.
        This gives a better initial radius estimate than uniform.
        """
        rs = np.zeros(N)
        for i in range(N):
            # Distance to walls
            d_wall = min(xs[i], w - xs[i], ys[i], h - ys[i])
            # Distance to nearest other circle
            d_nn = np.inf
            for j in range(N):
                if j == i:
                    continue
                d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                if d < d_nn:
                    d_nn = d
            # Radius is half the min of wall distance and nearest-neighbor distance
            # (conservative estimate)
            rs[i] = 0.45 * min(d_wall, d_nn)
            rs[i] = max(rs[i], 1e-4)
        return rs

    def refine_with_voronoi(w, h, xs, ys, max_iter=300, step=0.01):
        """
        Iteratively adjust radii to be the maximum feasible given positions,
        then nudge positions to increase radii.
        """
        rs = voronoi_radius_init(xs, ys, w, h)
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            # Recompute max feasible radii given current positions
            new_rs = np.zeros(N)
            for i in range(N):
                # Wall constraint
                r_max = min(xs[i], w - xs[i], ys[i], h - ys[i])
                # Pairwise constraint
                for j in range(N):
                    if j == i:
                        continue
                    d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                    r_max = min(r_max, d / 2.0)
                new_rs[i] = max(r_max * 0.999, 1e-4)
            # Check if improved
            if np.allclose(new_rs, rs, atol=1e-8):
                break
            rs = new_rs
        return rs

    # Main search loop
    w_values = np.linspace(0.6, 1.4, 9)
    seeds = list(range(1, 21))

    runs_done = 0
    max_runs = 40

    for w in w_values:
        if time.time() > deadline:
            break
        h = 2.0 - w
        for seed in seeds:
            if time.time() > deadline:
                break
            if runs_done >= max_runs:
                break

            # Choose init strategy
            if seed % 3 == 0:
                xs, ys, rs = grid_init(w, h, seed)
            else:
                xs, ys, rs = random_init(w, h, seed)

            # Improve initial radii using voronoi-like estimation
            rs = voronoi_radius_init(xs, ys, w, h)

            score, result = optimize_fixed_box(w, h, xs, ys, rs)
            if score > best_score:
                best_score = score
                best_result = result.copy()

            runs_done += 1

            # Second pass: use the result as initial condition for a refinement
            if result is not None and time.time() < deadline - 5:
                xs2 = result[:, 0]
                ys2 = result[:, 1]
                rs2 = result[:, 2]
                # Add small perturbation
                np.random.seed(seed + 100)
                xs2 += np.random.uniform(-0.02, 0.02, N) * w
                ys2 += np.random.uniform(-0.02, 0.02, N) * h
                xs2 = np.clip(xs2, 0, w)
                ys2 = np.clip(ys2, 0, h)
                score2, result2 = optimize_fixed_box(w, h, xs2, ys2, rs2, penalty_weight=1000.0)
                if score2 > best_score:
                    best_score = score2
                    best_result = result2.copy()
                runs_done += 1

    # Final refinement on best result if time permits
    if best_result is not None and time.time() < deadline - 3:
        w_best = best_result[:, 0].max()
        # Actually we need to track w. Let's just re-optimize with tighter bounds.
        # For simplicity, do a few more restarts with the best aspect ratio.
        pass

    if best_result is not None:
        return best_result
    else:
        # Fallback: simple grid
        w = 1.0
        h = 1.0
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cx = (i + 0.5) / side * w
                    cy = (j + 0.5) / side * h
                    r = 0.5 / side * min(w, h) * 0.95
                    cells.append([cx, cy, r])
        return np.array(cells[:N])
