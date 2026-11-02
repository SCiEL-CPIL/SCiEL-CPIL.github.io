"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.spatial import Delaunay
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    best_score = -1.0
    best_result = None

    def compute_rect(aspect):
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        return w, h

    def evaluate(circles, aspect):
        """Return (sum_radii, is_valid). circles: (N,3) array [x, y, r]."""
        w, h = compute_rect(aspect)
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]

        # Check bounds
        if np.any(xs < rs - 1e-9) or np.any(xs > w - rs + 1e-9) or \
           np.any(ys < rs - 1e-9) or np.any(ys > h - rs + 1e-9):
            return 0.0, False

        # Check pairwise distances
        n = len(circles)
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < rs[i] + rs[j] - 1e-9:
                    return 0.0, False

        return float(np.sum(rs)), True

    def fast_valid(circles, aspect):
        """Fast validity check for polish phase."""
        w, h = compute_rect(aspect)
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        if np.any(xs < rs - 1e-9) or np.any(xs > w - rs + 1e-9) or \
           np.any(ys < rs - 1e-9) or np.any(ys > h - rs + 1e-9):
            return False
        n = len(circles)
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                if dx * dx + dy * dy < (rs[i] + rs[j] - 1e-9) ** 2:
                    return False
        return True

    def delaunay_neighbors(circles):
        """Get Delaunay neighbors for constraint reduction."""
        pts = circles[:, :2]
        try:
            tri = Delaunay(pts)
            neighbors = [set() for _ in range(N)]
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = simplex[i], simplex[j]
                        neighbors[a].add(b)
                        neighbors[b].add(a)
            return neighbors
        except Exception:
            return None

    def grid_init(aspect):
        """Create initial grid packing."""
        w, h = compute_rect(aspect)
        side = int(np.ceil(np.sqrt(N)))
        circles = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            circles[i, 0] = (col + 0.5) / side * w
            circles[i, 1] = (row + 0.5) / side * h
            r_max = min(w, h) / (2 * side)
            circles[i, 2] = r_max * 0.95
        return circles

    def hex_init(aspect):
        """Create hexagonal packing initialization."""
        w, h = compute_rect(aspect)
        # Try to fit hexagonal pattern
        circles = np.zeros((N, 3))
        # Estimate radius from area
        area = w * h
        r_est = np.sqrt(area / (N * np.pi * 1.2))
        
        dx = 2 * r_est * 1.01
        dy = 2 * r_est * np.sqrt(3) / 2 * 1.01
        
        idx = 0
        row = 0
        while idx < N:
            col = 0
            offset = (dx / 2) if row % 2 == 1 else 0
            while idx < N:
                x = offset + col * dx + r_est
                y = row * dy + r_est
                if x + r_est <= w and y + r_est <= h:
                    circles[idx, 0] = x
                    circles[idx, 1] = y
                    circles[idx, 2] = r_est
                    idx += 1
                else:
                    break
                col += 1
            row += 1
        return circles

    def random_init(aspect):
        """Random initialization."""
        w, h = compute_rect(aspect)
        circles = np.zeros((N, 3))
        for i in range(N):
            r = 0.05
            circles[i, 0] = r + np.random.random() * (w - 2 * r)
            circles[i, 1] = r + np.random.random() * (h - 2 * r)
            circles[i, 2] = r
        return circles

    def optimize_phase(circles, aspect, max_iter=50000, T_init=0.03, T_min=1e-7, cooling=0.9998):
        """Simulated annealing optimization."""
        cur_circles = circles.copy()
        cur_aspect = aspect
        cur_score, valid = evaluate(cur_circles, cur_aspect)
        if not valid:
            return circles, aspect, 0.0

        T = T_init
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if T < T_min:
                break

            # Perturb
            new_circles = cur_circles.copy()
            new_aspect = cur_aspect
            pert_type = random.random()

            if pert_type < 0.45:
                idx = random.randint(0, N - 1)
                new_circles[idx, 0] += random.gauss(0, 0.015)
                new_circles[idx, 1] += random.gauss(0, 0.015)
            elif pert_type < 0.75:
                idx = random.randint(0, N - 1)
                new_circles[idx, 2] = max(0.01, new_circles[idx, 2] + random.gauss(0, 0.008))
            elif pert_type < 0.9:
                new_aspect *= (1.0 + random.gauss(0, 0.04))
                new_aspect = max(0.2, min(5.0, new_aspect))
                old_w, old_h = compute_rect(cur_aspect)
                new_w, new_h = compute_rect(new_aspect)
                new_circles[:, 0] *= new_w / old_w
                new_circles[:, 1] *= new_h / old_h
            else:
                # Move two circles
                idx1, idx2 = random.sample(range(N), 2)
                new_circles[idx1, 0] += random.gauss(0, 0.01)
                new_circles[idx1, 1] += random.gauss(0, 0.01)
                new_circles[idx2, 0] += random.gauss(0, 0.01)
                new_circles[idx2, 1] += random.gauss(0, 0.01)

            new_score, valid = evaluate(new_circles, new_aspect)
            delta = new_score - cur_score

            if delta > 0 or random.random() < np.exp(delta / T):
                cur_circles = new_circles
                cur_aspect = new_aspect
                cur_score = new_score

            T *= cooling

        return cur_circles, cur_aspect, cur_score

    def polish(circles, aspect, iterations=200):
        """Polish by growing radii greedily."""
        cur_circles = circles.copy()
        cur_aspect = aspect
        w, h = compute_rect(cur_aspect)

        for _ in range(iterations):
            if time.time() > deadline:
                break
            improved = False
            # Get Delaunay neighbors for constraint reduction
            neighbors = delaunay_neighbors(cur_circles)

            for i in range(N):
                old_r = cur_circles[i, 2]
                # Binary search for max radius
                lo, hi = old_r, min(w, h) * 0.5
                for _ in range(20):
                    mid = (lo + hi) / 2
                    cur_circles[i, 2] = mid
                    if fast_valid(cur_circles, cur_aspect):
                        lo = mid
                    else:
                        hi = mid
                cur_circles[i, 2] = lo
                if lo > old_r + 1e-8:
                    improved = True

            if not improved:
                break

        return cur_circles, cur_aspect

    def ultra_polish(circles, aspect, iterations=100):
        """Ultra-polish with tighter tolerance."""
        cur_circles = circles.copy()
        cur_aspect = aspect
        w, h = compute_rect(cur_aspect)

        for _ in range(iterations):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                old_r = cur_circles[i, 2]
                # Try small increments
                for dr in [0.001, 0.0005, 0.0001, 0.00005, 0.00001]:
                    cur_circles[i, 2] = old_r + dr
                    if fast_valid(cur_circles, cur_aspect):
                        old_r = cur_circles[i, 2]
                        improved = True
                    else:
                        cur_circles[i, 2] = old_r
                        break
            if not improved:
                break

        return cur_circles, cur_aspect

    # Main optimization loop with multiple restarts
    np.random.seed(42)
    random.seed(42)

    # Try different initializations and aspect ratios
    inits = [
        (grid_init(1.0), 1.0),
        (grid_init(1.2), 1.2),
        (grid_init(0.8), 0.8),
        (hex_init(1.0), 1.0),
        (hex_init(1.3), 1.3),
        (random_init(1.0), 1.0),
        (random_init(1.5), 1.5),
    ]

    for init_circles, init_aspect in inits:
        if time.time() > deadline:
            break

        # Phase 1: SA optimization
        opt_circles, opt_aspect, opt_score = optimize_phase(
            init_circles, init_aspect, max_iter=30000, T_init=0.03
        )

        if opt_score > best_score:
            best_score = opt_score
            best_result = (opt_circles.copy(), opt_aspect)

        # Phase 2: Polish
        pol_circles, pol_aspect = polish(opt_circles, opt_aspect, iterations=100)
        pol_score, pol_valid = evaluate(pol_circles, pol_aspect)
        if pol_valid and pol_score > best_score:
            best_score = pol_score
            best_result = (pol_circles.copy(), pol_aspect)

        # Phase 3: Ultra-polish
        up_circles, up_aspect = ultra_polish(pol_circles, pol_aspect, iterations=50)
        up_score, up_valid = evaluate(up_circles, up_aspect)
        if up_valid and up_score > best_score:
            best_score = up_score
            best_result = (up_circles.copy(), up_aspect)

    # Final: try scipy minimize on the best result
    if best_result is not None and time.time() < deadline - 10:
        final_circles, final_aspect = best_result[0].copy(), best_result[1]

        def neg_sum_radii(params):
            """Negative sum of radii (to minimize)."""
            n = len(params) // 3
            circles = params[:n * 3].reshape(n, 3)
            aspect = params[n * 3]
            w, h = compute_rect(aspect)
            xs = circles[:, 0]
            ys = circles[:, 1]
            rs = circles[:, 2]

            # Penalty for invalid
            penalty = 0.0
            if np.any(xs < rs - 1e-9) or np.any(xs > w - rs + 1e-9) or \
               np.any(ys < rs - 1e-9) or np.any(ys > h - rs + 1e-9):
                penalty += 10.0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < rs[i] + rs[j] - 1e-9:
                        penalty += 10.0
            return -(np.sum(rs)) + penalty

        x0 = np.concatenate([final_circles.flatten(), [final_aspect]])
        try:
            res = minimize(neg_sum_radii, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < -best_score + 1e-6:
                new_circles = res.x[:N * 3].reshape(N, 3)
                new_aspect = res.x[N * 3]
                new_score, new_valid = evaluate(new_circles, new_aspect)
                if new_valid and new_score > best_score:
                    best_score = new_score
                    best_result = (new_circles.copy(), new_aspect)
        except Exception:
            pass

    if best_result is None:
        # Fallback
        circles, aspect = grid_init(1.0)
        return circles

    circles, aspect = best_result
    w, h = compute_rect(aspect)

    # Clamp radii to ensure validity
    for i in range(N):
        r = min(circles[i, 2], circles[i, 0] - 1e-6, w - circles[i, 0] - 1e-6,
                circles[i, 1] - 1e-6, h - circles[i, 1] - 1e-6)
        r = max(r, 0.01)
        circles[i, 2] = r

    return circles
# EVOLVE-BLOCK-END
