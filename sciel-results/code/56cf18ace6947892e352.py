import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 142  # safety margin before 150s limit

    def smooth_penalty(params, beta=50.0):
        """Smooth penalty using log-sum-exp for near-differentiability."""
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]

        # Boundary constraints: x - r >= 0, 1-x-r >= 0, etc.
        # Use smooth max(0, -violation) via log-sum-exp
        def smooth_max_neg(arr, beta=50.0):
            # smooth max(0, -a) for each element
            # = logsumexp(0, -a) / beta approximately
            # Use: (log(1 + exp(-beta * a))) / beta
            return np.logaddexp(0, -beta * arr) / beta

        d_left = xy[:, 0] - r
        d_right = 1.0 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1.0 - xy[:, 1] - r

        boundary_pen = np.sum(smooth_max_neg(d_left, beta)) + \
                       np.sum(smooth_max_neg(d_right, beta)) + \
                       np.sum(smooth_max_neg(d_bottom, beta)) + \
                       np.sum(smooth_max_neg(d_top, beta))

        # Overlap: dist_ij >= r_i + r_j
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        overlap_viol = min_dist[iu] - dist[iu]
        overlap_pen = np.sum(smooth_max_neg(-overlap_viol, beta))

        return -np.sum(r) + 100.0 * (boundary_pen + overlap_pen)

    def is_valid(result, tol=1e-9):
        n = N
        xy = result[:2*n].reshape(n, 2)
        r = result[2*n:]

        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1.0 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1.0 - xy[:, 1] - r < -tol):
            return False

        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False
        return True

    def get_score(result):
        return np.sum(result[2*N:])

    def push_apart(params, n_iters=150):
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()

        for _ in range(n_iters):
            moved = False
            for i in range(n):
                for j in range(i+1, n):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = r[i] + r[j]
                    if dist < min_dist and dist > 1e-12:
                        overlap = min_dist - dist
                        wi = r[i] / (r[i] + r[j]) if r[i] + r[j] > 1e-15 else 0.5
                        wj = 1.0 - wi
                        push_x = (dx / dist) * overlap
                        push_y = (dy / dist) * overlap
                        xy[i, 0] -= push_x * wi
                        xy[i, 1] -= push_y * wi
                        xy[j, 0] += push_x * wj
                        xy[j, 1] += push_y * wj
                        moved = True
                    elif dist <= 1e-12:
                        angle = np.random.uniform(0, 2*np.pi)
                        xy[i, 0] -= 0.01 * np.cos(angle)
                        xy[i, 1] -= 0.01 * np.sin(angle)
                        xy[j, 0] += 0.01 * np.cos(angle)
                        xy[j, 1] += 0.01 * np.sin(angle)
                        moved = True
            for i in range(n):
                if xy[i, 0] < r[i]:
                    xy[i, 0] = r[i]; moved = True
                if xy[i, 0] > 1.0 - r[i]:
                    xy[i, 0] = 1.0 - r[i]; moved = True
                if xy[i, 1] < r[i]:
                    xy[i, 1] = r[i]; moved = True
                if xy[i, 1] > 1.0 - r[i]:
                    xy[i, 1] = 1.0 - r[i]; moved = True
            if not moved:
                break
        return np.concatenate([xy.flatten(), r])

    def optimize_radii_fixed_positions(xy, r_init, max_iter=200):
        """Optimize radii given fixed positions - smaller problem."""
        n = N
        def obj(r):
            r = np.maximum(r, 1e-8)
            # For each pair, constraint: dist_ij >= r_i + r_j
            # For each circle, constraint: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
            # Use smooth penalty
            dx = xy[:, 0][:, None] - xy[:, 0][None, :]
            dy = xy[:, 1][:, None] - xy[:, 1][None, :]
            dist = np.sqrt(dx**2 + dy**2)
            iu = np.triu_indices(n, k=1)
            # Overlap violation
            overlap = (r[iu[0]] + r[iu[1]]) - dist[iu]
            overlap_pen = np.sum(np.maximum(0, overlap)**2)
            # Boundary
            max_r = np.minimum(np.minimum(xy[:, 0], 1.0 - xy[:, 0]),
                               np.minimum(xy[:, 1], 1.0 - xy[:, 1]))
            bound_pen = np.sum(np.maximum(0, r - max_r)**2)
            return -np.sum(r) + 1000.0 * (overlap_pen + bound_pen)

        bounds = [(0, 0.5)] * n
        result = minimize(obj, r_init, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': max_iter, 'ftol': 1e-14})
        return result.x

    def refine_full(params, max_iter=500, beta=50.0):
        """Full optimization of positions and radii."""
        try:
            result = minimize(lambda p: smooth_penalty(p, beta), params,
                            method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-14})
            return result.x
        except Exception:
            return params

    def get_initial_config(idx, rng):
        n = N
        mod = idx % 12

        if mod == 0:
            # 6x6 grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row, col = i // 6, i % 6
                xy[i] = [(col + 0.5)/6, (row + 0.5)/6]
                r[i] = 0.5/6 - 0.003
        elif mod == 1:
            # Perturbed 6x6
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row, col = i // 6, i % 6
                xy[i] = [(col + 0.5)/6, (row + 0.5)/6]
                r[i] = 0.5/6 - 0.003
            xy += rng.uniform(-0.015, 0.015, size=(n, 2))
        elif mod == 2:
            # Hexagonal packing
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row, col = i // 6, i % 6
                x = (col + 0.5)/6 + (0.25/6 if row % 2 == 1 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.073
        elif mod == 3:
            # 8x4 grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row, col = i // 8, i % 8
                xy[i] = [(col + 0.5)/8, (row + 0.5)/4]
                r[i] = 0.5/8 - 0.002
            xy += rng.uniform(-0.008, 0.008, size=(n, 2))
        elif mod == 4:
            # 4x8 grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row, col = i // 4, i % 4
                xy[i] = [(col + 0.5)/4, (row + 0.5)/8]
                r[i] = 0.5/4 - 0.002
            xy += rng.uniform(-0.008, 0.008, size=(n, 2))
        elif mod == 5:
            # Random uniform
            xy = rng.uniform(0.08, 0.92, size=(n, 2))
            r = rng.uniform(0.055, 0.085, size=n)
        elif mod == 6:
            # 7x5 grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row, col = i // 7, i % 7
                xy[i] = [(col + 0.5)/7, (row + 0.5)/5]
                r[i] = 0.5/7 - 0.002
            xy += rng.uniform(-0.008, 0.008, size=(n, 2))
        elif mod == 7:
            # 5x7 grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row, col = i // 5, i % 5
                xy[i] = [(col + 0.5)/5, (row + 0.5)/7]
                r[i] = 0.5/5 - 0.002
            xy += rng.uniform(-0.008, 0.008, size=(n, 2))
        elif mod == 8:
            # Hexagonal variant with offset
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row, col = i // 6, i % 6
                x = (col + 0.5)/6 + (0.2/6 if row % 2 == 0 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.07
        elif mod == 9:
            # Random medium
            xy = rng.uniform(0.1, 0.9, size=(n, 2))
            r = rng.uniform(0.06, 0.08, size=n)
        elif mod == 10:
            # Clustered 4 groups
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
            for i in range(n):
                group = i // 8
                cx, cy = centers[group]
                sub_i = i % 8
                sx = (sub_i % 3) / 3.0 + 0.5/3
                sy = (sub_i // 3) / 3.0 + 0.5/3
                xy[i] = [cx + (sx - 0.5)*0.35, cy + (sy - 0.5)*0.35]
                r[i] = 0.055
        else:
            # Random small
            xy = rng.uniform(0.06, 0.94, size=(n, 2))
            r = rng.uniform(0.04, 0.06, size=n)

        return np.concatenate([xy.flatten(), r])

    rng = np.random.RandomState(42)
    best_result = None
    best_score = -np.inf

    # Phase 1: Diverse restarts with position+radius optimization
    n_restarts = 80
    for restart in range(n_restarts):
        if time.time() > deadline - 25:
            break

        params = get_initial_config(restart, rng)
        params = push_apart(params, n_iters=60)

        # First: optimize radii with fixed positions (fast)
        xy = params[:2*N].reshape(N, 2).copy()
        r = params[2*N:].copy()
        r_opt = optimize_radii_fixed_positions(xy, r, max_iter=150)
        params_rad = np.concatenate([xy.flatten(), r_opt])

        if is_valid(params_rad):
            score = get_score(params_rad)
            if score > best_score:
                best_score = score
                best_result = params_rad.copy()

        # Then: full refinement
        refined = refine_full(params_rad, max_iter=300, beta=50.0)
        if is_valid(refined):
            score = get_score(refined)
            if score > best_score:
                best_score = score
                best_result = refined.copy()
        else:
            # Repair
            repaired = push_apart(refined, n_iters=80)
            if is_valid(repaired):
                score = get_score(repaired)
                if score > best_score:
                    best_score = score
                    best_result = repaired.copy()

    # Phase 2: Grow radii iteratively on best solution
    if best_result is not None and time.time() < deadline - 15:
        for grow_iter in range(30):
            if time.time() > deadline - 15:
                break
            xy = best_result[:2*N].reshape(N, 2).copy()
            r = best_result[2*N:].copy()
            r_new = r * 1.004
            params_trial = np.concatenate([xy.flatten(), r_new])
            refined = refine_full(params_trial, max_iter=200, beta=50.0)
            if is_valid(refined):
                score = get_score(refined)
                if score > best_score:
                    best_score = score
                    best_result = refined.copy()
            else:
                break

    # Phase 3: High-precision polish
    if best_result is not None and time.time() < deadline - 8:
        polished = refine_full(best_result, max_iter=800, beta=100.0)
        if is_valid(polished):
            score = get_score(polished)
            if score > best_score:
                best_score = score
                best_result = polished.copy()

        # Perturbation-based local search
        for polish_iter in range(20):
            if time.time() > deadline - 5:
                break
            xy_p = best_result[:2*N].reshape(N, 2).copy()
            r_p = best_result[2*N:].copy()
            scale = 0.002 / (1 + polish_iter * 0.3)
            xy_p += rng.uniform(-scale, scale, size=(N, 2))
            perturbed = np.concatenate([xy_p.flatten(), r_p])
            perturbed = push_apart(perturbed, n_iters=40)
            polished2 = refine_full(perturbed, max_iter=400, beta=100.0)
            if is_valid(polished2):
                score = get_score(polished2)
                if score > best_score:
                    best_score = score
                    best_result = polished2.copy()

    # Final cleanup
    if best_result is not None:
        xy = best_result[:2*N].reshape(N, 2).copy()
        r = best_result[2*N:].copy()
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1.0 - r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1.0 - r)
        final_params = np.concatenate([xy.flatten(), r])
        if is_valid(final_params):
            return np.column_stack([xy, r])

    # Fallback
    side = 6
    cells = []
    for i in range(N):
        row, col = i // 6, i % 6
        cells.append([(col + 0.5)/6, (row + 0.5)/6, 0.5/6 - 1e-3])
    return np.array(cells)
