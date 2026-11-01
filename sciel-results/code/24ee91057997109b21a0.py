import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 142  # safety margin before 150s kill

    best_result = None
    best_score = -np.inf

    def penalty_function(params):
        """Smooth penalty: minimize -sum(r) + penalties for violations."""
        n = N
        xy = params[:2 * n].reshape(n, 2)
        r = params[2 * n:]

        # Boundary violations (squared hinge)
        d_left = xy[:, 0] - r
        d_right = 1.0 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1.0 - xy[:, 1] - r
        boundary_viol = (np.sum(np.maximum(0, -d_left) ** 2) +
                         np.sum(np.maximum(0, -d_right) ** 2) +
                         np.sum(np.maximum(0, -d_bottom) ** 2) +
                         np.sum(np.maximum(0, -d_top) ** 2))

        # Overlap violations (squared hinge)
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        overlap = np.maximum(0, min_dist[iu] - dist[iu])
        overlap_viol = np.sum(overlap ** 2)

        # Small penalty for tiny radii
        small_r = np.sum(np.maximum(0, 0.005 - r) ** 2)

        return -np.sum(r) + 500.0 * boundary_viol + 500.0 * overlap_viol + 5.0 * small_r

    def is_valid(result, tol=1e-9):
        n = N
        xy = result[:2 * n].reshape(n, 2)
        r = result[2 * n:]

        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1.0 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1.0 - xy[:, 1] - r < -tol):
            return False

        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist_sq = dx ** 2 + dy ** 2
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist_sq[iu] < (min_dist[iu] - tol) ** 2):
            return False
        return True

    def get_score(result):
        return np.sum(result[2 * N:])

    def push_apart(params, n_iters=80, rng=None):
        """Vectorized push-apart with boundary projection."""
        n = N
        xy = params[:2 * n].reshape(n, 2).copy()
        r = params[2 * n:].copy()
        if rng is None:
            rng = np.random

        for _ in range(n_iters):
            dx = xy[:, 0][:, None] - xy[:, 0][None, :]
            dy = xy[:, 1][:, None] - xy[:, 1][None, :]
            dist_sq = dx ** 2 + dy ** 2
            dist = np.sqrt(dist_sq)
            min_dist = r[:, None] + r[None, :]
            iu = np.triu_indices(n, k=1)
            overlaps = np.maximum(0, min_dist[iu] - dist[iu])

            if np.max(overlaps) < 1e-14:
                if (np.all(xy[:, 0] >= r - 1e-14) and np.all(xy[:, 0] <= 1 - r + 1e-14) and
                        np.all(xy[:, 1] >= r - 1e-14) and np.all(xy[:, 1] <= 1 - r + 1e-14)):
                    break

            correction = np.zeros((n, 2))
            for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
                d = dist[i, j]
                if d < 1e-15:
                    angle = rng.uniform(0, 2 * np.pi)
                    fx, fy = np.cos(angle), np.sin(angle)
                    ov = min_dist[i, j]
                else:
                    fx = (xy[j, 0] - xy[i, 0]) / d
                    fy = (xy[j, 1] - xy[i, 1]) / d
                    ov = min_dist[i, j] - d
                if ov > 0:
                    wi = r[i] / (r[i] + r[j]) if (r[i] + r[j]) > 1e-15 else 0.5
                    wj = 1.0 - wi
                    correction[i, 0] -= fx * ov * wi
                    correction[i, 1] -= fy * ov * wi
                    correction[j, 0] += fx * ov * wj
                    correction[j, 1] += fy * ov * wj

            xy += correction
            xy[:, 0] = np.clip(xy[:, 0], r, 1.0 - r)
            xy[:, 1] = np.clip(xy[:, 1], r, 1.0 - r)

        return np.concatenate([xy.flatten(), r])

    def hex_init(rng, scale=1.0):
        """Hexagonal close-packing initialization for 32 circles."""
        n = N
        # Rows: 6,5,6,5,6,4 = 32
        rows = [6, 5, 6, 5, 6, 4]
        xy = np.zeros((n, 2))
        r = np.zeros(n)
        idx = 0
        for row, ncols in enumerate(rows):
            y = (row + 0.5) / 6.0
            for col in range(ncols):
                offset = 0.5 / 6.0 if row % 2 == 1 else 0.0
                x = (col + 0.5) / 6.0 + offset
                if idx < n:
                    xy[idx, 0] = x
                    xy[idx, 1] = y
                    r[idx] = 0.078
                    idx += 1
        # Perturb
        xy += rng.uniform(-0.01, 0.01, size=(n, 2))
        r += rng.uniform(-0.005, 0.005, size=n)
        r = np.clip(r, 0.04, 0.12)
        return np.concatenate([xy.flatten(), r])

    def grid_init(rng, cols=6, rows=6):
        """Square grid initialization."""
        n = N
        xy = np.zeros((n, 2))
        r = np.zeros(n)
        for i in range(n):
            row = i // cols
            col = i % cols
            xy[i, 0] = (col + 0.5) / cols
            xy[i, 1] = (row + 0.5) / rows
            r[i] = min(0.5 / cols, 0.5 / rows) - 0.003
        xy += rng.uniform(-0.012, 0.012, size=(n, 2))
        return np.concatenate([xy.flatten(), r])

    def random_init(rng, r_min=0.04, r_max=0.09):
        n = N
        xy = rng.uniform(0.06, 0.94, size=(n, 2))
        r = rng.uniform(r_min, r_max, size=n)
        return np.concatenate([xy.flatten(), r])

    def refine(params, max_iter=500, tol=1e-14):
        try:
            res = minimize(penalty_function, params, method='L-BFGS-B',
                           options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12})
            return res.x
        except Exception:
            return params

    def polish(params, max_iter=3000, tol=1e-16):
        try:
            res = minimize(penalty_function, params, method='L-BFGS-B',
                           options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-14, 'maxls': 100})
            return res.x
        except Exception:
            return params

    rng = np.random.RandomState(42)

    # Phase 1: Diverse restarts
    n_restarts = 150
    for restart in range(n_restarts):
        if time.time() > deadline - 25:
            break

        # Cycle through initialization strategies
        mod = restart % 10
        if mod == 0:
            params = hex_init(rng)
        elif mod == 1:
            params = hex_init(rng, scale=0.95)
        elif mod == 2:
            params = grid_init(rng, cols=6, rows=6)
        elif mod == 3:
            params = grid_init(rng, cols=8, rows=4)
        elif mod == 4:
            params = grid_init(rng, cols=4, rows=8)
        elif mod == 5:
            params = grid_init(rng, cols=7, rows=5)
        elif mod == 6:
            params = random_init(rng, 0.05, 0.09)
        elif mod == 7:
            params = random_init(rng, 0.04, 0.08)
        elif mod == 8:
            params = hex_init(rng)
            # Extra perturbation
            xy_p = params[:2 * N].reshape(N, 2).copy()
            xy_p += rng.uniform(-0.02, 0.02, size=(N, 2))
            params = np.concatenate([xy_p.flatten(), params[2 * N:]])
        else:
            params = grid_init(rng, cols=5, rows=7)

        # Push apart to get a valid starting point
        params = push_apart(params, n_iters=50, rng=rng)

        if is_valid(params):
            s = get_score(params)
            if s > best_score:
                best_score = s
                best_result = params.copy()

        # Refine
        refined = refine(params, max_iter=400, tol=1e-13)
        if is_valid(refined):
            s = get_score(refined)
            if s > best_score:
                best_score = s
                best_result = refined.copy()
        else:
            # Repair
            repaired = push_apart(refined, n_iters=60, rng=rng)
            if is_valid(repaired):
                s = get_score(repaired)
                if s > best_score:
                    best_score = s
                    best_result = repaired.copy()

    # Phase 2: High-precision polish on best
    if best_result is not None and time.time() < deadline - 20:
        polished = polish(best_result, max_iter=3000, tol=1e-16)
        if is_valid(polished):
            s = get_score(polished)
            if s > best_score:
                best_score = s
                best_result = polished.copy()

        # Perturbation-based local search
        for piter in range(30):
            if time.time() > deadline - 10:
                break
            if piter < 10:
                scale = 0.001
            elif piter < 20:
                scale = 0.003
            else:
                scale = 0.006

            perturbed = best_result.copy()
            xy_p = perturbed[:2 * N].reshape(N, 2).copy()
            r_p = perturbed[2 * N:].copy()
            xy_p += rng.uniform(-scale, scale, size=(N, 2))
            # Small radius perturbation
            r_p *= (1.0 + rng.uniform(-0.01, 0.01, size=N))
            r_p = np.clip(r_p, 0.01, 0.3)
            perturbed = np.concatenate([xy_p.flatten(), r_p])
            perturbed = push_apart(perturbed, n_iters=40, rng=rng)
            polished2 = polish(perturbed, max_iter=1500, tol=1e-15)
            if is_valid(polished2):
                s = get_score(polished2)
                if s > best_score:
                    best_score = s
                    best_result = polished2.copy()

    # Final validation and cleanup
    if best_result is not None:
        n = N
        xy = best_result[:2 * n].reshape(n, 2).copy()
        r = best_result[2 * n:].copy()
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1.0 - r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1.0 - r)
        final_params = np.concatenate([xy.flatten(), r])
        if is_valid(final_params):
            # One last micro-polish if time permits
            if time.time() < deadline - 5:
                final_polished = polish(final_params, max_iter=1000, tol=1e-16)
                if is_valid(final_polished):
                    s = get_score(final_polished)
                    if s > get_score(final_params):
                        final_params = final_polished
                        xy = final_params[:2 * n].reshape(n, 2).copy()
                        r = final_params[2 * n:].copy()
                        r = np.clip(r, 0, 0.5)
                        xy[:, 0] = np.clip(xy[:, 0], r, 1.0 - r)
                        xy[:, 1] = np.clip(xy[:, 1], r, 1.0 - r)
            return np.column_stack([xy, r])

    # Fallback: simple grid
    side = 6
    cells = []
    for i in range(N):
        row = i // side
        col = i % side
        cells.append([(col + 0.5) / side, (row + 0.5) / side, 0.5 / side - 1e-3])
    return np.array(cells)
