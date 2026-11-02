"""Circle packing in unit square n=32."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin (150s limit)

    def compute_radii(x, y):
        """Compute the maximum radii for given centers."""
        dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_matrix, np.inf)
        min_dist = np.min(dist_matrix, axis=1)
        radii = np.minimum(dist_boundary, min_dist / 2.0)
        radii = np.maximum(radii, 0)
        return radii

    def objective(params):
        """Objective to minimize (negative sum of radii)."""
        x = params[0::2]
        y = params[1::2]
        radii = compute_radii(x, y)
        return -np.sum(radii)

    def smooth_objective(params, alpha=200.0):
        """Smooth objective using soft-min via log-sum-exp for gradient-based methods."""
        x = params[0::2]
        y = params[1::2]

        dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))

        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_matrix, 1e10)

        # Soft-min of pairwise distances
        scaled = -alpha * dist_matrix
        row_max = np.max(scaled, axis=1, keepdims=True)
        exp_scaled = np.exp(scaled - row_max)
        np.fill_diagonal(exp_scaled, 0.0)
        log_sum = np.log(np.sum(exp_scaled, axis=1)) + row_max[:, 0]
        min_dist_soft = -log_sum / alpha

        # Soft-min of boundary distances
        boundary_terms = np.stack([x, 1 - x, y, 1 - y], axis=1)
        scaled_b = -alpha * boundary_terms
        row_max_b = np.max(scaled_b, axis=1, keepdims=True)
        exp_b = np.exp(scaled_b - row_max_b)
        log_sum_b = np.log(np.sum(exp_b, axis=1)) + row_max_b[:, 0]
        min_boundary_soft = -log_sum_b / alpha

        # Soft-min of (boundary, dist/2)
        a = min_boundary_soft
        b = min_dist_soft / 2.0
        scaled_ab = -alpha * np.stack([a, b], axis=1)
        row_max_ab = np.max(scaled_ab, axis=1, keepdims=True)
        exp_ab = np.exp(scaled_ab - row_max_ab)
        log_sum_ab = np.log(np.sum(exp_ab, axis=1)) + row_max_ab[:, 0]
        radii_soft = -log_sum_ab / alpha

        radii_soft = np.maximum(radii_soft, 0)
        return -np.sum(radii_soft)

    def grid_init():
        side = int(np.ceil(np.sqrt(N)))  # 6
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                x0[idx] = (i + 0.5) / side
                y0[idx] = (j + 0.5) / side
                idx += 1
        if idx < N:
            for k in range(idx, N):
                x0[k] = 0.5
                y0[k] = 0.5
        params = np.zeros(2 * N)
        params[0::2] = x0
        params[1::2] = y0
        return params

    def hex_init():
        """Hexagonal close-packing initialization, tuned for n=32."""
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        idx = 0
        s = 1.0 / 5.5
        for row in range(6):
            for col in range(6):
                if idx >= N:
                    break
                x0[idx] = (col + 0.5) * s
                y0[idx] = (row + 0.5) * s * np.sqrt(3) / 2
                if row % 2 == 1:
                    x0[idx] += s / 2
                x0[idx] = np.clip(x0[idx], 0.02, 0.98)
                y0[idx] = np.clip(y0[idx], 0.02, 0.98)
                idx += 1
        params = np.zeros(2 * N)
        params[0::2] = x0
        params[1::2] = y0
        return params

    def hex_init_scaled():
        """Hexagonal lattice scaled to fit in unit square, taking N closest to center."""
        points = []
        for row in range(10):
            for col in range(10):
                x = col + (0.5 if row % 2 else 0)
                y = row * np.sqrt(3) / 2
                points.append((x, y))
        pts = np.array(points)
        center = pts.mean(axis=0)
        pts = pts - center
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:N]
        init_pts = pts[idx].copy()
        min_x, min_y = init_pts.min(axis=0)
        max_x, max_y = init_pts.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y
        scale = 0.9 / max(width, height)
        init_pts[:, 0] = (init_pts[:, 0] - min_x) * scale
        init_pts[:, 1] = (init_pts[:, 1] - min_y) * scale
        current_w = init_pts.max(axis=0) - init_pts.min(axis=0)
        offset_x = (1.0 - current_w[0]) / 2
        offset_y = (1.0 - current_w[1]) / 2
        init_pts[:, 0] += offset_x
        init_pts[:, 1] += offset_y
        params = np.zeros(2 * N)
        params[0::2] = init_pts[:, 0]
        params[1::2] = init_pts[:, 1]
        return params

    def random_init(seed):
        rng = np.random.RandomState(seed)
        params = np.zeros(2 * N)
        params[0::2] = rng.uniform(0.05, 0.95, N)
        params[1::2] = rng.uniform(0.05, 0.95, N)
        return params

    bounds = [(0.0, 1.0)] * (2 * N)

    # --- Phase 1: Differential Evolution global search ---
    np.random.seed(42)
    best_params = None
    best_obj = np.inf

    # Build initial population from diverse structured + random starts
    pop_size = 40
    pop = np.zeros((pop_size, 2 * N))

    # Seed population with structured inits
    pop[0] = grid_init()
    pop[1] = hex_init()
    pop[2] = hex_init_scaled()
    pop[3] = random_init(1)
    pop[4] = random_init(2)
    # Rotated hex
    x0, y0 = hex_init_scaled()[0::2], hex_init_scaled()[1::2]
    cx, cy = 0.5, 0.5
    angle = np.pi / 6
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    xr = (x0 - cx) * cos_a - (y0 - cy) * sin_a + cx
    yr = (x0 - cx) * sin_a + (y0 - cy) * cos_a + cy
    pop[5] = np.zeros(2 * N)
    pop[5][0::2] = np.clip(xr, 0.01, 0.99)
    pop[5][1::2] = np.clip(yr, 0.01, 0.99)
    # More random
    for i in range(6, pop_size):
        pop[i] = random_init(i * 7 + 13)

    # Evaluate initial population
    pop_objs = np.array([objective(p) for p in pop])
    best_idx = np.argmin(pop_objs)
    best_params = pop[best_idx].copy()
    best_obj = pop_objs[best_idx]

    # DE parameters
    F = 0.8  # mutation factor
    CR = 0.9  # crossover rate
    max_de_gens = 120
    rng = np.random.RandomState(42)

    for gen in range(max_de_gens):
        if time.time() > deadline - 55:
            break

        for i in range(pop_size):
            # Select 3 distinct random individuals
            while True:
                a, b, c = rng.choice(pop_size, 3, replace=False)
                if a != b and b != c and a != c and a != i and b != i and c != i:
                    break

            # Mutation: v = x_a + F * (x_b - x_c)
            v = pop[a] + F * (pop[b] - pop[c])
            v = np.clip(v, 0.0, 1.0)

            # Crossover: trial = v if rand < CR else x_i
            trial = np.empty_like(v)
            jrand = rng.randint(0, 2 * N)
            for j in range(2 * N):
                if rng.rand() < CR or j == jrand:
                    trial[j] = v[j]
                else:
                    trial[j] = pop[i][j]

            trial = np.clip(trial, 0.0, 1.0)
            trial_obj = objective(trial)

            if trial_obj <= pop_objs[i]:
                pop[i] = trial
                pop_objs[i] = trial_obj
                if trial_obj < best_obj:
                    best_obj = trial_obj
                    best_params = trial.copy()

        # Elitism: ensure best is kept
        if objective(best_params) < pop_objs.min():
            worst_idx = np.argmax(pop_objs)
            pop[worst_idx] = best_params
            pop_objs[worst_idx] = objective(best_params)

    # --- Phase 2: SLSQP local refinement from top DE candidates ---
    # Sort population by objective and take top 8 for local refinement
    sorted_idx = np.argsort(pop_objs)
    top_candidates = [pop[idx].copy() for idx in sorted_idx[:8]]

    for candidate in top_candidates:
        if time.time() > deadline - 40:
            break
        try:
            res = minimize(
                smooth_objective, candidate, method='SLSQP', bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-14}
            )
            true_obj = objective(res.x)
            if true_obj < best_obj:
                best_params = res.x.copy()
                best_obj = true_obj
        except Exception:
            pass

    # --- Phase 3: Powell refinement on true (non-smooth) objective ---
    if time.time() < deadline - 30:
        try:
            res = minimize(
                objective, best_params, method='Powell', bounds=bounds,
                options={'maxiter': 6000, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            if res.fun < best_obj:
                best_params = res.x.copy()
                best_obj = res.fun
        except Exception:
            pass

    # --- Phase 4: Nelder-Mead fine-tuning ---
    if time.time() < deadline - 18:
        for nm_restart in range(5):
            if time.time() > deadline - 8:
                break
            try:
                if nm_restart == 0:
                    p_start = best_params.copy()
                else:
                    p_start = best_params + np.random.normal(0, 0.001 * (nm_restart + 1), 2 * N)
                    p_start = np.clip(p_start, 0.001, 0.999)

                res = minimize(
                    objective, p_start, method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
                )
                if res.fun < best_obj:
                    best_params = res.x.copy()
                    best_obj = res.fun
            except Exception:
                pass

    # --- Phase 5: Final high-precision Powell polish ---
    if time.time() < deadline - 5:
        try:
            res = minimize(
                objective, best_params, method='Powell', bounds=bounds,
                options={'maxiter': 10000, 'xtol': 1e-14, 'ftol': 1e-14}
            )
            if res.fun < best_obj:
                best_params = res.x.copy()
                best_obj = res.fun
        except Exception:
            pass

    # Final construction
    x = best_params[0::2]
    y = best_params[1::2]
    radii = compute_radii(x, y)

    # Shrink radii slightly for floating point safety
    radii *= 0.999999

    result = np.column_stack((x, y, radii))
    return result

# EVOLVE-BLOCK-END
