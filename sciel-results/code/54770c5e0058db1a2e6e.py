import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    best_result = None
    best_score = -np.inf

    def is_valid(result, tol=1e-9):
        n = N
        xy = result[:2*n].reshape(n, 2)
        r = result[2*n:]

        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
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
        n = N
        r = result[2*n:]
        return np.sum(r)

    def push_apart_vectorized(params, n_iters=200):
        """Vectorized push-apart for speed."""
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()
        
        for _ in range(n_iters):
            moved = False
            # Compute all pairwise distances
            dx = xy[:, 0][:, None] - xy[:, 0][None, :]
            dy = xy[:, 1][:, None] - xy[:, 1][None, :]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = r[:, None] + r[None, :]
            
            # Find overlapping pairs
            overlap = np.maximum(0, min_dist - dist)
            iu = np.triu_indices(n, k=1)
            overlap_pairs = overlap[iu]
            
            if np.any(overlap_pairs > 1e-12):
                moved = True
                # Process overlapping pairs
                idx_i, idx_j = iu
                for k in range(len(idx_i)):
                    i, j = idx_i[k], idx_j[k]
                    d = dist[i, j]
                    if d < 1e-12:
                        angle = np.random.uniform(0, 2*np.pi)
                        xy[i, 0] -= 0.005 * np.cos(angle)
                        xy[i, 1] -= 0.005 * np.sin(angle)
                        xy[j, 0] += 0.005 * np.cos(angle)
                        xy[j, 1] += 0.005 * np.sin(angle)
                        continue
                    ov = overlap[i, j]
                    wi = r[i] / (r[i] + r[j]) if r[i] + r[j] > 1e-15 else 0.5
                    wj = 1 - wi
                    push_x = (dx[i, j] / d) * ov
                    push_y = (dy[i, j] / d) * ov
                    xy[i, 0] -= push_x * wi
                    xy[i, 1] -= push_y * wi
                    xy[j, 0] += push_x * wj
                    xy[j, 1] += push_y * wj
            
            # Clamp to boundaries
            xy[:, 0] = np.clip(xy[:, 0], r, 1 - r)
            xy[:, 1] = np.clip(xy[:, 1], r, 1 - r)
            
            if not moved:
                break

        return np.concatenate([xy.flatten(), r])

    def optimize_positions_fixed_radii(xy_init, r, max_iter=200):
        """Optimize positions for fixed radii using L-BFGS-B with penalty."""
        n = N
        def pos_obj(p):
            xy_p = p.reshape(n, 2)
            d_left = xy_p[:, 0] - r
            d_right = 1 - xy_p[:, 0] - r
            d_bottom = xy_p[:, 1] - r
            d_top = 1 - xy_p[:, 1] - r
            boundary_viol = np.sum(np.maximum(0, -d_left)**2) + np.sum(np.maximum(0, -d_right)**2) + \
                            np.sum(np.maximum(0, -d_bottom)**2) + np.sum(np.maximum(0, -d_top)**2)

            dx = xy_p[:, 0][:, None] - xy_p[:, 0][None, :]
            dy = xy_p[:, 1][:, None] - xy_p[:, 1][None, :]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = r[:, None] + r[None, :]
            iu = np.triu_indices(n, k=1)
            overlap = np.maximum(0, min_dist[iu] - dist[iu])
            overlap_viol = np.sum(overlap**2)

            return boundary_viol + overlap_viol

        try:
            result = minimize(pos_obj, xy_init.flatten(), method='L-BFGS-B',
                             options={'maxiter': max_iter, 'ftol': 1e-14})
            return result.x
        except Exception:
            return xy_init.flatten()

    def grow_and_optimize(params, n_grow=50, growth_rate=0.004, time_limit=10):
        """Grow radii and re-optimize positions."""
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()
        t0 = time.time()

        for _ in range(n_grow):
            if time.time() - t0 > time_limit:
                break
            r_new = r * (1 + growth_rate)
            xy_opt = optimize_positions_fixed_radii(xy, r_new, max_iter=100)
            xy_trial = xy_opt.reshape(n, 2)
            
            # Check validity
            if is_valid(np.concatenate([xy_trial.flatten(), r_new])):
                xy = xy_trial
                r = r_new
            else:
                # Try with slightly smaller growth
                r_new2 = r * (1 + growth_rate * 0.5)
                xy_opt2 = optimize_positions_fixed_radii(xy, r_new2, max_iter=100)
                xy_trial2 = xy_opt2.reshape(n, 2)
                if is_valid(np.concatenate([xy_trial2.flatten(), r_new2])):
                    xy = xy_trial2
                    r = r_new2
                else:
                    break

        return np.concatenate([xy.flatten(), r])

    def full_optimize(params, max_iter=500):
        """Full optimization of both positions and radii."""
        n = N
        def penalty_function(p):
            xy_p = p[:2*n].reshape(n, 2)
            r_p = p[2*n:]
            
            d_left = xy_p[:, 0] - r_p
            d_right = 1 - xy_p[:, 0] - r_p
            d_bottom = xy_p[:, 1] - r_p
            d_top = 1 - xy_p[:, 1] - r_p
            boundary_viol = np.sum(np.maximum(0, -d_left)**2) + np.sum(np.maximum(0, -d_right)**2) + \
                            np.sum(np.maximum(0, -d_bottom)**2) + np.sum(np.maximum(0, -d_top)**2)

            dx = xy_p[:, 0][:, None] - xy_p[:, 0][None, :]
            dy = xy_p[:, 1][:, None] - xy_p[:, 1][None, :]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = r_p[:, None] + r_p[None, :]
            iu = np.triu_indices(n, k=1)
            overlap = np.maximum(0, min_dist[iu] - dist[iu])
            overlap_viol = np.sum(overlap**2)

            return -np.sum(r_p) + 1000 * boundary_viol + 1000 * overlap_viol

        try:
            result = minimize(penalty_function, params, method='L-BFGS-B',
                             options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12})
            return result.x
        except Exception:
            return params

    def get_initial_config(restart, rng):
        n = N
        configs = []
        
        # Type 0: 6x6 grid
        xy = np.zeros((n, 2))
        r = np.zeros(n)
        for i in range(n):
            row = i // 6
            col = i % 6
            xy[i] = [(col + 0.5)/6, (row + 0.5)/6]
            r[i] = 0.5/6 - 0.002
        configs.append((xy, r))
        
        # Type 1: Perturbed 6x6
        xy1 = configs[0][0].copy() + rng.uniform(-0.015, 0.015, size=(n, 2))
        configs.append((xy1, configs[0][1].copy()))
        
        # Type 2: Hexagonal
        xy2 = np.zeros((n, 2))
        r2 = np.zeros(n)
        for i in range(n):
            row = i // 6
            col = i % 6
            x = (col + 0.5)/6 + (0.25/6 if row % 2 == 1 else 0)
            y = (row + 0.5)/6
            xy2[i] = [x, y]
            r2[i] = 0.075
        configs.append((xy2, r2))
        
        # Type 3: Random
        xy3 = rng.uniform(0.08, 0.92, size=(n, 2))
        r3 = rng.uniform(0.05, 0.09, size=n)
        configs.append((xy3, r3))
        
        # Type 4: 8x4 grid
        xy4 = np.zeros((n, 2))
        r4 = np.zeros(n)
        for i in range(n):
            row = i // 8
            col = i % 8
            xy4[i] = [(col + 0.5)/8, (row + 0.5)/4]
            r4[i] = 0.5/8 - 0.001
        xy4 += rng.uniform(-0.01, 0.01, size=(n, 2))
        configs.append((xy4, r4))
        
        # Type 5: 4x8 grid
        xy5 = np.zeros((n, 2))
        r5 = np.zeros(n)
        for i in range(n):
            row = i // 4
            col = i % 4
            xy5[i] = [(col + 0.5)/4, (row + 0.5)/8]
            r5[i] = 0.5/4 - 0.001
        xy5 += rng.uniform(-0.01, 0.01, size=(n, 2))
        configs.append((xy5, r5))
        
        # Type 6: Hexagonal variant
        xy6 = np.zeros((n, 2))
        r6 = np.zeros(n)
        for i in range(n):
            row = i // 6
            col = i % 6
            x = (col + 0.5)/6 + (0.2/6 if row % 2 == 0 else 0)
            y = (row + 0.5)/6
            xy6[i] = [x, y]
            r6[i] = 0.07
        configs.append((xy6, r6))
        
        # Type 7: Clustered
        xy7 = np.zeros((n, 2))
        r7 = np.zeros(n)
        centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
        for i in range(n):
            group = i // 8
            cx, cy = centers[group]
            sub_i = i % 8
            sx = (sub_i % 3) / 3.0 + 0.5/3
            sy = (sub_i // 3) / 3.0 + 0.5/3
            xy7[i] = [cx + (sx - 0.5)*0.35, cy + (sy - 0.5)*0.35]
            r7[i] = 0.055
        configs.append((xy7, r7))
        
        # Type 8: Random medium
        xy8 = rng.uniform(0.08, 0.92, size=(n, 2))
        r8 = rng.uniform(0.06, 0.08, size=n)
        configs.append((xy8, r8))
        
        # Type 9: 5x7 grid
        xy9 = np.zeros((n, 2))
        r9 = np.zeros(n)
        for i in range(n):
            row = i // 7
            col = i % 7
            xy9[i] = [(col + 0.5)/7, (row + 0.5)/5]
            r9[i] = 0.5/7 - 0.001
        xy9 += rng.uniform(-0.01, 0.01, size=(n, 2))
        configs.append((xy9, r9))
        
        # Type 10: 7x5 grid
        xy10 = np.zeros((n, 2))
        r10 = np.zeros(n)
        for i in range(n):
            row = i // 5
            col = i % 5
            xy10[i] = [(col + 0.5)/5, (row + 0.5)/7]
            r10[i] = 0.5/5 - 0.001
        xy10 += rng.uniform(-0.01, 0.01, size=(n, 2))
        configs.append((xy10, r10))
        
        # Type 11: Hexagonal with different offset
        xy11 = np.zeros((n, 2))
        r11 = np.zeros(n)
        for i in range(n):
            row = i // 6
            col = i % 6
            x = (col + 0.5)/6 + (0.15/6 if row % 2 == 1 else 0)
            y = (row + 0.5)/6
            xy11[i] = [x, y]
            r11[i] = 0.073
        configs.append((xy11, r11))
        
        idx = restart % len(configs)
        xy, r = configs[idx]
        return np.concatenate([xy.flatten(), r])

    rng = np.random.RandomState(42)

    # Phase 1: Diverse restarts
    n_restarts = 80
    for restart in range(n_restarts):
        if time.time() > deadline - 40:
            break

        params = get_initial_config(restart, rng)
        params = push_apart_vectorized(params, n_iters=100)

        if is_valid(params):
            score = get_score(params)
            if score > best_score:
                best_score = score
                best_result = params.copy()

        # Full optimization
        refined = full_optimize(params, max_iter=300)
        if is_valid(refined):
            score = get_score(refined)
            if score > best_score:
                best_score = score
                best_result = refined.copy()
        else:
            # Repair
            repaired = push_apart_vectorized(refined, n_iters=100)
            if is_valid(repaired):
                score = get_score(repaired)
                if score > best_score:
                    best_score = score
                    best_result = repaired.copy()
            else:
                # Shrink
                xy = refined[:2*N].reshape(N, 2).copy()
                r = refined[2*N:].copy()
                for _ in range(100):
                    if is_valid(np.concatenate([xy.flatten(), r])):
                        break
                    r *= 0.95
                if is_valid(np.concatenate([xy.flatten(), r])):
                    score = get_score(np.concatenate([xy.flatten(), r]))
                    if score > best_score:
                        best_score = score
                        best_result = np.concatenate([xy.flatten(), r]).copy()

    # Phase 2: Grow radii on best
    if best_result is not None and time.time() < deadline - 20:
        grown = grow_and_optimize(best_result, n_grow=80, growth_rate=0.004, time_limit=15)
        if is_valid(grown):
            score = get_score(grown)
            if score > best_score:
                best_score = score
                best_result = grown.copy()

    # Phase 3: Polish
    if best_result is not None and time.time() < deadline - 10:
        polished = full_optimize(best_result, max_iter=500)
        if is_valid(polished):
            score = get_score(polished)
            if score > best_score:
                best_score = score
                best_result = polished.copy()

        # Perturb and re-optimize
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            perturbed = best_result.copy()
            xy_p = perturbed[:2*N].reshape(N, 2).copy()
            r_p = perturbed[2*N:].copy()
            xy_p += rng.uniform(-0.002, 0.002, size=(N, 2))
            perturbed = np.concatenate([xy_p.flatten(), r_p])
            perturbed = push_apart_vectorized(perturbed, n_iters=50)
            polished2 = full_optimize(perturbed, max_iter=300)
            if is_valid(polished2):
                score = get_score(polished2)
                if score > best_score:
                    best_score = score
                    best_result = polished2.copy()

    # Phase 4: Final grow
    if best_result is not None and time.time() < deadline - 3:
        grown2 = grow_and_optimize(best_result, n_grow=30, growth_rate=0.003, time_limit=8)
        if is_valid(grown2):
            score = get_score(grown2)
            if score > best_score:
                best_score = score
                best_result = grown2.copy()

    if best_result is not None:
        n = N
        xy = best_result[:2*n].reshape(n, 2).copy()
        r = best_result[2*n:].copy()
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1-r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1-r)
        final_params = np.concatenate([xy.flatten(), r])
        if is_valid(final_params):
            return np.column_stack([xy, r])

    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
