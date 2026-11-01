import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 138  # Safety margin for 150s limit

    best_result = None
    best_score = -np.inf

    # --- Parameter layout: interleaved [x1, y1, r1, x2, y2, r2, ...] ---
    # This is more natural for SLSQP constraints and easier to reason about.

    def get_xy_r(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        return x, y, r

    def make_z(x, y, r):
        z = np.zeros(3 * N)
        z[0::3] = x
        z[1::3] = y
        z[2::3] = r
        return z

    def penalty_function(z):
        x, y, r = get_xy_r(z)
        # Boundary penalty
        d_left = x - r
        d_right = 1 - x - r
        d_bottom = y - r
        d_top = 1 - y - r
        boundary_viol = (np.sum(np.maximum(0, -d_left)**2) + np.sum(np.maximum(0, -d_right)**2) +
                         np.sum(np.maximum(0, -d_bottom)**2) + np.sum(np.maximum(0, -d_top)**2))

        # Overlap penalty
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        overlap = np.maximum(0, min_dist[iu] - dist[iu])
        overlap_viol = np.sum(overlap**2)

        return -np.sum(r) + 100 * boundary_viol + 100 * overlap_viol

    def is_valid(z, tol=1e-9):
        x, y, r = get_xy_r(z)

        if np.any(r < -tol):
            return False
        if np.any(x - r < -tol) or np.any(1 - x - r < -tol):
            return False
        if np.any(y - r < -tol) or np.any(1 - y - r < -tol):
            return False

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx**2 + dy**2
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        if np.any(dist_sq[iu] < (min_dist[iu] - tol)**2):
            return False

        return True

    def get_score(z):
        _, _, r = get_xy_r(z)
        return np.sum(r)

    def push_apart(z, n_iters=100, rng=None):
        x, y, r = get_xy_r(z)
        x = x.copy()
        y = y.copy()
        r = r.copy()
        if rng is None:
            rng = np.random

        for _ in range(n_iters):
            moved = False

            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    dist_sq = dx**2 + dy**2
                    min_dist = r[i] + r[j]

                    if dist_sq < min_dist**2 and dist_sq > 1e-24:
                        dist = np.sqrt(dist_sq)
                        overlap = min_dist - dist
                        wi = r[i] / (r[i] + r[j]) if r[i] + r[j] > 1e-15 else 0.5
                        wj = 1 - wi

                        push_x = (dx / dist) * overlap
                        push_y = (dy / dist) * overlap

                        x[i] -= push_x * wi
                        y[i] -= push_y * wi
                        x[j] += push_x * wj
                        y[j] += push_y * wj
                        moved = True
                    elif dist_sq <= 1e-24:
                        angle = rng.uniform(0, 2 * np.pi)
                        x[i] -= 0.01 * np.cos(angle)
                        y[i] -= 0.01 * np.sin(angle)
                        x[j] += 0.01 * np.cos(angle)
                        y[j] += 0.01 * np.sin(angle)
                        moved = True

            for i in range(N):
                if x[i] < r[i]:
                    x[i] = r[i]
                    moved = True
                if x[i] > 1 - r[i]:
                    x[i] = 1 - r[i]
                    moved = True
                if y[i] < r[i]:
                    y[i] = r[i]
                    moved = True
                if y[i] > 1 - r[i]:
                    y[i] = 1 - r[i]
                    moved = True

            if not moved:
                break

        return make_z(x, y, r)

    # --- SLSQP with explicit constraints (from inspiration) ---
    def calc_constraints(z):
        x, y, r = get_xy_r(z)

        c_bounds = np.concatenate([
            x - r,
            1.0 - x - r,
            y - r,
            1.0 - y - r
        ])

        # Pairwise non-overlap: dist^2 - (ri + rj)^2 >= 0
        c_pairwise = []
        for i in range(N):
            for j in range(i + 1, N):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                rad_sum = r[i] + r[j]
                c_pairwise.append(dist_sq - rad_sum**2)

        return np.concatenate([c_bounds, c_pairwise])

    def obj_func_slsqp(z):
        _, _, r = get_xy_r(z)
        return -np.sum(r)

    bounds_slsqp = []
    for _ in range(N):
        bounds_slsqp.append((0, 1))   # x
        bounds_slsqp.append((0, 1))   # y
        bounds_slsqp.append((0, 0.5)) # r

    constraints_slsqp = [
        {'type': 'ineq', 'fun': calc_constraints}
    ]

    def refine_slsqp(z, max_iter=500, ftol=1e-12):
        try:
            result = minimize(obj_func_slsqp, z, method='SLSQP',
                              bounds=bounds_slsqp, constraints=constraints_slsqp,
                              options={'maxiter': max_iter, 'ftol': ftol})
            return result.x
        except Exception:
            return z

    def refine_l_bfgs_b(z, max_iter=500, tol=1e-14):
        try:
            result = minimize(penalty_function, z, method='L-BFGS-B',
                              options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12, 'maxls': 50})
            return result.x
        except Exception:
            return z

    def high_precision_polish(z, max_iter=2000, tol=1e-16):
        try:
            result = minimize(penalty_function, z, method='L-BFGS-B',
                              options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-14, 'maxls': 50})
            return result.x
        except Exception:
            return z

    # --- Initializations ---
    def init_grid_6x6(rng):
        side = 6
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side - 0.001
        x += rng.uniform(-0.02, 0.02, N)
        y += rng.uniform(-0.02, 0.02, N)
        return make_z(x, y, r)

    def init_hex_4x8(rng):
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(4):
            offset = 0 if i % 2 == 0 else 0.0625
            for j in range(8):
                idx = i * 8 + j
                x[idx] = offset + j * 0.125 + 0.0625
                y[idx] = 0.15 + i * 0.216
                r[idx] = 0.06
        x += rng.uniform(-0.01, 0.01, N)
        y += rng.uniform(-0.01, 0.01, N)
        return make_z(x, y, r)

    def init_random(rng):
        x = rng.uniform(0.08, 0.92, N)
        y = rng.uniform(0.08, 0.92, N)
        r = rng.uniform(0.05, 0.1, N)
        return make_z(x, y, r)

    def init_grid_8x4(rng):
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row = i // 8
            col = i % 8
            x[i] = (col + 0.5) / 8
            y[i] = (row + 0.5) / 4
            r[i] = 0.5 / 8 - 0.001
        x += rng.uniform(-0.01, 0.01, N)
        y += rng.uniform(-0.01, 0.01, N)
        return make_z(x, y, r)

    def init_clustered(rng):
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
        for i in range(N):
            group = i // 8
            cx, cy = centers[group]
            sub_i = i % 8
            sx = (sub_i % 3) / 3.0 + 0.5 / 3
            sy = (sub_i // 3) / 3.0 + 0.5 / 3
            x[i] = cx + (sx - 0.5) * 0.4
            y[i] = cy + (sy - 0.5) * 0.4
            r[i] = 0.06
        x += rng.uniform(-0.01, 0.01, N)
        y += rng.uniform(-0.01, 0.01, N)
        return make_z(x, y, r)

    def init_voronoi_like(rng):
        side = 6
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.08
        x += rng.uniform(-0.03, 0.03, N)
        y += rng.uniform(-0.03, 0.03, N)
        r += rng.uniform(-0.01, 0.01, N)
        r = np.clip(r, 0.03, 0.15)
        return make_z(x, y, r)

    rng = np.random.RandomState(12345)

    init_funcs = [init_grid_6x6, init_hex_4x8, init_random, init_grid_8x4,
                  init_clustered, init_voronoi_like]

    # Phase 1: Multiple restarts with both SLSQP and L-BFGS-B
    n_restarts = 80
    for restart in range(n_restarts):
        if time.time() > deadline:
            break

        init_func = init_funcs[restart % len(init_funcs)]
        z = init_func(rng)
        z = push_apart(z, n_iters=50, rng=rng)

        if is_valid(z):
            score = get_score(z)
            if score > best_score:
                best_score = score
                best_result = z.copy()

        # Try SLSQP first (handles constraints natively)
        if time.time() < deadline - 15:
            refined_slsqp = refine_slsqp(z, max_iter=400, ftol=1e-12)
            if is_valid(refined_slsqp):
                score = get_score(refined_slsqp)
                if score > best_score:
                    best_score = score
                    best_result = refined_slsqp.copy()
            else:
                repaired = push_apart(refined_slsqp, n_iters=80, rng=rng)
                if is_valid(repaired):
                    score = get_score(repaired)
                    if score > best_score:
                        best_score = score
                        best_result = repaired.copy()

        # Then L-BFGS-B with penalty (can escape some SLSQP local optima)
        if time.time() < deadline - 10:
            refined_lbfgs = refine_l_bfgs_b(z, max_iter=300, tol=1e-13)
            if is_valid(refined_lbfgs):
                score = get_score(refined_lbfgs)
                if score > best_score:
                    best_score = score
                    best_result = refined_lbfgs.copy()
            else:
                repaired = push_apart(refined_lbfgs, n_iters=80, rng=rng)
                if is_valid(repaired):
                    score = get_score(repaired)
                    if score > best_score:
                        best_score = score
                        best_result = repaired.copy()

    # Phase 2: High-precision polish on best solution
    if best_result is not None and time.time() < deadline - 10:
        # SLSQP polish
        if time.time() < deadline - 8:
            polished_slsqp = refine_slsqp(best_result, max_iter=1000, ftol=1e-14)
            if is_valid(polished_slsqp):
                score = get_score(polished_slsqp)
                if score > best_score:
                    best_score = score
                    best_result = polished_slsqp.copy()

        # L-BFGS-B high-precision polish
        if time.time() < deadline - 8:
            polished_lbfgs = high_precision_polish(best_result, max_iter=2000, tol=1e-16)
            if is_valid(polished_lbfgs):
                score = get_score(polished_lbfgs)
                if score > best_score:
                    best_score = score
                    best_result = polished_lbfgs.copy()

        # Perturbation attempts
        for polish_iter in range(15):
            if time.time() > deadline - 5:
                break
            scale = 0.002 if polish_iter < 8 else (0.005 if polish_iter < 12 else 0.01)

            x, y, r = get_xy_r(best_result)
            x_p = x + rng.uniform(-scale, scale, N)
            y_p = y + rng.uniform(-scale, scale, N)
            perturbed = make_z(x_p, y_p, r.copy())
            perturbed = push_apart(perturbed, n_iters=40, rng=rng)

            # Try both optimizers
            if time.time() < deadline - 4:
                p1 = refine_slsqp(perturbed, max_iter=500, ftol=1e-13)
                if is_valid(p1):
                    score = get_score(p1)
                    if score > best_score:
                        best_score = score
                        best_result = p1.copy()

            if time.time() < deadline - 4:
                p2 = high_precision_polish(perturbed, max_iter=1000, tol=1e-15)
                if is_valid(p2):
                    score = get_score(p2)
                    if score > best_score:
                        best_score = score
                        best_result = p2.copy()

        # Radius perturbation
        if time.time() < deadline - 5:
            for r_pert_iter in range(5):
                if time.time() > deadline - 5:
                    break
                x, y, r = get_xy_r(best_result)
                r_p = r * (1 + rng.uniform(-0.02, 0.02, N))
                r_p = np.clip(r_p, 0.01, 0.3)
                perturbed = make_z(x.copy(), y.copy(), r_p)
                perturbed = push_apart(perturbed, n_iters=60, rng=rng)

                p = high_precision_polish(perturbed, max_iter=1000, tol=1e-15)
                if is_valid(p):
                    score = get_score(p)
                    if score > best_score:
                        best_score = score
                        best_result = p.copy()

    if best_result is not None:
        x, y, r = get_xy_r(best_result)
        r = np.clip(r, 0, 0.5)
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        final_z = make_z(x, y, r)
        if is_valid(final_z):
            # Final micro-polish
            if time.time() < deadline - 1:
                final_polished = high_precision_polish(final_z, max_iter=500, tol=1e-16)
                if is_valid(final_polished):
                    score = get_score(final_polished)
                    if score > get_score(final_z):
                        final_z = final_polished
                        x, y, r = get_xy_r(final_z)
                        r = np.clip(r, 0, 0.5)
                        x = np.clip(x, r, 1 - r)
                        y = np.clip(y, r, 1 - r)
                        final_z = make_z(x, y, r)
            return np.column_stack([x, y, r])

    # Fallback: grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
