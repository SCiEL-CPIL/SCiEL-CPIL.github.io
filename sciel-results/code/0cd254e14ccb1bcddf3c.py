"""Circle packing in perimeter-4 rectangle: force-directed + SLSQP with smooth penalties."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w <= 0.05 or h <= 0.05:
            return False, 0.0

        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0

        val = float(np.sum(r))
        return True, val

    def _is_valid(cx, cy, r, W, H, tol=1e-8):
        if W <= 0.05 or H <= 0.05:
            return False
        if np.any(cx - r < -tol) or np.any(cy - r < -tol):
            return False
        if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                    return False
        return True

    def _vectorized_objective(z, penalty_weight=500.0):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6

        obj = -np.sum(r)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(N, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        overlap_penalty = penalty_weight * np.sum(overlap * overlap)

        obj += overlap_penalty

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)

        boundary_penalty = penalty_weight * (
            np.sum(left_viol * left_viol) +
            np.sum(right_viol * right_viol) +
            np.sum(bottom_viol * bottom_viol) +
            np.sum(top_viol * top_viol)
        )

        obj += boundary_penalty

        if w < 0.4:
            obj += penalty_weight * (0.4 - w) ** 2
        elif w > 1.6:
            obj += penalty_weight * (w - 1.6) ** 2

        return obj

    def objective(z):
        return _vectorized_objective(z, 500.0)

    def _force_relax(z, max_iter=500, dt=0.01):
        z = z.copy().astype(float)
        n = N
        iu = np.triu_indices(n, k=1)

        for it in range(max_iter):
            x = z[0::3][:n]
            y = z[1::3][:n]
            r = z[2::3][:n]
            w = z[-1]
            h = 2.0 - w

            if w < 0.1 or h < 0.1:
                break

            fx = np.zeros(n)
            fy = np.zeros(n)

            left_viol = np.maximum(r - x, 0.0)
            right_viol = np.maximum(x + r - w, 0.0)
            bottom_viol = np.maximum(r - y, 0.0)
            top_viol = np.maximum(y + r - h, 0.0)
            fx -= left_viol * 100
            fx += right_viol * 100
            fy -= bottom_viol * 100
            fy += top_viol * 100

            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            r_sum = r[:, None] + r[None, :]

            d_sq_pairs = d_sq[iu]
            r_sum_pairs = r_sum[iu]

            d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
            overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
            max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

            if max_overlap > 1e-10:
                f_pairs = overlap * 50.0
                dx_pairs = dx[iu]
                dy_pairs = dy[iu]
                ux = dx_pairs / d_pairs
                uy = dy_pairs / d_pairs

                for idx in range(len(iu[0])):
                    i_idx = iu[0][idx]
                    j_idx = iu[1][idx]
                    fx[i_idx] += ux[idx] * f_pairs[idx]
                    fy[i_idx] += uy[idx] * f_pairs[idx]
                    fx[j_idx] -= ux[idx] * f_pairs[idx]
                    fy[j_idx] -= uy[idx] * f_pairs[idx]

            if max_overlap < 1e-10:
                break

            force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
            if force_mag > 2.0:
                dt = max(0.0001, dt * 0.5)
            elif force_mag < 0.05:
                dt = min(0.02, dt * 1.3)

            z[0::3][:n] += dt * fx
            z[1::3][:n] += dt * fy

            for i in range(n):
                z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
                z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

        return z

    def _polish_positions(x, y, r, w, h, time_budget):
        """Force-directed polish to resolve any residual overlaps."""
        n = N
        iu = np.triu_indices(n, k=1)
        dt = 0.003
        max_iter = int(min(5000, time_budget * 200))

        for it in range(max_iter):
            if time.time() > time_budget:
                break

            fx = np.zeros(n)
            fy = np.zeros(n)

            left_viol = np.maximum(r - x, 0.0)
            right_viol = np.maximum(x + r - w, 0.0)
            bottom_viol = np.maximum(r - y, 0.0)
            top_viol = np.maximum(y + r - h, 0.0)
            fx -= left_viol * 50
            fx += right_viol * 50
            fy -= bottom_viol * 50
            fy += top_viol * 50

            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            r_sum = r[:, None] + r[None, :]

            d_sq_pairs = d_sq[iu]
            r_sum_pairs = r_sum[iu]

            d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
            overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

            if np.max(overlap) > 1e-11:
                f_pairs = overlap * 30.0
                dx_pairs = dx[iu]
                dy_pairs = dy[iu]
                ux = dx_pairs / d_pairs
                uy = dy_pairs / d_pairs

                for idx in range(len(iu[0])):
                    i_idx = iu[0][idx]
                    j_idx = iu[1][idx]
                    fx[i_idx] += ux[idx] * f_pairs[idx]
                    fy[i_idx] += uy[idx] * f_pairs[idx]
                    fx[j_idx] -= ux[idx] * f_pairs[idx]
                    fy[j_idx] -= uy[idx] * f_pairs[idx]

            max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
            if max_force > 0.2:
                dt = 0.001
            elif max_force > 0.03:
                dt = 0.003
            else:
                dt = 0.005

            x = x + dt * fx
            y = y + dt * fy
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

        return x, y

    def _grow_radii(x, y, r, w, h, time_budget):
        """Binary search for maximum uniform radius scaling."""
        for _ in range(60):
            if time.time() > time_budget:
                break
            lo, hi = 1.0, 1.02
            for _ in range(25):
                mid = (lo + hi) / 2.0
                r_trial = r * mid
                if _is_valid(x, y, r_trial, w, h):
                    lo = mid
                else:
                    hi = mid
            if lo > 1.0 + 1e-11:
                r = r * lo
            else:
                break
        return r

    def _expand_individual(x, y, r, w, h, time_budget):
        """Try to expand each circle individually."""
        n = N
        for rnd in range(30):
            if time.time() > time_budget:
                break
            improved = False
            for i in range(n):
                r_wall = min(x[i], w - x[i], y[i], h - y[i])
                r_max = r_wall
                for j in range(n):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    r_allowed = d - r[j]
                    if r_allowed < r_max:
                        r_max = r_allowed
                r_max = max(0.0, r_max)

                if r_max > r[i] + 1e-7:
                    lo = r[i]
                    hi = r_max
                    for _ in range(20):
                        mid = (lo + hi) / 2.0
                        r_trial = r.copy()
                        r_trial[i] = mid
                        if _is_valid(x, y, r_trial, w, h):
                            lo = mid
                        else:
                            hi = mid
                    if lo > r[i] + 1e-8:
                        r[i] = lo
                        improved = True

            if not improved:
                break
        return r

    def _local_search(x, y, r, w, h, time_budget):
        """8-direction local search for position optimization."""
        n = N
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                      (0.707, 0.707), (-0.707, 0.707), (0.707, -0.707), (-0.707, -0.707)]
        step_sizes = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002]

        for rnd in range(20):
            if time.time() > time_budget:
                break
            improved = False
            for i in range(n):
                # Find most constrained direction
                min_dist = float('inf')
                best_dir = None
                for dx, dy in directions:
                    nx = x[i] + dx * 0.01
                    ny = y[i] + dy * 0.01
                    if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                        continue
                    dist = min(nx - r[i], w - nx - r[i], ny - r[i], h - ny - r[i])
                    for j in range(n):
                        if i == j:
                            continue
                        ddx = nx - x[j]
                        ddy = ny - y[j]
                        d = np.sqrt(ddx * ddx + ddy * ddy)
                        gap = d - (r[i] + r[j])
                        if gap < dist:
                            dist = gap
                    if dist < min_dist:
                        min_dist = dist
                        best_dir = (dx, dy)

                if best_dir is None:
                    continue

                dx, dy = best_dir
                for step in step_sizes:
                    nx = x[i] + dx * step
                    ny = y[i] + dy * step
                    if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                        continue
                    valid = True
                    for j in range(n):
                        if i == j:
                            continue
                        ddx = nx - x[j]
                        ddy = ny - y[j]
                        d2 = ddx * ddx + ddy * ddy
                        rs = r[i] + r[j]
                        if d2 < rs * rs - 1e-9:
                            valid = False
                            break
                    if valid:
                        x[i] = nx
                        y[i] = ny
                        improved = True
                        break

            if not improved:
                break
        return x, y

    def _constraint_refine(z, time_budget):
        """SLSQP with explicit constraints for high-precision refinement."""
        n = N

        def obj_refine(zv):
            r = zv[2::3][:n]
            return -np.sum(r)

        constraints = []
        bounds = [(0, 2), (0, 2), (0.001, 1.0)] * n + [(0.3, 1.7)]

        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3*i] - p[3*i+2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3*i+1] - p[3*i+2]})

        for i in range(n):
            for j in range(i + 1, n):
                constraints.append({'type': 'ineq', 'fun': lambda p, i=i, j=j: (
                    (p[3*i] - p[3*j]) ** 2 + (p[3*i+1] - p[3*j+1]) ** 2 -
                    (p[3*i+2] + p[3*j+2]) ** 2
                )})

        maxiter = int(min(1000, time_budget * 100))
        try:
            res = minimize(obj_refine, z, method='SLSQP', bounds=bounds,
                           constraints=constraints,
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
            z_opt = res.x
            x = z_opt[0::3][:n]
            y = z_opt[1::3][:n]
            r = z_opt[2::3][:n]
            w = z_opt[-1]
            h = 2.0 - w
            if _is_valid(x, y, r, w, h):
                return z_opt
            r_repair = r * 0.98
            if _is_valid(x, y, r_repair, w, h):
                z_repair = z_opt.copy()
                z_repair[2::3][:n] = r_repair
                return z_repair
            return None
        except Exception:
            return None

    # Generate initial guesses
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))
        aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 0.7, 1.5, 0.65, 1.75, 0.6, 2.0, 0.55, 1.2, 0.85, 1.4, 0.72, 1.6, 0.68]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.25 or w0 < 0.25:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.40
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        rng = np.random.default_rng(42)
        for _ in range(10):
            w0 = rng.uniform(0.5, 1.5)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.12)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    # Phase 1: Differential Evolution
    if time.time() < deadline - 70:
        de_bounds = [(0, 2)] * (3 * N) + [(0.4, 1.6)]
        de_obj = lambda zv: _vectorized_objective(zv, 500.0)
        try:
            de_res = differential_evolution(
                de_obj, de_bounds, seed=42, maxiter=60,
                tol=1e-9, popsize=15, mutation=(0.4, 0.9),
                recombination=0.75, init='latinhypercube'
            )
            z_de = de_res.x
            valid, val = check_validity(z_de)
            if valid and val > best_value:
                best_value = val
                best_result = z_de.copy()
        except Exception:
            pass

    # Phase 2: Local optimization from multiple starts
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 35:
            break

        z0_relaxed = _force_relax(z0, max_iter=300)

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(400, max(80, remaining * 30)))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 60})
            z_opt = res.x
            valid, val = check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()
        except Exception:
            continue

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Phase 3: Polish pipeline
    if best_result is not None and time.time() < deadline - 20:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w

        # 1. Position polish
        x, y = _polish_positions(x, y, r, w, h, deadline - (time.time() - start_time) - 15)

        # 2. Local search
        x, y = _local_search(x, y, r, w, h, deadline - (time.time() - start_time) - 12)

        # 3. Grow radii
        r = _grow_radii(x, y, r, w, h, deadline - (time.time() - start_time) - 10)

        # 4. Individual expansion
        r = _expand_individual(x, y, r, w, h, deadline - (time.time() - start_time) - 8)

        # 5. Final position polish after expansion
        x, y = _polish_positions(x, y, r, w, h, deadline - (time.time() - start_time) - 6)

        # 6. Final radius growth
        r = _grow_radii(x, y, r, w, h, deadline - (time.time() - start_time) - 4)

        if _is_valid(x, y, r, w, h):
            z_new = np.empty(3 * N + 1)
            z_new[0::3][:N] = x
            z_new[1::3][:N] = y
            z_new[2::3][:N] = r
            z_new[-1] = w
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = z_new

    # Phase 4: Constraint-based refinement
    if best_result is not None and time.time() < deadline - 10:
        refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 5)
        if refined is not None:
            valid, val = check_validity(refined)
            if valid and val > best_value:
                best_value = val
                best_result = refined

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final safety shrink
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    for _ in range(5):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
