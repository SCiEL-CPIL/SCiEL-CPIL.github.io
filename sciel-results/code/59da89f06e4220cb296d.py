"""Circle packing in perimeter-4 rectangle: SLSQP + Sequential Placement + Contact Graph + Enhanced Polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import fsolve

N = 21


def _soft_max(v, beta=200.0):
    """Smooth approximation of max(v, 0) using log-sum-exp."""
    m = np.maximum(v, 0.0)
    return m + np.logaddexp(0.0, beta * (v - m)) / beta


def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth (soft-max) penalties for differentiability."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Vectorized pairwise overlap penalty (smooth)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap = _soft_max(r_sum_pairs - d_pairs, beta=200.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)

    obj += overlap_penalty

    # Vectorized boundary penalty (smooth)
    left_viol = _soft_max(r - x, beta=200.0)
    right_viol = _soft_max(x + r - w, beta=200.0)
    bottom_viol = _soft_max(r - y, beta=200.0)
    top_viol = _soft_max(y + r - h, beta=200.0)

    boundary_penalty = penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    obj += boundary_penalty

    # Keep aspect ratio in reasonable range
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    penalty_weight = 1000.0

    def objective(z):
        return _vectorized_objective(z, penalty_weight)

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

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(10):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 20:
            break

        # Force-directed relaxation to resolve initial overlaps
        z0_relaxed = _force_relax_unified(z0, max_iter=500)

        # SLSQP optimization
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(600, max(150, remaining * 50)))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})

            z_opt = res.x
            valid, val = check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                # Try repair: shrink radii slightly
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            continue

    # --- Phase 2: Sequential Placement Polish ---
    if best_result is not None and time.time() < deadline - 30:
        sequential = _sequential_placement_polish(best_result, deadline - (time.time() - start_time) - 20)
        if sequential is not None:
            valid, val = check_validity(sequential)
            if valid and val > best_value:
                best_value = val
                best_result = sequential

    # --- Phase 3: Advanced Polish on best result ---
    if best_result is not None and time.time() < deadline - 25:
        # 1. Force-directed position polish
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 15)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Expansion/Relocation Polish
        if best_result is not None and time.time() < deadline - 20:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 3. Contact graph optimization (NEW - from inspiration)
        if best_result is not None and time.time() < deadline - 15:
            contact_opt = _contact_graph_optimize(best_result, deadline - (time.time() - start_time) - 8)
            if contact_opt is not None:
                valid, val = check_validity(contact_opt)
                if valid and val > best_value:
                    best_value = val
                    best_result = contact_opt

        # 4. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 12:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 6)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 5. Final high-precision polish
        if best_result is not None and time.time() < deadline - 5:
            polished2 = _polish_result_tight(best_result, deadline - (time.time() - start_time) - 1)
            if polished2 is not None:
                valid, val = check_validity(polished2)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished2

        # 6. Multi-start local search
        if best_result is not None and time.time() < deadline - 1:
            best_local = _multi_start_local_search(best_result, deadline - (time.time() - start_time) - 0.5)
            if best_local is not None:
                valid, val = check_validity(best_local)
                if valid and val > best_value:
                    best_value = val
                    best_result = best_local

    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Final safety: ensure strict validity with minimal shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))


def _sequential_placement_polish(z, time_budget):
    """
    Sequential placement polish: treat circles in a greedy order (largest to smallest),
    and for each circle, reposition it to maximize its radius given all previously
    placed circles are fixed.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    order = np.argsort(-r)

    for idx in order:
        if time.time() > time_budget:
            break

        i = idx
        r_max_cur = _max_radius_for_circle(x, y, r, w, h, i)

        directions = [
            (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
            (0.7071, 0.7071), (-0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, -0.7071)
        ]
        steps = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]

        best_new_r_max = r_max_cur
        best_new_pos = None

        for dx_dir, dy_dir in directions:
            for step in steps:
                new_x = x[i] + dx_dir * step
                new_y = y[i] + dy_dir * step

                if new_x < r[i] or new_x > w - r[i] or new_y < r[i] or new_y > h - r[i]:
                    continue

                valid_move = True
                for j in range(n):
                    if i == j:
                        continue
                    dx = new_x - x[j]
                    dy = new_y - y[j]
                    if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                        valid_move = False
                        break
                if not valid_move:
                    continue

                r_wall_new = min(new_x, w - new_x, new_y, h - new_y)
                r_max_new = r_wall_new
                for j in range(n):
                    if i == j:
                        continue
                    dx = new_x - x[j]
                    dy = new_y - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    r_allowed = d - r[j]
                    if r_allowed < r_max_new:
                        r_max_new = r_allowed
                r_max_new = max(0.0, r_max_new)

                if r_max_new > best_new_r_max:
                    best_new_r_max = r_max_new
                    best_new_pos = (new_x, new_y)

        if best_new_pos is not None and best_new_r_max > r_max_cur + 1e-8:
            x[i], y[i] = best_new_pos
            lo = r[i]
            hi = best_new_r_max
            for _ in range(20):
                mid = (lo + hi) / 2.0
                r_test = r.copy()
                r_test[i] = mid
                if _is_valid(x, y, r_test, w, h):
                    lo = mid
                else:
                    hi = mid
            if lo > r[i] + 1e-11:
                r[i] = lo

    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0001
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _force_relax_unified(z, max_iter=500):
    """Force-directed relaxation in unified parameter space with adaptive step (vectorized)."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001
    convergence_tol = 1e-8

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

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > convergence_tol:
            f_pairs = overlap * 15.0
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

        if max_overlap < convergence_tol:
            break

        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.01, dt * 1.1)

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _polish_result(z, time_budget):
    """Polish a valid packing with force-directed position adjustment and radius growth (vectorized)."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.001
    max_iter = int(min(3000, time_budget * 150))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 35
        fx += right_viol * 35
        fy -= bottom_viol * 35
        fy += top_viol * 35

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            f_pairs = overlap * 18.0
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
        if max_force > 0.1:
            dt = 0.0005
        else:
            dt = 0.001

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    grow_factor = 1.0003
    for _ in range(300):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w

    if _is_valid(x, y, r, w, h):
        return z_new
    return None


def _expansion_relaxation_polish(z, time_budget):
    """
    Advanced polish: Iteratively expand circles with position optimization.
    Uses 16-direction local search to reposition circles for larger radii.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # 16 directions for local search
    directions = []
    for k in range(16):
        angle = 2.0 * np.pi * k / 16.0
        directions.append((np.cos(angle), np.sin(angle)))

    max_rounds = 40
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            
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
            
            if r_max > r[i] + 1e-6:
                lo = r[i]
                hi = r_max
                for _ in range(25):
                    mid = (lo + hi) / 2.0
                    if _is_valid_move(x, y, np.where(np.arange(n) == i, mid, r), w, h, i, x[i], y[i]):
                        lo = mid
                    else:
                        hi = mid
                new_r = lo
                if new_r > r[i] + 1e-7:
                    r[i] = new_r
                    improved = True
            
            min_gap = float('inf')
            best_dx, best_dy = 0.0, 0.0
            
            d_left = x[i] - r[i]
            d_right = w - x[i] - r[i]
            d_bottom = y[i] - r[i]
            d_top = h - y[i] - r[i]
            
            if d_left < min_gap:
                min_gap = d_left
                best_dx, best_dy = 1.0, 0.0
            if d_right < min_gap:
                min_gap = d_right
                best_dx, best_dy = -1.0, 0.0
            if d_bottom < min_gap:
                min_gap = d_bottom
                best_dx, best_dy = 0.0, 1.0
            if d_top < min_gap:
                min_gap = d_top
                best_dx, best_dy = 0.0, -1.0
            
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_gap:
                    min_gap = gap
                    if d > 1e-6:
                        best_dx = dx / d
                        best_dy = dy / d
            
            if min_gap < 0.05:
                found_better = False
                for dx_dir, dy_dir in directions:
                    if found_better:
                        break
                    for step in [0.04, 0.02, 0.01, 0.005, 0.002, 0.001]:
                        new_x = x[i] + dx_dir * step
                        new_y = y[i] + dy_dir * step
                        if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                            old_r_max = _max_radius_for_circle(x, y, r, w, h, i)
                            x[i], y[i] = new_x, new_y
                            new_r_max = _max_radius_for_circle(x, y, r, w, h, i)
                            if new_r_max > old_r_max + 1e-6:
                                improved = True
                                found_better = True
                                break
                            else:
                                x[i], y[i] = x[i] - dx_dir * step, y[i] - dy_dir * step

                if found_better:
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
                    
                    if r_max > r[i] + 1e-6:
                        lo = r[i]
                        hi = r_max
                        for _ in range(25):
                            mid = (lo + hi) / 2.0
                            if _is_valid_move(x, y, np.where(np.arange(n) == i, mid, r), w, h, i, x[i], y[i]):
                                lo = mid
                            else:
                                hi = mid
                        new_r = lo
                        if new_r > r[i] + 1e-7:
                            r[i] = new_r
                            improved = True

        if not improved:
            break

    # Third pass: force-directed relaxation after expansion
    if time.time() < time_budget - 5:
        z_temp = np.empty(3 * N + 1)
        z_temp[0::3][:n] = x
        z_temp[1::3][:n] = y
        z_temp[2::3][:n] = r
        z_temp[-1] = w
        z_relaxed = _force_relax_unified(z_temp, max_iter=200)
        x = z_relaxed[0::3][:n]
        y = z_relaxed[1::3][:n]
        r = z_relaxed[2::3][:n]
        w = z_relaxed[-1]
        h = 2.0 - w

    # Second pass: more aggressive expansion with position optimization
    if time.time() < time_budget - 5:
        for rnd in range(20):
            if time.time() > time_budget:
                break
            
            improved = False
            for i in range(n):
                if time.time() > time_budget:
                    break
                
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
                
                if r_max > r[i] + 1e-5:
                    lo = r[i]
                    hi = r_max
                    for _ in range(30):
                        mid = (lo + hi) / 2.0
                        if _is_valid_move(x, y, np.where(np.arange(n) == i, mid, r), w, h, i, x[i], y[i]):
                            lo = mid
                        else:
                            hi = mid
                    new_r = lo
                    if new_r > r[i] + 1e-6:
                        r[i] = new_r
                        improved = True

            if not improved:
                break

    grow_factor = 1.0005
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _contact_graph_optimize(z, time_budget):
    """
    Contact graph optimization: identify contacts and solve tangency equations
    to find a more optimal configuration.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Identify contacts: pairs of circles that are nearly touching
    contact_tol = 1e-4
    contacts = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_sum = r[i] + r[j]
            if abs(d - r_sum) < contact_tol:
                contacts.append((i, j))

    # Also check wall contacts
    wall_contacts = []
    for i in range(n):
        if abs(x[i] - r[i]) < contact_tol:
            wall_contacts.append((i, 'left'))
        if abs(w - x[i] - r[i]) < contact_tol:
            wall_contacts.append((i, 'right'))
        if abs(y[i] - r[i]) < contact_tol:
            wall_contacts.append((i, 'bottom'))
        if abs(h - y[i] - r[i]) < contact_tol:
            wall_contacts.append((i, 'top'))

    # Try to solve for a configuration where all contacts are exact
    # Use fsolve to find positions that satisfy tangency constraints
    if len(contacts) >= 3 and time.time() < time_budget - 5:
        # Count degrees of freedom: 2n positions + n radii - constraints
        # For a rigid packing, we need enough contacts to determine the configuration
        n_constraints = len(contacts) + len(wall_contacts)
        n_unknowns = 3 * n  # positions and radii
        
        # Only proceed if we have a reasonable number of constraints
        if n_constraints >= n_unknowns - 3:  # Allow some flexibility
            # Define the system of equations
            def equations(params):
                x_new = params[0:n]
                y_new = params[n:2*n]
                r_new = params[2*n:3*n]
                eqs = []
                
                # Circle-circle tangency
                for i, j in contacts:
                    dx = x_new[i] - x_new[j]
                    dy = y_new[i] - y_new[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    eqs.append(d - r_new[i] - r_new[j])
                
                # Wall tangency
                for i, side in wall_contacts:
                    if side == 'left':
                        eqs.append(x_new[i] - r_new[i])
                    elif side == 'right':
                        eqs.append(w - x_new[i] - r_new[i])
                    elif side == 'bottom':
                        eqs.append(y_new[i] - r_new[i])
                    elif side == 'top':
                        eqs.append(h - y_new[i] - r_new[i])
                
                # Perimeter constraint
                eqs.append(2.0 * (w + h) - 4.0)
                
                return eqs
            
            # Initial guess
            params0 = np.concatenate([x, y, r])
            
            try:
                sol, info, ier, msg = fsolve(equations, params0, full_output=True, maxfev=10000)
                
                if ier == 1:
                    x_sol = sol[0:n]
                    y_sol = sol[n:2*n]
                    r_sol = sol[2*n:3*n]
                    
                    # Check if solution is valid
                    if _is_valid(x_sol, y_sol, r_sol, w, h):
                        val = float(np.sum(r_sol))
                        if val > float(np.sum(r)) + 1e-6:
                            z_new = np.empty(3 * N + 1)
                            z_new[0::3][:n] = x_sol
                            z_new[1::3][:n] = y_sol
                            z_new[2::3][:n] = r_sol
                            z_new[-1] = w
                            return z_new
            except Exception:
                pass

    # Fallback: use SLSQP with tighter tolerance on the contact structure
    if time.time() < time_budget - 3:
        def obj_tight(zv):
            rv = zv[2::3][:n]
            return -np.sum(rv)

        constraints = []
        bounds = []

        for i in range(n):
            bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
        bounds.append((0.3, 1.7))

        def make_boundary_constraint(idx, axis, side):
            if axis == 0:
                if side == 0:
                    return lambda p: p[3 * idx] - p[3 * idx + 2]
                else:
                    return lambda p: p[-1] - p[3 * idx] - p[3 * idx + 2]
            else:
                if side == 0:
                    return lambda p: p[3 * idx + 1] - p[3 * idx + 2]
                else:
                    return lambda p: (2.0 - p[-1]) - p[3 * idx + 1] - p[3 * idx + 2]

        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 0)})
            constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 1)})
            constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 0)})
            constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 1)})

        for i in range(n):
            for j in range(i + 1, n):
                def make_overlap_constraint(idx1, idx2):
                    return lambda p: (
                        (p[3 * idx1] - p[3 * idx2]) ** 2 +
                        (p[3 * idx1 + 1] - p[3 * idx2 + 1]) ** 2 -
                        (p[3 * idx1 + 2] + p[3 * idx2 + 2]) ** 2
                    )
                constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})

        maxiter = int(min(800, (time_budget - (time.time() - start_time)) * 150))
        try:
            res = minimize(obj_tight, z, method='SLSQP',
                           bounds=bounds,
                           constraints=constraints,
                           options={'maxiter': maxiter, 'ftol': 1e-16, 'maxls': 100})

            z_opt = res.x
            if _is_valid(z_opt[0::3][:n], z_opt[1::3][:n], z_opt[2::3][:n], z_opt[-1], 2.0 - z_opt[-1]):
                return z_opt
        except Exception:
            pass

    return None


def _max_radius_for_circle(x, y, r, w, h, i):
    """Calculate maximum possible radius for circle i given other positions."""
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    for j in range(N):
        if i == j: continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx*dx + dy*dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)


def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    """Check if moving circle i to new_x, new_y is valid."""
    if new_x < r[i] or new_x > w - r[i]:
        return False
    if new_y < r[i] or new_y > h - r[i]:
        return False
    
    for j in range(N):
        if i == j: continue
        dx = new_x - x[j]
        dy = new_y - y[j]
        d2 = dx*dx + dy*dy
        r_sum = r[i] + r[j]
        if d2 < r_sum*r_sum - 1e-9:
            return False
    return True


def _constraint_refine(z, time_budget):
    """High-precision SLSQP refinement using explicit constraints."""
    z = z.copy().astype(float)
    n = N

    def obj_refine(zv):
        r = zv[2::3][:n]
        return -np.sum(r)

    constraints = []
    bounds = []

    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    def make_boundary_constraint(idx, axis, side):
        if axis == 0:
            if side == 0:
                return lambda p: p[3 * idx] - p[3 * idx + 2]
            else:
                return lambda p: p[-1] - p[3 * idx] - p[3 * idx + 2]
        else:
            if side == 0:
                return lambda p: p[3 * idx + 1] - p[3 * idx + 2]
            else:
                return lambda p: (2.0 - p[-1]) - p[3 * idx + 1] - p[3 * idx + 2]

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 0)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 1)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 0)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 1)})

    for i in range(n):
        for j in range(i + 1, n):
            def make_overlap_constraint(idx1, idx2):
                return lambda p: (
                    (p[3 * idx1] - p[3 * idx2]) ** 2 +
                    (p[3 * idx1 + 1] - p[3 * idx2 + 1]) ** 2 -
                    (p[3 * idx1 + 2] + p[3 * idx2 + 2]) ** 2
                )
            constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})

    maxiter = int(min(800, time_budget * 150))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-16, 'maxls': 100})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if _is_valid(x, y, r, w, h):
            return z_opt
        return None
    except Exception:
        return None


def _polish_result_tight(z, time_budget):
    """High-precision final polish: maximize each circle to its limit, then local search."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    for outer_iter in range(50):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
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
            if r_max > r[i] + 1e-10:
                lo = r[i]
                hi = r_max
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-11:
                    r[i] = lo
                    improved = True
        if not improved:
            break

    directions = [
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (0.7071, 0.7071), (-0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, -0.7071)
    ]
    steps = [0.04, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for ls_round in range(40):
        if time.time() > time_budget:
            break
        round_improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max_cur = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max_cur:
                    r_max_cur = r_allowed
            r_max_cur = max(0.0, r_max_cur)
            
            best_new_r_max = r_max_cur
            best_new_pos = None
            for dx_dir, dy_dir in directions:
                for step in steps:
                    new_x = x[i] + dx_dir * step
                    new_y = y[i] + dy_dir * step
                    if new_x < r[i] or new_x > w - r[i] or new_y < r[i] or new_y > h - r[i]:
                        continue
                    valid_move = True
                    for j in range(n):
                        if i == j:
                            continue
                        dx = new_x - x[j]
                        dy = new_y - y[j]
                        if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                            valid_move = False
                            break
                    if not valid_move:
                        continue
                    r_wall_new = min(new_x, w - new_x, new_y, h - new_y)
                    r_max_new = r_wall_new
                    for j in range(n):
                        if i == j:
                            continue
                        dx = new_x - x[j]
                        dy = new_y - y[j]
                        d = np.sqrt(dx * dx + dy * dy)
                        r_allowed = d - r[j]
                        if r_allowed < r_max_new:
                            r_max_new = r_allowed
                    r_max_new = max(0.0, r_max_new)
                    if r_max_new > best_new_r_max:
                        best_new_r_max = r_max_new
                        best_new_pos = (new_x, new_y)
            
            if best_new_pos is not None and best_new_r_max > r_max_cur + 1e-8:
                x[i], y[i] = best_new_pos
                lo = r[i]
                hi = best_new_r_max
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-10:
                    r[i] = lo
                    round_improved = True

        if not round_improved:
            break

    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0001
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid (vectorized)."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    return np.all(d_sq[iu] >= r_sum[iu] ** 2 - tol)


def _multi_start_local_search(z, time_budget):
    """Multi-start local search: perturb and maximize radii."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    best_val = float(np.sum(r))
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    rng = np.random.default_rng(123)
    directions = [
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (0.7071, 0.7071), (-0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, -0.7071)
    ]
    
    for trial in range(30):
        if time.time() > time_budget:
            break
        
        x_p = x.copy()
        y_p = y.copy()
        r_p = r.copy()
        
        n_perturb = rng.integers(2, 7)
        indices = rng.choice(n, size=n_perturb, replace=False)
        for idx in indices:
            dx_dir, dy_dir = directions[rng.integers(0, 8)]
            step = rng.uniform(0.005, 0.03)
            new_x = x_p[idx] + dx_dir * step
            new_y = y_p[idx] + dy_dir * step
            if _is_valid_move(x_p, y_p, r_p, w, h, idx, new_x, new_y):
                x_p[idx], y_p[idx] = new_x, new_y
        
        for outer_iter in range(20):
            improved = False
            for i in range(n):
                if time.time() > time_budget:
                    break
                r_wall = min(x_p[i], w - x_p[i], y_p[i], h - y_p[i])
                r_max = r_wall
                for j in range(n):
                    if i == j:
                        continue
                    dx = x_p[i] - x_p[j]
                    dy = y_p[i] - y_p[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    r_allowed = d - r_p[j]
                    if r_allowed < r_max:
                        r_max = r_allowed
                r_max = max(0.0, r_max)
                if r_max > r_p[i] + 1e-10:
                    lo = r_p[i]
                    hi = r_max
                    for _ in range(15):
                        mid = (lo + hi) / 2.0
                        r_test = r_p.copy()
                        r_test[i] = mid
                        if _is_valid(x_p, y_p, r_test, w, h):
                            lo = mid
                        else:
                            hi = mid
                    if lo > r_p[i] + 1e-11:
                        r_p[i] = lo
                        improved = True
            if not improved:
                break
        
        val_p = float(np.sum(r_p))
        if _is_valid(x_p, y_p, r_p, w, h) and val_p > best_val:
            best_val = val_p
            best_x, best_y, best_r = x_p.copy(), y_p.copy(), r_p.copy()
    
    if best_val > float(np.sum(r)) + 1e-12:
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = best_x
        z_new[1::3][:n] = best_y
        z_new[2::3][:n] = best_r
        z_new[-1] = w
        return z_new
    return None
