"""Circle packing in perimeter-4 rectangle: SLSQP + Advanced Expansion Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(r <= 0):
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


def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    """Check if moving circle i to new_x, new_y is valid."""
    if new_x < r[i] or new_x > w - r[i]:
        return False
    if new_y < r[i] or new_y > h - r[i]:
        return False
    for j in range(N):
        if i == j:
            continue
        dx = new_x - x[j]
        dy = new_y - y[j]
        d2 = dx * dx + dy * dy
        r_sum = r[i] + r[j]
        if d2 < r_sum * r_sum - 1e-9:
            return False
    return True


def _force_relax_unified(z, max_iter=500):
    """Force-directed relaxation in unified parameter space with adaptive step."""
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

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

        # Pairwise repulsion
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


def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth penalties for speed."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap_raw = r_sum_pairs - d_pairs
    overlap = np.maximum(overlap_raw, 0.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)
    obj += overlap_penalty

    # Boundary penalty
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

    # Aspect ratio penalty
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _polish_result(z, time_budget):
    """Polish a valid packing with force-directed position adjustment and radius growth."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.001
    max_iter = int(min(5000, time_budget * 200))
    gap_smooth_len = 0.05

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

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        gap = d_pairs - r_sum_pairs

        overlap_force = np.maximum(-gap, 0.0) * 25.0
        soft_force = 0.8 * np.exp(-gap / gap_smooth_len)
        f_pairs = overlap_force + soft_force

        active = f_pairs > 0.01
        if np.any(active):
            idx_active = np.where(active)[0]
            for idx in idx_active:
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                d = d_pairs[idx]
                if d > 1e-6:
                    ux = dx[i_idx, j_idx] / d
                    uy = dy[i_idx, j_idx] / d
                    f = f_pairs[idx]
                    fx[i_idx] += ux * f
                    fy[i_idx] += uy * f
                    fx[j_idx] -= ux * f
                    fy[j_idx] -= uy * f

        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.1:
            dt = 0.0004
        elif max_force > 0.01:
            dt = 0.0008
        else:
            dt = 0.0012

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Try to grow radii uniformly with binary search for max growth
    for _ in range(200):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.05
        for _bs in range(25):
            mid = (lo + hi) / 2.0
            r_trial = r * mid
            if _is_valid(x, y, r_trial, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-8:
            r = r * lo
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
    """Advanced polish: Iteratively expand circles and shift positions to create space."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    max_rounds = 40
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            # 1. Try to expand radius i to max possible using binary search
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

            # 2. If expansion is limited, try to shift circle i to create space
            min_dist = float('inf')
            blocker = None
            d_left = x[i] - r[i]
            d_right = w - x[i] - r[i]
            d_bottom = y[i] - r[i]
            d_top = h - y[i] - r[i]

            if d_left < min_dist:
                min_dist = d_left
                blocker = 'left'
            if d_right < min_dist:
                min_dist = d_right
                blocker = 'right'
            if d_bottom < min_dist:
                min_dist = d_bottom
                blocker = 'bottom'
            if d_top < min_dist:
                min_dist = d_top
                blocker = 'top'

            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_dist:
                    min_dist = gap
                    blocker = j

            if min_dist < 0.03:
                move_x = 0.0
                move_y = 0.0
                if blocker == 'left':
                    move_x = 1.0
                elif blocker == 'right':
                    move_x = -1.0
                elif blocker == 'bottom':
                    move_y = 1.0
                elif blocker == 'top':
                    move_y = -1.0
                else:
                    j = blocker
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d > 1e-6:
                        move_x = dx / d
                        move_y = dy / d

                directions = [
                    (1, 0), (-1, 0), (0, 1), (0, -1),
                    (1, 1), (-1, -1), (1, -1), (-1, 1)
                ]
                directions.sort(key=lambda d: -(d[0]*move_x + d[1]*move_y))

                found = False
                for step in [0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001]:
                    for (dx_dir, dy_dir) in directions:
                        norm = np.sqrt(dx_dir*dx_dir + dy_dir*dy_dir)
                        new_x = x[i] + (dx_dir/norm) * step
                        new_y = y[i] + (dy_dir/norm) * step
                        if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                            x[i] = new_x
                            y[i] = new_y
                            improved = True
                            found = True
                            break
                    if found:
                        break

                if not found:
                    perp_x = -move_y
                    perp_y = move_x
                    for step in [0.03, 0.01, 0.005, 0.002]:
                        for sign in [1.0, -1.0]:
                            new_x = x[i] + sign * perp_x * step
                            new_y = y[i] + sign * perp_y * step
                            if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                                x[i] = new_x
                                y[i] = new_y
                                improved = True
                                found = True
                                break
                        if found:
                            break

        # Smooth gap-aware relaxation to redistribute circles
        if rnd % 3 == 0 and time.time() < time_budget - 2:
            fx = np.zeros(n)
            fy = np.zeros(n)
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            r_sum = r[:, None] + r[None, :]
            iu = np.triu_indices(n, k=1)
            d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
            gap = d_pairs - r_sum[iu]
            soft_force = 0.5 * np.exp(-np.maximum(gap, 0) / 0.03)
            overlap_force = np.maximum(-gap, 0) * 20.0
            f_pairs = soft_force + overlap_force
            active = f_pairs > 0.005
            for idx in np.where(active)[0]:
                ii = iu[0][idx]
                jj = iu[1][idx]
                d = d_pairs[idx]
                if d > 1e-6:
                    ux = dx[ii, jj] / d
                    uy = dy[ii, jj] / d
                    f = f_pairs[idx]
                    fx[ii] += ux * f
                    fy[ii] += uy * f
                    fx[jj] -= ux * f
                    fy[jj] -= uy * f
            fx -= np.maximum(r - x, 0) * 30
            fx += np.maximum(x + r - w, 0) * 30
            fy -= np.maximum(r - y, 0) * 30
            fy += np.maximum(y + r - h, 0) * 30
            x += 0.0005 * fx
            y += 0.0005 * fy
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

        if not improved:
            break

    # Final uniform expansion with binary search
    for _ in range(150):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.02
        for _bs in range(20):
            mid = (lo + hi) / 2.0
            r_trial = r * mid
            if _is_valid(x, y, r_trial, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-8:
            r = r * lo
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

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3 * i] - p[3 * i + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3 * i] - p[3 * i + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3 * i + 1] - p[3 * i + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3 * i + 1] - p[3 * i + 2]})

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (
                    (p[3 * i] - p[3 * j]) ** 2 +
                    (p[3 * i + 1] - p[3 * j + 1]) ** 2 -
                    (p[3 * i + 2] + p[3 * j + 2]) ** 2
                )
            })

    maxiter = int(min(800, time_budget * 150))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
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
        return None
    except Exception:
        return None


def _simulated_annealing_polish(z, time_budget):
    """Simulated annealing to escape local optima in circle packing.
    
    This is a STRUCTURAL change: instead of deterministic gradient-based methods,
    we use stochastic perturbation with acceptance probability to explore the
    solution space more broadly before exploiting.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    current_val = float(np.sum(r))
    best_val = current_val
    best_x = x.copy()
    best_y = y.copy()
    best_r = r.copy()

    # Annealing parameters
    T_start = 0.05
    T_end = 0.001
    cooling_rate = 0.995
    max_iter = int(min(3000, time_budget * 100))
    rng = np.random.default_rng(123)

    T = T_start
    for it in range(max_iter):
        if time.time() > time_budget:
            break

        # Random perturbation: move 1-3 circles
        n_moves = rng.integers(1, 4)
        indices = rng.choice(n, size=n_moves, replace=False)

        x_trial = x.copy()
        y_trial = y.copy()
        r_trial = r.copy()

        for idx in indices:
            # Perturb position
            dx = rng.uniform(-0.05, 0.05)
            dy = rng.uniform(-0.05, 0.05)
            x_trial[idx] += dx
            y_trial[idx] += dy
            x_trial[idx] = np.clip(x_trial[idx], r_trial[idx], w - r_trial[idx])
            y_trial[idx] = np.clip(y_trial[idx], r_trial[idx], h - r_trial[idx])

            # Try to grow radius slightly
            growth = rng.uniform(1.0, 1.02)
            r_trial[idx] *= growth
            r_wall = min(x_trial[idx], w - x_trial[idx], y_trial[idx], h - y_trial[idx])
            r_trial[idx] = min(r_trial[idx], r_wall)

        # Check validity
        valid = _is_valid(x_trial, y_trial, r_trial, w, h)
        if valid:
            trial_val = float(np.sum(r_trial))
            delta = trial_val - current_val

            # Accept if better, or with probability exp(delta/T) if worse
            if delta > 0 or rng.random() < np.exp(delta / T):
                x = x_trial
                y = y_trial
                r = r_trial
                current_val = trial_val

                if current_val > best_val:
                    best_val = current_val
                    best_x = x.copy()
                    best_y = y.copy()
                    best_r = r.copy()
        else:
            # If invalid, shrink radii to make valid
            for scale in [0.95, 0.9, 0.85, 0.8]:
                r_shrink = r_trial * scale
                if _is_valid(x_trial, y_trial, r_shrink, w, h):
                    trial_val = float(np.sum(r_shrink))
                    delta = trial_val - current_val
                    if delta > 0 or rng.random() < np.exp(delta / T):
                        x = x_trial
                        y = y_trial
                        r = r_shrink
                        current_val = trial_val
                        if current_val > best_val:
                            best_val = current_val
                            best_x = x.copy()
                            best_y = y.copy()
                            best_r = r.copy()
                    break

        T *= cooling_rate

    # Final polish on best solution
    z_best = np.empty(3 * N + 1)
    z_best[0::3][:n] = best_x
    z_best[1::3][:n] = best_y
    z_best[2::3][:n] = best_r
    z_best[-1] = w

    if best_val > float(np.sum(r)) - 1e-9:
        return z_best
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

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

        z0_relaxed = _force_relax_unified(z0, max_iter=500)

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
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            continue

    # --- Phase 2: Polish on best result ---
    if best_result is not None and time.time() < deadline - 30:
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 20)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        if best_result is not None and time.time() < deadline - 20:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 12)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # NEW: Simulated annealing to escape local optima
        if best_result is not None and time.time() < deadline - 15:
            sa_result = _simulated_annealing_polish(best_result, deadline - (time.time() - start_time) - 10)
            if sa_result is not None:
                valid, val = check_validity(sa_result)
                if valid and val > best_value:
                    best_value = val
                    best_result = sa_result

        if best_result is not None and time.time() < deadline - 12:
            polished2 = _polish_result(best_result, deadline - (time.time() - start_time) - 8)
            if polished2 is not None:
                valid, val = check_validity(polished2)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished2

        if best_result is not None and time.time() < deadline - 10:
            expanded2 = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 6)
            if expanded2 is not None:
                valid, val = check_validity(expanded2)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded2

        if best_result is not None and time.time() < deadline - 8:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 4)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        if best_result is not None and time.time() < deadline - 4:
            polished3 = _polish_result(best_result, deadline - (time.time() - start_time))
            if polished3 is not None:
                valid, val = check_validity(polished3)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished3

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
