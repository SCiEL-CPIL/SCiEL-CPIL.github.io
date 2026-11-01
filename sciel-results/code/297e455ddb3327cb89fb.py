"""Circle packing in perimeter-4 rectangle: SLSQP + Epigraph + Constraint + Greedy Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
_LSE_K = 500.0  # temperature for log-sum-exp (smaller = sharper min)


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

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)
    obj += penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _lse(vals):
    """Numerically-stable log-sum-exp of an array."""
    m = np.max(vals)
    if not np.isfinite(m):
        return m
    return m + np.log(np.sum(np.exp(vals - m)))


def construct():
    start_time = time.time()
    deadline = start_time + 143.0  # safety margin before 150s kill

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
        return True, float(np.sum(r))

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
                z[3 * i] = (col + 0.5) / side * w0
                z[3 * i + 1] = (row + 0.5) / side * h0
                z[3 * i + 2] = min(w0 / side, h0 / side) * 0.45
            z[-1] = w0
            guesses.append(z)
        rng = np.random.default_rng(42)
        for _ in range(10):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                z[3 * i] = rng.uniform(r0, w0 - r0)
                z[3 * i + 1] = rng.uniform(r0, h0 - r0)
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)
        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 22:
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

    if best_result is not None and time.time() < deadline - 25:
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 15)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        if best_result is not None and time.time() < deadline - 22:
            expanded = _expansion_relaxation_polish_enhanced(best_result, deadline - (time.time() - start_time) - 12)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # Smooth epigraph refine: better-conditioned, escapes local minima
        if best_result is not None and time.time() < deadline - 18:
            ep = _epigraph_refine(best_result, deadline - (time.time() - start_time) - 10)
            if ep is not None:
                valid, val = check_validity(ep)
                if valid and val > best_value:
                    best_value = val
                    best_result = ep

        # Constraint refine: harvest gains with exact constraints
        if best_result is not None and time.time() < deadline - 10:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 5)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # Final greedy per-circle expansion: squeeze out last gains
        if best_result is not None and time.time() < deadline - 3:
            greedy = _greedy_expansion(best_result, deadline - (time.time() - start_time))
            if greedy is not None:
                valid, val = check_validity(greedy)
                if valid and val > best_value:
                    best_value = val
                    best_result = greedy

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

    for _ in range(20):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))


def _force_relax_unified(z, max_iter=500):
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
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.002
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
        fx -= left_viol * 40
        fx += right_viol * 40
        fy -= bottom_viol * 40
        fy += top_viol * 40

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-12:
            f_pairs = overlap * 20.0
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
        elif max_force > 0.01:
            dt = 0.001
        else:
            dt = 0.002

        x_new = x + dt * fx
        y_new = y + dt * fy
        x_new = np.clip(x_new, r, w - r)
        y_new = np.clip(y_new, r, h - r)

        if _is_valid(x_new, y_new, r, w, h):
            x = x_new
            y = y_new

    for _ in range(100):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.05
        for _ in range(15):
            mid = (lo + hi) / 2.0
            if _is_valid(x, y, r * mid, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-9:
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


def _expansion_relaxation_polish_enhanced(z, time_budget):
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    dirs = [(np.cos(a), np.sin(a)) for a in angles]

    def try_expand_circle(i):
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
            for _ in range(30):
                mid = (lo + hi) / 2.0
                r_test = r.copy()
                r_test[i] = mid
                if _is_valid(x, y, r_test, w, h):
                    lo = mid
                else:
                    hi = mid
            if lo > r[i] + 1e-7:
                r[i] = lo
                return True
        return False

    def try_move_circle(i):
        best_dir = None
        best_score = -float('inf')

        extra_dirs = [
            (0.5, 0.866), (-0.5, 0.866), (0.866, 0.5), (0.866, -0.5),
            (-0.5, -0.866), (0.5, -0.866), (-0.866, 0.5), (-0.866, -0.5),
            (0.3, 0.954), (-0.3, 0.954), (0.954, 0.3), (0.954, -0.3),
            (-0.3, -0.954), (0.3, -0.954), (-0.954, 0.3), (-0.954, -0.3),
            (0.1, 0.995), (-0.1, 0.995), (0.995, 0.1), (0.995, -0.1),
            (-0.1, -0.995), (0.1, -0.995), (-0.995, 0.1), (-0.995, -0.1),
        ]
        all_dirs = dirs + extra_dirs

        for dx_d, dy_d in all_dirs:
            new_x = x[i] + dx_d * 0.01
            new_y = y[i] + dy_d * 0.01
            if new_x < r[i] or new_x > w - r[i] or new_y < r[i] or new_y > h - r[i]:
                continue

            min_gap = float('inf')
            for j in range(n):
                if i == j:
                    continue
                dx = new_x - x[j]
                dy = new_y - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_gap:
                    min_gap = gap

            wall_dist = min(new_x - r[i], w - new_x - r[i], new_y - r[i], h - new_y - r[i])
            score = min(min_gap, wall_dist)

            if score > best_score:
                best_score = score
                best_dir = (dx_d, dy_d)

        if best_dir is None:
            return False

        dx_d, dy_d = best_dir

        for step in [0.06, 0.04, 0.03, 0.02, 0.015, 0.01, 0.005, 0.003, 0.001]:
            new_x = x[i] + dx_d * step
            new_y = y[i] + dy_d * step
            if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                x[i] = new_x
                y[i] = new_y
                return True

        return False

    def _soft_repulsion_relax(x_curr, y_curr, r_curr, w_curr, h_curr, max_iter=200):
        x_r = x_curr.copy()
        y_r = y_curr.copy()
        dt = 0.002

        for it in range(max_iter):
            if time.time() > time_budget:
                break

            fx = np.zeros(n)
            fy = np.zeros(n)

            left_viol = np.maximum(r_curr - x_r, 0.0)
            right_viol = np.maximum(x_r + r_curr - w_curr, 0.0)
            bottom_viol = np.maximum(r_curr - y_r, 0.0)
            top_viol = np.maximum(y_r + r_curr - h_curr, 0.0)
            fx -= left_viol * 40
            fx += right_viol * 40
            fy -= bottom_viol * 40
            fy += top_viol * 40

            dx = x_r[:, None] - x_r[None, :]
            dy = y_r[:, None] - y_r[None, :]
            d_sq = dx * dx + dy * dy
            r_sum = r_curr[:, None] + r_curr[None, :]

            iu = np.triu_indices(n, k=1)
            d_sq_pairs = d_sq[iu]
            r_sum_pairs = r_sum[iu]

            d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
            overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

            gap = d_pairs - r_sum_pairs
            close = (gap < 0.02) & (gap >= 0)
            push_force = np.where(close, 0.5 * (0.02 - gap), 0.0)

            total_force = overlap * 20.0 + push_force * 5.0

            if np.max(total_force) > 1e-12:
                dx_pairs = dx[iu]
                dy_pairs = dy[iu]
                ux = dx_pairs / d_pairs
                uy = dy_pairs / d_pairs

                for idx in range(len(iu[0])):
                    i_idx = iu[0][idx]
                    j_idx = iu[1][idx]
                    fx[i_idx] += ux[idx] * total_force[idx]
                    fy[i_idx] += uy[idx] * total_force[idx]
                    fx[j_idx] -= ux[idx] * total_force[idx]
                    fy[j_idx] -= uy[idx] * total_force[idx]

            max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
            if max_force > 0.1:
                dt = 0.0005
            elif max_force > 0.01:
                dt = 0.001
            else:
                dt = 0.002

            x_new = x_r + dt * fx
            y_new = y_r + dt * fy
            x_new = np.clip(x_new, r_curr, w_curr - r_curr)
            y_new = np.clip(y_new, r_curr, h_curr - r_curr)

            if _is_valid(x_new, y_new, r_curr, w_curr, h_curr):
                x_r = x_new
                y_r = y_new
            else:
                break

        return x_r, y_r

    max_rounds = 50
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break

            if try_expand_circle(i):
                improved = True

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

            if r_max - r[i] < 0.01:
                if try_move_circle(i):
                    improved = True

        if rnd % 5 == 4:
            x_relaxed, y_relaxed = _soft_repulsion_relax(x, y, r, w, h, max_iter=100)
            if _is_valid(x_relaxed, y_relaxed, r, w, h):
                x = x_relaxed
                y = y_relaxed

        if not improved:
            break

    for rnd in range(30):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            if try_expand_circle(i):
                improved = True

        if not improved:
            break

    if time.time() < time_budget - 5:
        x_relaxed, y_relaxed = _soft_repulsion_relax(x, y, r, w, h, max_iter=300)
        if _is_valid(x_relaxed, y_relaxed, r, w, h):
            x = x_relaxed
            y = y_relaxed

        for rnd in range(20):
            if time.time() > time_budget:
                break
            improved = False
            for i in range(n):
                if time.time() > time_budget:
                    break
                if try_expand_circle(i):
                    improved = True
            if not improved:
                break

    for _ in range(150):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.05
        for _ in range(20):
            mid = (lo + hi) / 2.0
            if _is_valid(x, y, r * mid, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-9:
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


def _epigraph_refine(z, time_budget):
    """
    SLSQP refinement using epigraph formulation with log-sum-exp penalty.
    This provides a smoother landscape than sum-of-squares for the worst constraint.
    """
    z = z.copy().astype(float)
    n = N

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    def obj_epigraph(zv):
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w

        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6

        obj = -np.sum(r)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = r_sum_pairs - d_pairs

        left_viol = r - x
        right_viol = x + r - w
        bottom_viol = r - y
        top_viol = y + r - h

        all_viol = np.concatenate([overlap, left_viol, right_viol, bottom_viol, top_viol])

        # Log-sum-exp for smooth max penalty
        max_viol = _lse(all_viol / _LSE_K) * _LSE_K
        if max_viol > 0:
            obj += 5000.0 * max_viol * max_viol

        pos_viol = np.maximum(all_viol, 0.0)
        obj += 100.0 * np.sum(pos_viol * pos_viol)

        return obj

    maxiter = int(min(1500, time_budget * 150))
    try:
        res = minimize(obj_epigraph, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 150})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if not _is_valid(x, y, r, w, h):
            for factor in [0.999, 0.998, 0.997, 0.995, 0.99]:
                r_trial = r * factor
                if _is_valid(x, y, r_trial, w, h):
                    r = r_trial
                    break

        if _is_valid(x, y, r, w, h):
            z_opt[2::3][:n] = r
            return z_opt
        return None
    except Exception:
        return None


def _constraint_refine(z, time_budget, tight=False):
    """
    High-precision SLSQP refinement using smooth penalty + epigraph formulation.
    Uses log-sum-exp for the worst constraint to provide a smoother landscape.
    """
    z = z.copy().astype(float)
    n = N

    penalty = 1500.0 if tight else 800.0
    ftol = 1e-18 if tight else 1e-16
    maxls = 250 if tight else 180

    def obj_refine(zv):
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w

        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6

        obj = -np.sum(r)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = r_sum_pairs - d_pairs

        left_viol = r - x
        right_viol = x + r - w
        bottom_viol = r - y
        top_viol = y + r - h

        all_viol = np.concatenate([overlap, left_viol, right_viol, bottom_viol, top_viol])

        max_viol = _lse(all_viol / _LSE_K) * _LSE_K
        if max_viol > 0:
            obj += penalty * max_viol * max_viol

        pos_viol = np.maximum(all_viol, 0.0)
        obj += 150.0 * np.sum(pos_viol * pos_viol)

        return obj

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    maxiter = int(min(3000 if tight else 1500, time_budget * 250))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': ftol, 'maxls': maxls})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if not _is_valid(x, y, r, w, h):
            for factor in [0.9995, 0.999, 0.998, 0.997, 0.995, 0.99]:
                r_trial = r * factor
                if _is_valid(x, y, r_trial, w, h):
                    r = r_trial
                    break

        if _is_valid(x, y, r, w, h):
            z_opt[2::3][:n] = r
            return z_opt
        return None
    except Exception:
        return None


def _greedy_expansion(z, time_budget):
    """
    Final greedy per-circle expansion: for each circle, try to increase its radius
    by optimizing its position using adaptive local search with finer steps.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    def max_radius_at_position(px, py, other_r, skip_i):
        r_wall = min(px, w - px, py, h - py)
        r_max = r_wall
        for j in range(n):
            if j == skip_i:
                continue
            dx = px - x[j]
            dy = py - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - other_r[j]
            if r_allowed < r_max:
                r_max = r_allowed
        return max(0.0, r_max)

    step_scales = [1.0, 0.5, 0.25, 0.1]
    base_steps = [0.04, 0.02, 0.01, 0.005, 0.002, 0.001]

    for rnd in range(40):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break

            best_r = r[i]
            best_x = x[i]
            best_y = y[i]

            for scale in step_scales:
                for dx_p in [-1, 0, 1]:
                    for dy_p in [-1, 0, 1]:
                        if dx_p == 0 and dy_p == 0:
                            continue
                        step = base_steps[min(abs(dx_p) + abs(dy_p) - 1, len(base_steps) - 1)] * scale
                        px = x[i] + dx_p * step
                        py = y[i] + dy_p * step
                        if px < 0 or px > w or py < 0 or py > h:
                            continue

                        r_test = max_radius_at_position(px, py, r, i)
                        if r_test > best_r + 1e-9:
                            r_new = r.copy()
                            r_new[i] = r_test
                            x_new = x.copy()
                            y_new = y.copy()
                            x_new[i] = px
                            y_new[i] = py
                            if _is_valid(x_new, y_new, r_new, w, h):
                                best_r = r_test
                                best_x = px
                                best_y = py

            if best_r > r[i] + 1e-9:
                x[i] = best_x
                y[i] = best_y
                r[i] = best_r
                improved = True

        if not improved:
            break

    for _ in range(100):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.03
        for _ in range(18):
            mid = (lo + hi) / 2.0
            if _is_valid(x, y, r * mid, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-9:
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


def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
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
