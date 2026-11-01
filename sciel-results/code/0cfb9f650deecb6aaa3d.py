"""Circle packing in perimeter-4 rectangle: SLSQP + Epigraph + Adaptive Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
_LSE_K = 400.0  # temperature for log-sum-exp (smaller = sharper)


def _soft_max(v, beta=200.0):
    m = np.maximum(v, 0.0)
    return m + np.logaddexp(0.0, beta * (v - m)) / beta


def _lse(vals):
    """Numerically-stable log-sum-exp."""
    m = np.max(vals)
    if not np.isfinite(m):
        return m
    return m + np.log(np.sum(np.exp(vals - m)))


def _vectorized_objective(z, penalty_weight=1000.0):
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

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap = _soft_max(r_sum_pairs - d_pairs, beta=200.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    left_viol = _soft_max(r - x, beta=200.0)
    right_viol = _soft_max(x + r - w, beta=200.0)
    bottom_viol = _soft_max(r - y, beta=200.0)
    top_viol = _soft_max(y + r - h, beta=200.0)

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
    if new_x < r[i] - 1e-9 or new_x > w - r[i] + 1e-9:
        return False
    if new_y < r[i] - 1e-9 or new_y > h - r[i] + 1e-9:
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


def _max_radius_for_circle(x, y, r, w, h, i):
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)


def _force_relax(z, max_iter=500):
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001

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

        if max_overlap > 1e-8:
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
        else:
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


def _polish_positions(x, y, r, w, h, time_budget, dt=0.001, wall_k=35, pair_k=18, max_iter=3000):
    n = N
    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * wall_k
        fx += right_viol * wall_k
        fy -= bottom_viol * wall_k
        fy += top_viol * wall_k

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
            f_pairs = overlap * pair_k
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
            dt_use = 0.0005
        else:
            dt_use = dt

        x = x + dt_use * fx
        y = y + dt_use * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    return x, y


def _expand_radii(x, y, r, w, h, time_budget):
    n = N
    for _ in range(500):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0003
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    for _ in range(80):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            r_max = _max_radius_for_circle(x, y, r, w, h, i)
            if r_max > r[i] + 1e-7:
                lo = r[i]
                hi = r_max
                for _ in range(25):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True

            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max_now = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max_now:
                    r_max_now = r_allowed
            r_max_now = max(0.0, r_max_now)

            if r_max_now > r[i] + 1e-6:
                best_pos = None
                best_r = r_max_now
                for dx_m in np.linspace(-0.05, 0.05, 11):
                    for dy_m in np.linspace(-0.05, 0.05, 11):
                        nx = x[i] + dx_m
                        ny = y[i] + dy_m
                        if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                            continue
                        if not _is_valid_move(x, y, r, w, h, i, nx, ny):
                            continue
                        rw = min(nx, w - nx, ny, h - ny)
                        rm = rw
                        for j in range(n):
                            if i == j:
                                continue
                            dx2 = nx - x[j]
                            dy2 = ny - y[j]
                            d2 = np.sqrt(dx2 * dx2 + dy2 * dy2)
                            ra = d2 - r[j]
                            if ra < rm:
                                rm = ra
                        rm = max(0.0, rm)
                        if rm > best_r + 1e-6:
                            best_r = rm
                            best_pos = (nx, ny)

                if best_pos is not None:
                    x[i], y[i] = best_pos
                    r_max = _max_radius_for_circle(x, y, r, w, h, i)
                    if r_max > r[i] + 1e-7:
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
                        if lo > r[i] + 1e-8:
                            r[i] = lo
                            improved = True

        if not improved:
            break

    return x, y, r


def _adaptive_local_search(x, y, r, w, h, time_budget):
    """Adaptive 16-direction local search with geometric step decay per circle."""
    n = N
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ux = np.cos(angles)
    uy = np.sin(angles)
    base_step = 0.008

    for _ in range(3):  # multiple outer passes
        if time.time() > time_budget:
            break
        improved_any = False
        for i in range(n):
            if time.time() > time_budget:
                break
            step = base_step
            for _decay in range(12):
                if step < 1e-5:
                    break
                found = False
                for a in range(16):
                    nx = x[i] + ux[a] * step
                    ny = y[i] + uy[a] * step
                    if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                        continue
                    if not _is_valid_move(x, y, r, w, h, i, nx, ny):
                        continue
                    # Does moving here allow a larger radius?
                    r_new = _max_radius_for_circle(x, y, r, w, h, i)
                    # evaluate gain: use new position's max radius
                    r_wall = min(nx, w - nx, ny, h - ny)
                    r_allow = r_wall
                    for j in range(n):
                        if i == j:
                            continue
                        dx2 = nx - x[j]
                        dy2 = ny - y[j]
                        d2 = np.sqrt(dx2 * dx2 + dy2 * dy2)
                        ra = d2 - r[j]
                        if ra < r_allow:
                            r_allow = ra
                    r_allow = max(0.0, r_allow)
                    if r_allow > r[i] + 1e-6:
                        x[i], y[i] = nx, ny
                        # expand radius to the allowed amount
                        lo = r[i]
                        hi = r_allow
                        for _ in range(20):
                            mid = (lo + hi) / 2.0
                            r_test = r.copy()
                            r_test[i] = mid
                            if _is_valid(x, y, r_test, w, h):
                                lo = mid
                            else:
                                hi = mid
                        if lo > r[i] + 1e-9:
                            r[i] = lo
                        found = True
                        improved_any = True
                        break
                if found:
                    break
                step *= 0.5
        if not improved_any:
            break

    return x, y, r


def _high_precision_radius_expansion(x, y, r, w, h, time_budget):
    n = N
    max_passes = 50
    for pass_num in range(max_passes):
        if time.time() > time_budget:
            break
        any_improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            r_max = _max_radius_for_circle(x, y, r, w, h, i)
            if r_max <= r[i] + 1e-12:
                continue
            lo = r[i]
            hi = r_max
            for _ in range(40):
                mid = (lo + hi) / 2.0
                r_test = r.copy()
                r_test[i] = mid
                if _is_valid(x, y, r_test, w, h):
                    lo = mid
                else:
                    hi = mid
            if lo > r[i] + 1e-13:
                r[i] = lo
                any_improved = True
        if not any_improved:
            break
    return x, y, r


def _epigraph_refine(z, time_budget):
    """Epigraph formulation: introduce t = max radius constraint, smooth min via LSE."""
    z = z.copy().astype(float)
    n = N

    def obj_ep(zv):
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w
        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6
        # maximize sum r
        obj = -np.sum(r)
        # smooth penalty on overlaps using LSE of (r_i + r_j - d)^2
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        viol = np.maximum(r_sum[iu] - d_pairs, 0.0)
        obj += 1000.0 * np.sum(viol * viol)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        obj += 1000.0 * (np.sum(left_viol * left_viol) + np.sum(right_viol * right_viol) +
                         np.sum(bottom_viol * bottom_viol) + np.sum(top_viol * top_viol))
        return obj

    bounds = [(0, 2)] * (3 * N) + [(0.3, 1.7)]
    maxiter = int(min(700, max(150, time_budget * 40)))
    try:
        res = minimize(obj_ep, z, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'maxls': 80})
        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            return z_opt
    except Exception:
        pass
    return None


def _constraint_refine(z, time_budget):
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
        constraints.append({'type': 'ineq', 'fun': (lambda p, i=i: p[3 * i] - p[3 * i + 2])})
        constraints.append({'type': 'ineq', 'fun': (lambda p, i=i: p[-1] - p[3 * i] - p[3 * i + 2])})
        constraints.append({'type': 'ineq', 'fun': (lambda p, i=i: p[3 * i + 1] - p[3 * i + 2])})
        constraints.append({'type': 'ineq', 'fun': (lambda p, i=i: (2.0 - p[-1]) - p[3 * i + 1] - p[3 * i + 2])})

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({'type': 'ineq', 'fun': (
                lambda p, i=i, j=j: ((p[3 * i] - p[3 * j]) ** 2 +
                                     (p[3 * i + 1] - p[3 * j + 1]) ** 2 -
                                     (p[3 * i + 2] + p[3 * j + 2]) ** 2))})

    maxiter = int(min(800, max(150, time_budget * 50)))
    try:
        res = minimize(obj_refine, z, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'maxls': 100})
        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            return z_opt
    except Exception:
        pass
    return None


def _greedy_expansion(z, time_budget):
    """Final greedy per-circle expansion with tight tolerance."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    x, y, r = _high_precision_radius_expansion(x, y, r, w, h, time_budget)
    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _jiggle_and_grow(x, y, r, w, h, time_budget, max_jiggles=500):
    """
    Structural change: Jiggle and Grow.
    1. Try to scale up radii.
    2. If invalid, apply small random perturbations (jiggle) and relax.
    3. Repeat.
    """
    n = N
    best_val = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    # Initial check
    if not _is_valid(x, y, r, w, h):
        # Try to repair first
        z = np.empty(3 * N + 1)
        z[0::3][:n] = x
        z[1::3][:n] = y
        z[2::3][:n] = r
        z[-1] = w
        z = _force_relax(z, max_iter=200)
        x = z[0::3][:n].copy()
        y = z[1::3][:n].copy()
        r = z[2::3][:n].copy()
        if not _is_valid(x, y, r, w, h):
            return best_x, best_y, best_r, best_val

    scale_factor = 1.0
    for _ in range(max_jiggles):
        if time.time() > time_budget:
            break
            
        # Try to grow
        new_r = r * (1.0 + 0.005)
        if _is_valid(x, y, new_r, w, h):
            r = new_r
            best_r = r.copy()
            best_x = x.copy()
            best_y = y.copy()
            best_val = np.sum(r)
        else:
            # Jiggle
            # Random perturbations
            perturb_scale = 0.01
            dx = np.random.uniform(-perturb_scale, perturb_scale, n)
            dy = np.random.uniform(-perturb_scale, perturb_scale, n)
            new_x = x + dx
            new_y = y + dy
            
            # Clip to boundaries
            new_x = np.clip(new_x, r, w - r)
            new_y = np.clip(new_y, r, h - r)
            
            # Relax
            z = np.empty(3 * N + 1)
            z[0::3][:n] = new_x
            z[1::3][:n] = new_y
            z[2::3][:n] = r
            z[-1] = w
            z = _force_relax(z, max_iter=100)
            
            x = z[0::3][:n].copy()
            y = z[1::3][:n].copy()
            r_curr = z[2::3][:n].copy()
            
            # If relaxation reduced radii, keep them? No, we want to grow.
            # We only update x,y if valid.
            if _is_valid(x, y, r, w, h):
                # Keep current r, update x,y
                pass
            else:
                # Revert to best valid state
                x = best_x.copy()
                y = best_y.copy()
                r = best_r.copy()
                
    return best_x, best_y, best_r, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

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
        return True, float(np.sum(r))

    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))
        # More diverse aspect ratios
        aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 2.0, 0.4, 2.2, 0.35]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                z[3 * i] = (col + 0.5) / side * w0
                z[3 * i + 1] = (row + 0.5) / side * h0
                z[3 * i + 2] = min(w0 / side, h0 / side) * 0.4
            z[-1] = w0
            guesses.append(z)
            
        # Random guesses
        rng = np.random.default_rng(42)
        for _ in range(15):
            w0 = rng.uniform(0.5, 1.5)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.03, 0.1)
                z[3 * i] = rng.uniform(r0, w0 - r0)
                z[3 * i + 1] = rng.uniform(r0, h0 - r0)
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)
        return guesses

    guesses = get_initial_guesses()

    # Phase 1: Global Search with Nelder-Mead
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break
            
        # Pre-relax
        z0_relaxed = _force_relax(z0, max_iter=200)
        
        # Nelder-Mead Optimization
        def nm_objective(z):
            return _vectorized_objective(z, penalty_weight=100.0)
            
        try:
            res = minimize(nm_objective, z0_relaxed, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            z_opt = res.x
            
            # Post-relax to fix any small violations
            z_opt = _force_relax(z_opt, max_iter=500)
            
            valid, val = check_validity(z_opt)
            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
        except Exception:
            continue

    # Phase 2: Jiggle and Grow
    if best_result is not None:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w
        
        # Run Jiggle and Grow with remaining time
        time_left = deadline - (time.time() - start_time) - 10
        if time_left > 5:
            x, y, r, val = _jiggle_and_grow(x, y, r, w, h, deadline - 10)
            if val > best_value:
                best_value = val
                z_new = np.empty(3 * N + 1)
                z_new[0::3][:N] = x
                z_new[1::3][:N] = y
                z_new[2::3][:N] = r
                z_new[-1] = w
                best_result = z_new

    # Phase 3: Final Polish (SLSQP + Expansion)
    if best_result is not None and time.time() < deadline - 5:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w
        
        # SLSQP Refine
        def slsqp_objective(z):
            return _vectorized_objective(z, penalty_weight=1000.0)
            
        try:
            res = minimize(slsqp_objective, best_result, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': 500, 'ftol': 1e-12})
            z_opt = res.x
            z_opt = _force_relax(z_opt, max_iter=200)
            valid, val = check_validity(z_opt)
            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
        except Exception:
            pass

        # Final Expansion
        if best_result is not None:
            x = best_result[0::3][:N].copy()
            y = best_result[1::3][:N].copy()
            r = best_result[2::3][:N].copy()
            w = best_result[-1]
            h = 2.0 - w
            x, y, r = _high_precision_radius_expansion(x, y, r, w, h, deadline - 1)
            if _is_valid(x, y, r, w, h):
                z_new = np.empty(3 * N + 1)
                z_new[0::3][:N] = x
                z_new[1::3][:N] = y
                z_new[2::3][:N] = r
                z_new[-1] = w
                valid, val = check_validity(z_new)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_new

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Safety shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
