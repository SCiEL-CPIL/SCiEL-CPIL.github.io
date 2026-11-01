"""Circle packing in perimeter-4 rectangle: SLSQP + LSE Refine + Coordinate Descent."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
_IU = np.triu_indices(N, k=1)


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Vectorized validity check for a packing."""
    if W <= 1e-6 or H <= 1e-6:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    dx = cx[_IU[0]] - cx[_IU[1]]
    dy = cy[_IU[0]] - cy[_IU[1]]
    d2 = dx * dx + dy * dy
    rs = r[_IU[0]] + r[_IU[1]]
    if np.any(d2 < rs * rs - tol):
        return False
    return True


def _is_valid_move(cx, cy, r, W, H, i, nx, ny):
    """Check if moving circle i to (nx, ny) keeps the packing valid."""
    if nx < r[i] - 1e-9 or nx > W - r[i] + 1e-9:
        return False
    if ny < r[i] - 1e-9 or ny > H - r[i] + 1e-9:
        return False
    dx = nx - np.delete(cx, i)
    dy = ny - np.delete(cy, i)
    d2 = dx * dx + dy * dy
    rs = r[i] + np.delete(r, i)
    if np.any(d2 < rs * rs - 1e-9):
        return False
    return True


def _max_radius_for(i, cx, cy, r, W, H):
    """Max radius circle i can have at its current position."""
    r_wall = min(cx[i], W - cx[i], cy[i], H - cy[i])
    r_max = r_wall
    if N > 1:
        dx = cx[i] - np.delete(cx, i)
        dy = cy[i] - np.delete(cy, i)
        d = np.sqrt(dx * dx + dy * dy)
        r_allowed = d - np.delete(r, i)
        r_max = min(r_max, float(np.min(r_allowed)))
    return max(0.0, r_max)


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

    dx = x[_IU[0]] - x[_IU[1]]
    dy = y[_IU[0]] - y[_IU[1]]
    d_pairs = np.sqrt(np.maximum(dx * dx + dy * dy, 0.0))
    r_sum_pairs = r[_IU[0]] + r[_IU[1]]
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)
    obj += penalty_weight * (
        np.sum(left_viol * left_viol) + np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) + np.sum(top_viol * top_viol)
    )

    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _force_relax(z, max_iter=400):
    """Force-directed relaxation to resolve initial overlaps."""
    z = z.copy().astype(float)
    n = N
    dt = 0.004
    min_dt = 1e-4

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

        fx -= np.maximum(r - x, 0.0) * 100
        fx += np.maximum(x + r - w, 0.0) * 100
        fy -= np.maximum(r - y, 0.0) * 100
        fy += np.maximum(y + r - h, 0.0) * 100

        dx = x[_IU[0]] - x[_IU[1]]
        dy = y[_IU[0]] - y[_IU[1]]
        d = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
        overlap = np.maximum(r[_IU[0]] + r[_IU[1]] - d, 0.0)
        if overlap.size and np.max(overlap) > 1e-8:
            f = overlap * 15.0
            ux = dx / d
            uy = dy / d
            np.add.at(fx, _IU[0], ux * f)
            np.add.at(fy, _IU[0], uy * f)
            np.add.at(fx, _IU[1], -ux * f)
            np.add.at(fy, _IU[1], -uy * f)
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


def _get_initial_guesses():
    """Hex-like equal-radius grid layouts at several aspect ratios + random starts."""
    guesses = []

    def hex_grid(w0, rows, cols):
        h0 = 2.0 - w0
        if h0 < 0.3:
            return None
        z = np.zeros(3 * N + 1)
        placed = 0
        for row in range(rows):
            y_c = (row + 0.5) / rows * h0
            offset = 0.5 / cols * w0 if (row % 2 == 1) else 0.0
            for col in range(cols):
                if placed >= N:
                    break
                x_c = (col + 0.5) / cols * w0 + offset
                x_c = min(max(x_c, 0.05), w0 - 0.05)
                z[3 * placed] = x_c
                z[3 * placed + 1] = y_c
                z[3 * placed + 2] = 0.05
                placed += 1
        z[-1] = w0
        return z

    # (w0, rows, cols)
    layouts = [
        (1.0, 5, 5),
        (0.9, 5, 5),
        (1.1, 5, 5),
        (0.8, 5, 5),
        (1.2, 5, 5),
        (1.0, 6, 4),
        (1.0, 4, 6),
        (0.7, 5, 5),
        (1.3, 5, 5),
        (0.85, 6, 4),
        (1.15, 4, 6),
        (1.0, 7, 3),
        (1.4, 4, 5),
    ]
    for w0, rows, cols in layouts:
        z = hex_grid(w0, rows, cols)
        if z is not None:
            guesses.append(z)

    rng = np.random.default_rng(7)
    for _ in range(8):
        w0 = rng.uniform(0.7, 1.3)
        h0 = 2.0 - w0
        z = np.zeros(3 * N + 1)
        for i in range(N):
            r0 = rng.uniform(0.04, 0.1)
            z[3 * i] = rng.uniform(r0, w0 - r0)
            z[3 * i + 1] = rng.uniform(r0, h0 - r0)
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)

    return guesses


def _soft_expansion_refine(z, time_budget):
    """Joint position+radius optimization with LSE margin, then per-circle + aspect polish."""
    z = z.copy().astype(float)
    n = N

    def soft_obj(zv):
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w
        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6
        margin = 0.003
        dx = x[_IU[0]] - x[_IU[1]]
        dy = y[_IU[0]] - y[_IU[1]]
        d_pairs = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
        viol = (r[_IU[0]] + r[_IU[1]]) + margin - d_pairs
        wall_viol = np.concatenate([
            np.maximum(r - x, 0.0),
            np.maximum(x + r - w, 0.0),
            np.maximum(r - y, 0.0),
            np.maximum(y + r - h, 0.0),
        ])
        beta = 300.0
        all_viol = np.concatenate([viol, wall_viol])
        m = np.max(all_viol)
        lse = m + np.log(np.sum(np.exp(beta * (all_viol - m)))) / beta
        return -np.sum(r) + 100.0 * lse

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    maxiter = int(min(1200, time_budget * 250))
    try:
        res = minimize(soft_obj, z, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-13, 'maxls': 200})
        z_opt = res.x
        x = z_opt[0::3][:n].copy()
        y = z_opt[1::3][:n].copy()
        r = z_opt[2::3][:n].copy()
        w = z_opt[-1]
        h = 2.0 - w

        if not _is_valid(x, y, r, w, h):
            for factor in [0.999, 0.998, 0.995, 0.99, 0.98]:
                if _is_valid(x, y, r * factor, w, h):
                    r = r * factor
                    break
            if not _is_valid(x, y, r, w, h):
                return None

        # uniform growth
        lo, hi = 1.0, 1.08
        for _ in range(20):
            mid = (lo + hi) / 2.0
            if _is_valid(x, y, r * mid, w, h):
                lo = mid
            else:
                hi = mid
        r = r * lo

        # per-circle expansion
        for i in range(n):
            r_max = _max_radius_for(i, x, y, r, w, h)
            if r_max > r[i] + 1e-7:
                lo_r, hi_r = r[i], r_max
                for _ in range(20):
                    mid_r = (lo_r + hi_r) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid_r
                    if _is_valid(x, y, r_test, w, h):
                        lo_r = mid_r
                    else:
                        hi_r = mid_r
                r[i] = lo_r

        # aspect ratio search
        best_sum = float(np.sum(r))
        for _ in range(4):
            best_w = w
            for w_try in np.linspace(0.5, 1.5, 11):
                h_try = 2.0 - w_try
                if h_try < 0.3:
                    continue
                if _is_valid(x, y, r, w_try, h_try):
                    if float(np.sum(r)) > best_sum:
                        best_sum = float(np.sum(r))
                        best_w = w_try
            w = best_w
            h = 2.0 - w
            # re-expand after aspect change
            for i in range(n):
                r_max = _max_radius_for(i, x, y, r, w, h)
                if r_max > r[i] + 1e-7:
                    lo_r, hi_r = r[i], r_max
                    for _ in range(15):
                        mid_r = (lo_r + hi_r) / 2.0
                        r_test = r.copy()
                        r_test[i] = mid_r
                        if _is_valid(x, y, r_test, w, h):
                            lo_r = mid_r
                        else:
                            hi_r = mid_r
                    r[i] = lo_r

        if _is_valid(x, y, r, w, h):
            z_new = np.empty(3 * N + 1)
            z_new[0::3][:n] = x
            z_new[1::3][:n] = y
            z_new[2::3][:n] = r
            z_new[-1] = w
            return z_new
        return None
    except Exception:
        return None


def _lse_refine(z, time_budget):
    """Smooth LSE surrogate refinement on the full state."""
    z = z.copy().astype(float)
    n = N

    def lse_obj(zv):
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w
        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6
        dx = x[_IU[0]] - x[_IU[1]]
        dy = y[_IU[0]] - y[_IU[1]]
        d_pairs = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
        viol = (r[_IU[0]] + r[_IU[1]]) - d_pairs
        wall_viol = np.concatenate([
            np.maximum(r - x, 0.0),
            np.maximum(x + r - w, 0.0),
            np.maximum(r - y, 0.0),
            np.maximum(y + r - h, 0.0),
        ])
        beta = 200.0
        all_viol = np.concatenate([viol, wall_viol])
        m = np.max(all_viol)
        lse = m + np.log(np.sum(np.exp(beta * (all_viol - m)))) / beta
        return -np.sum(r) + 50.0 * lse

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    maxiter = int(min(1000, time_budget * 250))
    try:
        res = minimize(lse_obj, z, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 150})
        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w
        if not _is_valid(x, y, r, w, h):
            for factor in [0.999, 0.998, 0.997, 0.995, 0.99, 0.98]:
                if _is_valid(x, y, r * factor, w, h):
                    r = r * factor
                    break
        if _is_valid(x, y, r, w, h):
            z_opt[2::3][:n] = r
            return z_opt
        return None
    except Exception:
        return None


def _polish_result(z, time_budget):
    """Force-directed position polish + uniform radius growth."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.002
    max_iter = int(min(4000, time_budget * 200))

    for it in range(max_iter):
        if time.time() > time_budget:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)
        fx -= np.maximum(r - x, 0.0) * 40
        fx += np.maximum(x + r - w, 0.0) * 40
        fy -= np.maximum(r - y, 0.0) * 40
        fy += np.maximum(y + r - h, 0.0) * 40

        dx = x[_IU[0]] - x[_IU[1]]
        dy = y[_IU[0]] - y[_IU[1]]
        d = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
        overlap = np.maximum(r[_IU[0]] + r[_IU[1]] - d, 0.0)
        if overlap.size and np.max(overlap) > 1e-12:
            f = overlap * 20.0
            ux = dx / d
            uy = dy / d
            np.add.at(fx, _IU[0], ux * f)
            np.add.at(fy, _IU[0], uy * f)
            np.add.at(fx, _IU[1], -ux * f)
            np.add.at(fy, _IU[1], -uy * f)

        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        dt = 0.0005 if max_force > 0.1 else (0.001 if max_force > 0.01 else 0.002)

        x_new = np.clip(x + dt * fx, r, w - r)
        y_new = np.clip(y + dt * fy, r, h - r)
        if _is_valid(x_new, y_new, r, w, h):
            x, y = x_new, y_new

    # uniform growth
    for _ in range(60):
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

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _coordinate_descent_polish(z, time_budget):
    """Per-circle (x,y,r) coordinate descent with position shifts + aspect search."""
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
            if time.time() > time_budget:
                break
            r_max = _max_radius_for(i, x, y, r, w, h)
            if r_max > r[i] + 1e-10:
                lo, hi = r[i], r_max
                for _ in range(22):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-10:
                    r[i] = lo
                    improved = True

            # small position shifts to create room
            for dx_s in [0.004, -0.004, 0.008, -0.008, 0.002, -0.002]:
                for dy_s in [0.004, -0.004, 0.008, -0.008, 0.002, -0.002]:
                    if time.time() > time_budget:
                        break
                    nx = x[i] + dx_s
                    ny = y[i] + dy_s
                    if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                        continue
                    if not _is_valid_move(x, y, r, w, h, i, nx, ny):
                        continue
                    r_wall2 = min(nx, w - nx, ny, h - ny)
                    r_max2 = r_wall2
                    if N > 1:
                        dx2 = nx - np.delete(x, i)
                        dy2 = ny - np.delete(y, i)
                        d2 = np.sqrt(dx2 * dx2 + dy2 * dy2)
                        r_max2 = min(r_max2, float(np.min(d2 - np.delete(r, i))))
                    if max(0.0, r_max2) > r[i] + 1e-8:
                        x[i], y[i] = nx, ny
                        improved = True
                        break

        # aspect search each round
        best_sum = float(np.sum(r))
        for w_try in np.linspace(0.55, 1.45, 7):
            h_try = 2.0 - w_try
            if h_try < 0.3:
                continue
            if _is_valid(x, y, r, w_try, h_try) and float(np.sum(r)) > best_sum:
                best_sum = float(np.sum(r))
                w, h = w_try, h_try

        if not improved and rnd > 0:
            # do one more round after aspect change
            pass
        if not improved:
            # try a couple more rounds in case aspect helped
            if rnd > 2:
                break

    # final uniform growth
    lo, hi = 1.0, 1.03
    for _ in range(15):
        mid = (lo + hi) / 2.0
        if _is_valid(x, y, r, w, h):
            lo = mid
        else:
            hi = mid
    if lo > 1.0 + 1e-9:
        r = r * lo

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
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
        if w <= 0.05 or h <= 0.05 or np.any(r <= 0):
            return False, 0.0
        if _is_valid(x, y, r, w, h, tol):
            return True, float(np.sum(r))
        return False, 0.0

    guesses = _get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 25:
            break
        z0_relaxed = _force_relax(z0, max_iter=400)
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(500, max(120, remaining * 40)))
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

    # --- Phase 2: polish on best result ---
    def try_update(zcand):
        nonlocal best_value, best_result
        if zcand is None:
            return
        valid, val = check_validity(zcand)
        if valid and val > best_value:
            best_value = val
            best_result = zcand

    if best_result is not None:
        if time.time() < deadline - 25:
            try_update(_polish_result(best_result, deadline - (time.time() - start_time) - 15))
        if best_result is not None and time.time() < deadline - 20:
            try_update(_soft_expansion_refine(best_result, deadline - (time.time() - start_time) - 10))
        if best_result is not None and time.time() < deadline - 15:
            try_update(_lse_refine(best_result, deadline - (time.time() - start_time) - 8))
        if best_result is not None and time.time() < deadline - 12:
            try_update(_soft_expansion_refine(best_result, deadline - (time.time() - start_time) - 6))
        if best_result is not None and time.time() < deadline - 8:
            try_update(_coordinate_descent_polish(best_result, deadline - (time.time() - start_time) - 3))
        if best_result is not None and time.time() < deadline - 5:
            try_update(_lse_refine(best_result, deadline - (time.time() - start_time) - 3))
        if best_result is not None and time.time() < deadline - 3:
            try_update(_coordinate_descent_polish(best_result, deadline - (time.time() - start_time) - 1))
        if best_result is not None and time.time() < deadline - 1:
            try_update(_polish_result(best_result, deadline - (time.time() - start_time)))

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

    if time.time() < start_time + 148:
        cd_final = _coordinate_descent_polish(np.concatenate([x, y, r, [w]]), start_time + 148)
        if cd_final is not None:
            x2 = cd_final[0::3][:N]
            y2 = cd_final[1::3][:N]
            r2 = cd_final[2::3][:N]
            w2 = cd_final[-1]
            h2 = 2.0 - w2
            if _is_valid(x2, y2, r2, w2, h2) and np.sum(r2) > np.sum(r):
                x, y, r, w, h = x2, y2, r2, w2, h2

    return np.column_stack((x, y, r))
