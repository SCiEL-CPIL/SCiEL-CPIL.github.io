"""Circle packing in perimeter-4 rectangle: Multi-strategy SLSQP + L-BFGS-B + Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


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

        val = float(np.sum(r))
        return True, val

    def objective_slsqp(z, penalty_weight=500.0):
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

        if w < 0.4:
            obj += penalty_weight * (0.4 - w) ** 2
        elif w > 1.6:
            obj += penalty_weight * (w - 1.6) ** 2

        return obj

    def objective_lbgfsb(z, penalty_weight=200.0):
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

        return obj

    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.65, 1.55, 0.6, 1.7, 0.55, 1.85, 0.5, 1.15, 0.85, 0.75, 1.3, 0.68, 1.45]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.42
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        rng = np.random.default_rng(123)
        for seed in range(25):
            w0 = rng.uniform(0.5, 1.5)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.13)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Hexagonal-like packing
        for w0 in [0.85, 1.0, 1.15]:
            h0 = 2.0 - w0
            if h0 < 0.4:
                continue
            z = np.zeros(3 * N + 1)
            rows = 5
            cols = 5
            spacing_x = w0 / (cols + 0.5)
            spacing_y = h0 / (rows + 0.5)
            for i in range(N):
                row, col = divmod(i, cols)
                cx = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) * spacing_x
                cy = (row + 0.5) * spacing_y
                r0 = min(spacing_x, spacing_y) * 0.46
                cx = min(max(cx, r0), w0 - r0)
                cy = min(max(cy, r0), h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Staggered rows
        for w0 in [0.9, 1.1]:
            h0 = 2.0 - w0
            if h0 < 0.4:
                continue
            z = np.zeros(3 * N + 1)
            rows = 5
            cols = 5
            for i in range(N):
                row, col = divmod(i, cols)
                cx = (col + 0.5 + (0.3 if row % 2 == 1 else 0)) * (w0 / (cols + 0.3))
                cy = (row + 0.5) * (h0 / (rows + 0.5))
                r0 = min(w0 / (cols + 0.3), h0 / (rows + 0.5)) * 0.44
                cx = min(max(cx, r0), w0 - r0)
                cy = min(max(cy, r0), h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    def force_relax(z, max_iter=400):
        z = z.copy().astype(float)
        n = N
        dt = 0.004
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
            fx -= left_viol * 80
            fx += right_viol * 80
            fy -= bottom_viol * 80
            fy += top_viol * 80

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
                f_pairs = overlap * 12.0
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

            if max_overlap < 1e-8:
                break

            force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
            if force_mag > 1.0:
                dt = max(min_dt, dt * 0.5)
            elif force_mag < 0.1:
                dt = min(0.008, dt * 1.1)

            z[0::3][:n] += dt * fx
            z[1::3][:n] += dt * fy

            for i in range(n):
                z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
                z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

        return z

    def polish_positions(z, time_budget):
        z = z.copy().astype(float)
        n = N
        x = z[0::3][:n].copy()
        y = z[1::3][:n].copy()
        r = z[2::3][:n].copy()
        w = z[-1]
        h = 2.0 - w

        dt = 0.0008
        max_iter = int(min(2500, time_budget * 120))

        for it in range(max_iter):
            if time.time() > time_budget:
                break

            fx = np.zeros(n)
            fy = np.zeros(n)

            left_viol = np.maximum(r - x, 0.0)
            right_viol = np.maximum(x + r - w, 0.0)
            bottom_viol = np.maximum(r - y, 0.0)
            top_viol = np.maximum(y + r - h, 0.0)
            fx -= left_viol * 30
            fx += right_viol * 30
            fy -= bottom_viol * 30
            fy += top_viol * 30

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

            max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
            if max_force > 0.05:
                dt = 0.0004
            else:
                dt = 0.0008

            x += dt * fx
            y += dt * fy
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

        # Grow radii
        for _ in range(200):
            if time.time() > time_budget:
                break
            r_trial = r * 1.0002
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

    def expansion_polish(z, time_budget):
        z = z.copy().astype(float)
        n = N
        x = z[0::3][:n].copy()
        y = z[1::3][:n].copy()
        r = z[2::3][:n].copy()
        w = z[-1]
        h = 2.0 - w

        max_rounds = 25
        for rnd in range(max_rounds):
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

                if r_max > r[i] + 1e-6:
                    lo = r[i]
                    hi = r_max
                    for _ in range(18):
                        mid = (lo + hi) / 2.0
                        if _is_valid_move(x, y, np.where(np.arange(n) == i, mid, r), w, h, i, x[i], y[i]):
                            lo = mid
                        else:
                            hi = mid
                    new_r = lo
                    if new_r > r[i] + 1e-7:
                        r[i] = new_r
                        improved = True

                # Try shifting
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

                if min_dist < 0.015:
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

                    for step in [0.04, 0.02, 0.01, 0.005, 0.002]:
                        new_x = x[i] + move_x * step
                        new_y = y[i] + move_y * step
                        if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                            x[i] = new_x
                            y[i] = new_y
                            improved = True
                            break
                    else:
                        perp_x = -move_y
                        perp_y = move_x
                        for step in [0.02, 0.01, 0.005]:
                            for sign in [1.0, -1.0]:
                                new_x = x[i] + sign * perp_x * step
                                new_y = y[i] + sign * perp_y * step
                                if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                                    x[i] = new_x
                                    y[i] = new_y
                                    improved = True
                                    break
                            if improved:
                                break

            if not improved:
                break

        # Final uniform expansion
        for _ in range(80):
            if time.time() > time_budget:
                break
            r_trial = r * 1.0008
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

    def is_valid_move(x, y, r, w, h, i, new_x, new_y):
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

    def is_valid(cx, cy, r, W, H, tol=1e-8):
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

    # Main optimization loop
    guesses = get_initial_guesses()
    bounds_slsqp = [(0, 2)] * (3 * N) + [(0.3, 1.7)]

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 15:
            break

        z0_relaxed = force_relax(z0, max_iter=350)

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(400, max(100, remaining * 40)))

        # Try SLSQP
        try:
            res = minimize(objective_slsqp, z0_relaxed, method='SLSQP',
                           bounds=bounds_slsqp,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 60})
            z_opt = res.x
            valid, val = check_validity(z_opt)
            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
        except Exception:
            pass

        # Try L-BFGS-B on same start (different optimizer, may find different basin)
        if time.time() < deadline - 10:
            try:
                res2 = minimize(objective_lbgfsb, z0_relaxed, method='L-BFGS-B',
                                bounds=bounds_slsqp,
                                options={'maxiter': int(min(300, max(80, remaining * 30))),
                                         'ftol': 1e-12, 'gtol': 1e-8})
                z_opt2 = res2.x
                valid2, val2 = check_validity(z_opt2)
                if valid2 and val2 > best_value:
                    best_value = val2
                    best_result = z_opt2.copy()
            except Exception:
                pass

    # Polish phase
    if best_result is not None and time.time() < deadline - 20:
        budget1 = deadline - (time.time() - start_time) - 12
        polished = polish_positions(best_result, budget1)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        if best_result is not None and time.time() < deadline - 10:
            budget2 = deadline - (time.time() - start_time) - 6
            expanded = expansion_polish(best_result, budget2)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        if best_result is not None and time.time() < deadline - 4:
            budget3 = deadline - (time.time() - start_time)
            polished2 = polish_positions(best_result, budget3)
            if polished2 is not None:
                valid, val = check_validity(polished2)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished2

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
        if is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
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
