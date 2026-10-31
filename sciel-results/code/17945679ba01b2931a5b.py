"""Circle packing in perimeter-4 rectangle: SLSQP + Contact-Graph Polish + Multi-Phase Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _smooth_penalty(z, penalty_weight=1000.0):
    """
    Objective function using smooth soft-plus based penalty for overlaps and boundaries.
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.1 or h < 0.1 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # --- Pairwise Overlap Penalty (Smoothed) ---
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    v_pairs = r_sum_pairs - d_pairs

    # Smooth max(0, v)
    eps = 1e-6
    smooth_v = (v_pairs + np.sqrt(v_pairs * v_pairs + eps)) / 2.0

    # Penalty is quadratic on smooth violation
    overlap_penalty = penalty_weight * np.sum(smooth_v * smooth_v)
    obj += overlap_penalty

    # --- Boundary Penalty (Smoothed) ---
    left_viol = r - x
    right_viol = x + r - w
    bottom_viol = r - y
    top_viol = y + r - h

    smooth_left = (left_viol + np.sqrt(left_viol * left_viol + eps)) / 2.0
    smooth_right = (right_viol + np.sqrt(right_viol * right_viol + eps)) / 2.0
    smooth_bottom = (bottom_viol + np.sqrt(bottom_viol * bottom_viol + eps)) / 2.0
    smooth_top = (top_viol + np.sqrt(top_viol * top_viol + eps)) / 2.0

    boundary_penalty = penalty_weight * (
        np.sum(smooth_left * smooth_left) +
        np.sum(smooth_right * smooth_right) +
        np.sum(smooth_bottom * smooth_bottom) +
        np.sum(smooth_top * smooth_top)
    )
    obj += boundary_penalty

    # Aspect ratio regularization
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    best_value = 0.0
    best_result = None

    # --- Phase 1: Multi-Start SLSQP with Smooth Penalty ---

    # Initial guesses
    guesses = []

    # 1. Hexagonal Lattice Initializations
    aspect_ratios_hex = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 1.1, 0.9]
    for w0 in aspect_ratios_hex:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        z = np.zeros(3 * N + 1)
        z[-1] = w0

        r_est = np.sqrt(w0 * h0 / (N * np.pi * 0.85))

        points = []
        y_pos = r_est
        row_idx = 0
        while y_pos < h0 - r_est * 0.5:
            x_offset = 0 if row_idx % 2 == 0 else r_est
            x_pos = r_est + x_offset
            while x_pos < w0 - r_est * 0.5:
                if x_pos >= r_est and x_pos <= w0 - r_est and y_pos >= r_est and y_pos <= h0 - r_est:
                    points.append((x_pos, y_pos))
                x_pos += 2 * r_est
            y_pos += r_est * np.sqrt(3)
            row_idx += 1

        if len(points) >= N:
            for i in range(N):
                cx, cy = points[i]
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r_est
            guesses.append(z)
        else:
            # Try smaller radius
            for scale in [0.9, 0.8, 0.7, 0.6]:
                r_est_new = r_est * scale
                points = []
                y_pos = r_est_new
                row_idx = 0
                while y_pos < h0 - r_est_new * 0.5:
                    x_offset = 0 if row_idx % 2 == 0 else r_est_new
                    x_pos = r_est_new + x_offset
                    while x_pos < w0 - r_est_new * 0.5:
                        if x_pos >= r_est_new and x_pos <= w0 - r_est_new and y_pos >= r_est_new and y_pos <= h0 - r_est_new:
                            points.append((x_pos, y_pos))
                        x_pos += 2 * r_est_new
                    y_pos += r_est_new * np.sqrt(3)
                    row_idx += 1

                if len(points) >= N:
                    for i in range(N):
                        cx, cy = points[i]
                        z[3 * i] = cx
                        z[3 * i + 1] = cy
                        z[3 * i + 2] = r_est_new
                    guesses.append(z)
                    break

    # 2. Square Grid Initializations
    side = int(np.ceil(np.sqrt(N)))
    aspect_ratios_sq = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8]
    for w0 in aspect_ratios_sq:
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

    # 3. Random starts
    rng = np.random.default_rng(123)
    for _ in range(15):
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

    # Run SLSQP on each guess
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        # Force relaxation to fix initial overlaps
        z0_relaxed = _force_relax(z0, max_iter=300)

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(400, max(100, remaining * 30)))

        try:
            res = minimize(_smooth_penalty, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})

            z_opt = res.x
            valid, val = _check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()

        except Exception:
            continue

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # --- Phase 2: Advanced Polish ---

    # 2a. Force-Directed Polish
    if time.time() < deadline - 20:
        polished = _polish_force_directed(best_result, deadline - (time.time() - start_time) - 10)
        if polished is not None:
            valid, val = _check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

    # 2b. Expansion Relaxation Polish
    if time.time() < deadline - 15:
        expanded = _expansion_polish(best_result, deadline - (time.time() - start_time) - 8)
        if expanded is not None:
            valid, val = _check_validity(expanded)
            if valid and val > best_value:
                best_value = val
                best_result = expanded

    # 2c. Constraint-based SLSQP Refinement (High Precision)
    if time.time() < deadline - 10:
        refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 5)
        if refined is not None:
            valid, val = _check_validity(refined)
            if valid and val > best_value:
                best_value = val
                best_result = refined

    # Final safety shrink
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    for _ in range(5):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))


def _check_validity(z, tol=1e-9):
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


def _force_relax(z, max_iter=300):
    """Force-directed relaxation to resolve overlaps."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005

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

        if np.max(overlap) < 1e-8:
            break

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

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy
        z[0::3][:n] = np.clip(z[0::3][:n], r, w - r)
        z[1::3][:n] = np.clip(z[1::3][:n], r, h - r)

    return z


def _polish_force_directed(z, time_budget):
    """Polish positions using force-directed relaxation."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.001
    max_iter = int(min(2000, time_budget * 100))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 35
        fx += right_viol * 35
        fy -= bottom_viol * 35
        fy += top_viol * 35

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

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Try to grow radii
    grow_factor = 1.0003
    for _ in range(200):
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


def _expansion_polish(z, time_budget):
    """Iteratively expand circles and shift to create space."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    max_rounds = 20
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            # Max radius
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j: continue
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
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    if _is_valid_move(x, y, r, w, h, i, x[i], y[i], mid):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-7:
                    r[i] = lo
                    improved = True

            # Shift if blocked
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
                if i == j: continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_dist:
                    min_dist = gap
                    blocker = j

            if min_dist < 0.02:
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

                for step in [0.05, 0.03, 0.02, 0.01, 0.005]:
                    new_x = x[i] + move_x * step
                    new_y = y[i] + move_y * step
                    if _is_valid_move(x, y, r, w, h, i, new_x, new_y, r[i]):
                        x[i] = new_x
                        y[i] = new_y
                        improved = True
                        break

        if not improved:
            break

    # Final uniform expansion
    grow_factor = 1.001
    for _ in range(100):
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


def _constraint_refine(z, time_budget):
    """SLSQP with explicit constraints for high precision."""
    z = z.copy().astype(float)
    n = N

    def obj_refine(zv):
        r = zv[2::3][:n]
        return -np.sum(r)

    constraints = []
    bounds = [(0, 2), (0, 2), (0.001, 1.0)] * n
    bounds.append((0.3, 1.7))

    for i in range(n):
        # x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3 * i] - p[3 * i + 2]})
        # w - x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3 * i] - p[3 * i + 2]})
        # y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3 * i + 1] - p[3 * i + 2]})
        # h - y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3 * i + 1] - p[3 * i + 2]})

    for i in range(n):
        for j in range(i + 1, n):
            def make_overlap_constraint(idx1, idx2):
                return lambda p: (
                    (p[3 * idx1] - p[3 * idx2]) ** 2 +
                    (p[3 * idx1 + 1] - p[3 * idx2 + 1]) ** 2 -
                    (p[3 * idx1 + 2] + p[3 * idx2 + 2]) ** 2
                )
            constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})

    maxiter = int(min(500, time_budget * 100))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 80})

        z_opt = res.x
        if _is_valid(z_opt[0::3][:n], z_opt[1::3][:n], z_opt[2::3][:n], z_opt[-1], 2.0 - z_opt[-1]):
            return z_opt
        return None
    except Exception:
        return None


def _is_valid_move(x, y, r, w, h, i, new_x, new_y, new_r):
    """Check if moving circle i to new position/radius is valid."""
    if new_x < new_r or new_x > w - new_r:
        return False
    if new_y < new_r or new_y > h - new_r:
        return False

    for j in range(N):
        if i == j: continue
        dx = new_x - x[j]
        dy = new_y - y[j]
        d2 = dx*dx + dy*dy
        r_sum = new_r + r[j]
        if d2 < r_sum*r_sum - 1e-9:
            return False
    return True


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
