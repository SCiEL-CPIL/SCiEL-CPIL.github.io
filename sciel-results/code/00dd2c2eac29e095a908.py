"""Circle packing in perimeter-4 rectangle: Direct construction + optimization polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-9):
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


def _objective(z, penalty_weight=5000.0):
    """Objective for SLSQP with penalties."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.1 or h < 0.1 or np.any(r <= 0):
        return 1e8

    obj = -np.sum(r)

    # Pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    # Boundary penalty
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)
    obj += penalty_weight * (np.sum(left_viol**2) + np.sum(right_viol**2) +
                             np.sum(bottom_viol**2) + np.sum(top_viol**2))

    # Aspect ratio penalty to stay in reasonable range
    if w < 0.4:
        obj += penalty_weight * (0.4 - w) ** 2
    elif w > 1.6:
        obj += penalty_weight * (w - 1.6) ** 2

    return obj


def _force_relax(x, y, r, w, h, max_iter=800, dt_init=0.003):
    """Force-directed relaxation to resolve overlaps and push circles to boundaries."""
    n = N
    dt = dt_init
    for it in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 80
        fx += right_viol * 80
        fy -= bottom_viol * 80
        fy += top_viol * 80

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

        if np.max(overlap) < 1e-9:
            break

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

        # Adaptive step
        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.5:
            dt = max(0.0001, dt * 0.7)
        elif max_force < 0.05:
            dt = min(0.005, dt * 1.1)

        x = x + dt * fx
        y = y + dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    return x, y


def _polish_positions(x, y, r, w, h, time_budget):
    """Polish positions via force-directed method, then grow radii."""
    n = N
    x = x.copy()
    y = y.copy()
    r = r.copy()

    max_iter = int(min(3000, (time_budget - time.time()) * 120))
    if max_iter < 150:
        max_iter = 150
    dt = 0.0008
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

        x = x + dt * fx
        y = y + dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

        # Adaptive step size
        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.3:
            dt = max(0.0002, dt * 0.8)
        elif max_force < 0.02:
            dt = min(0.0015, dt * 1.1)

    # Binary search for max uniform scale factor
    lo, hi = 1.0, 1.6
    for _ in range(45):
        if time.time() > time_budget:
            break
        mid = (lo + hi) / 2
        r_trial = r * mid
        if _is_valid(x, y, r_trial, w, h):
            lo = mid
        else:
            hi = mid

    r = r * lo

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None


def _expand_individual(x, y, r, w, h, time_budget):
    """Try to expand each circle individually by moving it to free space."""
    n = N
    improved = False
    for i in range(n):
        if time.time() > time_budget:
            break
        # Find max possible radius at current position
        r_wall = min(x[i], w - x[i], y[i], h - y[i])
        r_max = r_wall
        for j in range(n):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - r[j]
            r_max = min(r_max, r_allowed)
        r_max = max(0.0, r_max)

        if r_max > r[i] + 1e-5:
            r[i] = r_max
            improved = True

    if _is_valid(x, y, r, w, h):
        return x, y, r, improved
    return None


def _slsqp_optimize(x, y, r, w, h, time_budget):
    """SLSQP optimization of positions and radii."""
    n = N
    z = np.zeros(3 * n + 1)
    z[0::3][:n] = x
    z[1::3][:n] = y
    z[2::3][:n] = r
    z[-1] = w

    bounds = [(0.01, 2.0)] * (2 * n) + [(0.01, 1.0)] * n + [(0.3, 1.7)]
    maxiter = int(min(800, (time_budget - time.time()) * 60))
    if maxiter < 50:
        maxiter = 50

    try:
        res = minimize(_objective, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 100})
        z_opt = res.x
        x_opt = z_opt[0::3][:n]
        y_opt = z_opt[1::3][:n]
        r_opt = z_opt[2::3][:n]
        w_opt = z_opt[-1]
        h_opt = 2.0 - w_opt

        if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
            return x_opt, y_opt, r_opt, w_opt, h_opt
        else:
            # Try to repair
            r_repair = r_opt * 0.98
            if _is_valid(x_opt, y_opt, r_repair, w_opt, h_opt):
                return x_opt, y_opt, r_repair, w_opt, h_opt
    except Exception:
        pass

    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    # Direct construction: grid-based initial layouts for various aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55, 1.8, 0.5, 0.9, 1.1, 1.3, 0.65]

    for w0 in aspect_ratios:
        if time.time() > deadline - 20:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        # Try multiple grid configurations and pick the best
        best_grid = None
        best_grid_r = 0
        for rows in range(3, 9):
            for cols in range(3, 9):
                if rows * cols < N:
                    continue
                if rows * cols > N + 6:
                    continue
                cell_w = w0 / cols
                cell_h = h0 / rows
                r_grid = min(cell_w, cell_h) * 0.48
                if r_grid > best_grid_r:
                    best_grid_r = r_grid
                    best_grid = (rows, cols)

        if best_grid is None:
            continue

        rows, cols = best_grid
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)

        for i in range(N):
            row = i // cols
            col = i % cols
            # Stagger odd rows for better packing
            if row % 2 == 1:
                x[i] = (col + 0.5 + 0.25) * (w0 / cols)
            else:
                x[i] = (col + 0.5) * (w0 / cols)
            y[i] = (row + 0.5) * (h0 / rows)
            r[i] = min(w0 / cols, h0 / rows) * 0.45

        # Force relaxation
        x, y = _force_relax(x, y, r, w0, h0, max_iter=800)

        if not _is_valid(x, y, r, w0, h0):
            r_shrink = r * 0.92
            if not _is_valid(x, y, r_shrink, w0, h0):
                continue
            r = r_shrink

        # Polish and grow
        polished = _polish_positions(x, y, r, w0, h0, deadline - 15)
        
        # Additional high-precision polish pass
        if polished is not None and time.time() < deadline - 10:
            x_h, y_h, r_h = polished
            # Do a final tight polish with small steps
            n = N
            x_f = x_h.copy()
            y_f = y_h.copy()
            r_f = r_h.copy()
            dt_f = 0.0003
            for it_f in range(500):
                if time.time() > deadline - 8:
                    break
                fx = np.zeros(n)
                fy = np.zeros(n)
                left_viol = np.maximum(r_f - x_f, 0.0)
                right_viol = np.maximum(x_f + r_f - w0, 0.0)
                bottom_viol = np.maximum(r_f - y_f, 0.0)
                top_viol = np.maximum(y_f + r_f - h0, 0.0)
                fx -= left_viol * 50
                fx += right_viol * 50
                fy -= bottom_viol * 50
                fy += top_viol * 50
                dx = x_f[:, None] - x_f[None, :]
                dy = y_f[:, None] - y_f[None, :]
                d_sq = dx * dx + dy * dy
                r_sum = r_f[:, None] + r_f[None, :]
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
                x_f = x_f + dt_f * fx
                y_f = y_f + dt_f * fy
                x_f = np.clip(x_f, r_f, w0 - r_f)
                y_f = np.clip(y_f, r_f, h0 - r_f)
            # Try to grow radii after tight polish
            lo_g, hi_g = 1.0, 1.3
            for _ in range(30):
                if time.time() > deadline - 7:
                    break
                mid_g = (lo_g + hi_g) / 2
                r_trial_g = r_f * mid_g
                if _is_valid(x_f, y_f, r_trial_g, w0, h0):
                    lo_g = mid_g
                else:
                    hi_g = mid_g
            r_f = r_f * lo_g
            if _is_valid(x_f, y_f, r_f, w0, h0):
                val_f = float(np.sum(r_f))
                if val_f > best_value:
                    best_value = val_f
                    best_result = (x_f.copy(), y_f.copy(), r_f.copy(), w0, h0)
        if polished is not None:
            x_p, y_p, r_p = polished
            val = float(np.sum(r_p))
            if val > best_value:
                best_value = val
                best_result = (x_p.copy(), y_p.copy(), r_p.copy(), w0, h0)

            # Try individual expansion
            if time.time() < deadline - 25:
                expanded = _expand_individual(x_p, y_p, r_p, w0, h0, deadline - 20)
                if expanded is not None:
                    x_e, y_e, r_e, improved = expanded
                    if improved:
                        val_e = float(np.sum(r_e))
                        if val_e > best_value:
                            best_value = val_e
                            best_result = (x_e.copy(), y_e.copy(), r_e.copy(), w0, h0)

        # SLSQP optimization if time permits
        if time.time() < deadline - 35 and best_result is not None:
            x_b, y_b, r_b, w_b, h_b = best_result
            opt_result = _slsqp_optimize(x_b, y_b, r_b, w_b, h_b, deadline - 30)
            if opt_result is not None:
                x_o, y_o, r_o, w_o, h_o = opt_result
                val = float(np.sum(r_o))
                if val > best_value:
                    best_value = val
                    best_result = (x_o.copy(), y_o.copy(), r_o.copy(), w_o, h_o)

                # Polish after SLSQP with tighter settings
                if time.time() < deadline - 12:
                    polished2 = _polish_positions(x_o, y_o, r_o, w_o, h_o, deadline - 8)
                    if polished2 is not None:
                        x_p2, y_p2, r_p2 = polished2
                        val2 = float(np.sum(r_p2))
                        if val2 > best_value:
                            best_value = val2
                            best_result = (x_p2.copy(), y_p2.copy(), r_p2.copy(), w_o, h_o)

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result

    # Final safety: ensure strict validity
    for _ in range(20):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
