"""Circle packing in perimeter-4 rectangle: Hybrid SLSQP + L-BFGS-B with smooth penalties and iterative expansion."""
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


def _objective_slsqp(z, penalty_weight=5000.0):
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


def _objective_smooth(z, beta=50.0):
    """
    Smooth log-sum-exp based objective for L-BFGS-B.
    Uses soft-max to create a differentiable surrogate for the max-overlap constraint.
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Smooth pairwise overlap: use log-sum-exp of (r_i + r_j - d_ij)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]
    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))

    # overlap_ij = r_i + r_j - d_ij (positive when overlapping)
    overlap_vals = r_sum_pairs - d_pairs

    # Smooth max via log-sum-exp
    shifted = beta * overlap_vals
    shifted = np.clip(shifted, -500, 500)
    max_shifted = np.max(shifted)
    lse = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    smooth_max_overlap = lse / beta

    # Also penalize individual overlaps with quadratic for local structure
    hard_overlap = np.maximum(overlap_vals, 0.0)
    quad_penalty = 100.0 * np.sum(hard_overlap * hard_overlap)

    obj += 500.0 * smooth_max_overlap + quad_penalty

    # Smooth boundary violations
    left_viol = r - x
    right_viol = x + r - w
    bottom_viol = r - y
    top_viol = y + r - h

    all_viol = np.concatenate([left_viol, right_viol, bottom_viol, top_viol])
    shifted_b = beta * all_viol
    shifted_b = np.clip(shifted_b, -500, 500)
    max_b = np.max(shifted_b)
    lse_b = max_b + np.log(np.sum(np.exp(shifted_b - max_b)))
    smooth_max_viol = lse_b / beta

    hard_viol = np.maximum(all_viol, 0.0)
    obj += 500.0 * smooth_max_viol + 100.0 * np.sum(hard_viol * hard_viol)

    # Aspect ratio penalty (smooth)
    if w < 0.5:
        obj += 200.0 * (0.5 - w) ** 2
    elif w > 1.5:
        obj += 200.0 * (w - 1.5) ** 2

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

    max_iter = int(min(2000, time_budget * 100))
    dt = 0.001
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

        x = x + dt * fx
        y = y + dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Grow radii
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0005
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

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


def _get_initial_guesses():
    """Generate diverse initial guesses for optimization."""
    guesses = []
    side = int(np.ceil(np.sqrt(N)))

    # Focused set of aspect ratios to balance exploration and time
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.2, 0.8]
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

    # Random starts for diversity
    rng = np.random.default_rng(42)
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

    # Hexagonal guess for structural diversity
    w0 = 1.0
    h0 = 1.0
    z = np.zeros(3 * N + 1)
    rows = 5
    cols = 5
    spacing_x = w0 / (cols + 0.5)
    spacing_y = h0 / (rows + 0.5)
    for i in range(N):
        row, col = divmod(i, cols)
        cx = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) * spacing_x
        cy = (row + 0.5) * spacing_y
        r0 = min(spacing_x, spacing_y) * 0.48
        cx = min(max(cx, r0), w0 - r0)
        cy = min(max(cy, r0), h0 - r0)
        z[3 * i] = cx
        z[3 * i + 1] = cy
        z[3 * i + 2] = r0
    z[-1] = w0
    guesses.append(z)

    # Staggered grid with tighter packing
    w0 = 1.0
    h0 = 1.0
    z = np.zeros(3 * N + 1)
    rows = 5
    cols = 5
    sx = w0 / (cols + 0.3)
    sy = h0 / (rows + 0.3)
    for i in range(N):
        row, col = divmod(i, cols)
        cx = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) * sx
        cy = (row + 0.5) * sy
        r0 = min(sx, sy) * 0.49
        cx = min(max(cx, r0), w0 - r0)
        cy = min(max(cy, r0), h0 - r0)
        z[3 * i] = cx
        z[3 * i + 1] = cy
        z[3 * i + 2] = r0
    z[-1] = w0
    guesses.append(z)

    # Corner-biased: place larger circles at corners
    w0 = 1.0
    h0 = 1.0
    z = np.zeros(3 * N + 1)
    corners = [(0.15, 0.15), (w0-0.15, 0.15), (0.15, h0-0.15), (w0-0.15, h0-0.15)]
    for i in range(N):
        if i < 4:
            cx, cy = corners[i]
            r0 = 0.12
        else:
            row, col = divmod(i-4, 4)
            cx = 0.25 + col * 0.18
            cy = 0.25 + row * 0.18
            r0 = 0.08
        cx = min(max(cx, r0), w0 - r0)
        cy = min(max(cy, r0), h0 - r0)
        z[3 * i] = cx
        z[3 * i + 1] = cy
        z[3 * i + 2] = r0
    z[-1] = w0
    guesses.append(z)

    return guesses


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_value = 0.0
    best_result = None

    # Phase 1: Grid-based initial layouts with force relaxation
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55, 1.8, 0.5]

    for w0 in aspect_ratios:
        if time.time() > deadline - 20:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        # Determine grid dimensions
        best_grid = None
        best_grid_r = 0
        for rows in range(3, 8):
            cols = int(np.ceil(N / rows))
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
            x[i] = (col + 0.5) * (w0 / cols)
            y[i] = (row + 0.5) * (h0 / rows)
            r[i] = min(w0 / cols, h0 / rows) * 0.45

        # Force relaxation
        x, y = _force_relax(x, y, r, w0, h0, max_iter=500)

        if not _is_valid(x, y, r, w0, h0):
            r *= 0.95
            if not _is_valid(x, y, r, w0, h0):
                continue

        val = float(np.sum(r))
        if val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w0, h0)

    # Phase 2: Optimization with both SLSQP and L-BFGS-B
    if best_result is not None and time.time() < deadline - 30:
        x_best, y_best, r_best, w_best, h_best = best_result

        # Build z vector
        z = np.zeros(3 * N + 1)
        z[0::3][:N] = x_best
        z[1::3][:N] = y_best
        z[2::3][:N] = r_best
        z[-1] = w_best

        # SLSQP optimization
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(500, max(200, remaining * 40)))
        try:
            res = minimize(_objective_slsqp, z, method='SLSQP',
                           bounds=[(0.01, 2.0)] * (2 * N) + [(0.01, 1.0)] * N + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})
            z_opt = res.x
            x_opt = z_opt[0::3][:N]
            y_opt = z_opt[1::3][:N]
            r_opt = z_opt[2::3][:N]
            w_opt = z_opt[-1]
            h_opt = 2.0 - w_opt

            if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
                val = float(np.sum(r_opt))
                if val > best_value:
                    best_value = val
                    best_result = (x_opt.copy(), y_opt.copy(), r_opt.copy(), w_opt, h_opt)
            else:
                # Repair
                r_repair = r_opt * 0.95
                if _is_valid(x_opt, y_opt, r_repair, w_opt, h_opt):
                    val = float(np.sum(r_repair))
                    if val > best_value:
                        best_value = val
                        best_result = (x_opt.copy(), y_opt.copy(), r_repair.copy(), w_opt, h_opt)
        except Exception:
            pass

        # L-BFGS-B with smooth objective for additional refinement
        if time.time() < deadline - 20:
            z_current = np.zeros(3 * N + 1)
            z_current[0::3][:N] = best_result[0]
            z_current[1::3][:N] = best_result[1]
            z_current[2::3][:N] = best_result[2]
            z_current[-1] = best_result[3]

            try:
                res2 = minimize(_objective_smooth, z_current, method='L-BFGS-B',
                                bounds=[(0.01, 2.0)] * (2 * N) + [(0.01, 1.0)] * N + [(0.3, 1.7)],
                                options={'maxiter': 300, 'ftol': 1e-12, 'maxls': 50})
                z_opt2 = res2.x
                x_opt2 = z_opt2[0::3][:N]
                y_opt2 = z_opt2[1::3][:N]
                r_opt2 = z_opt2[2::3][:N]
                w_opt2 = z_opt2[-1]
                h_opt2 = 2.0 - w_opt2

                if _is_valid(x_opt2, y_opt2, r_opt2, w_opt2, h_opt2):
                    val = float(np.sum(r_opt2))
                    if val > best_value:
                        best_value = val
                        best_result = (x_opt2.copy(), y_opt2.copy(), r_opt2.copy(), w_opt2, h_opt2)
                else:
                    r_repair2 = r_opt2 * 0.95
                    if _is_valid(x_opt2, y_opt2, r_repair2, w_opt2, h_opt2):
                        val = float(np.sum(r_repair2))
                        if val > best_value:
                            best_value = val
                            best_result = (x_opt2.copy(), y_opt2.copy(), r_repair2.copy(), w_opt2, h_opt2)
            except Exception:
                pass

    # Phase 3: Multi-start with diverse initial guesses
    if time.time() < deadline - 15:
        guesses = _get_initial_guesses()
        for z_guess in guesses:
            if time.time() > deadline - 10:
                break

            # Quick force relaxation
            x_g = z_guess[0::3][:N]
            y_g = z_guess[1::3][:N]
            r_g = z_guess[2::3][:N]
            w_g = z_guess[-1]
            h_g = 2.0 - w_g

            if w_g < 0.3 or h_g < 0.3:
                continue

            x_g, y_g = _force_relax(x_g, y_g, r_g, w_g, h_g, max_iter=300)

            if not _is_valid(x_g, y_g, r_g, w_g, h_g):
                r_g *= 0.95
                if not _is_valid(x_g, y_g, r_g, w_g, h_g):
                    continue

            val = float(np.sum(r_g))
            if val > best_value:
                best_value = val
                best_result = (x_g.copy(), y_g.copy(), r_g.copy(), w_g, h_g)

            # Quick optimization on this guess
            if time.time() < deadline - 10 and val > best_value * 0.9:
                z_opt = np.zeros(3 * N + 1)
                z_opt[0::3][:N] = x_g
                z_opt[1::3][:N] = y_g
                z_opt[2::3][:N] = r_g
                z_opt[-1] = w_g

                try:
                    res3 = minimize(_objective_slsqp, z_opt, method='SLSQP',
                                    bounds=[(0.01, 2.0)] * (2 * N) + [(0.01, 1.0)] * N + [(0.3, 1.7)],
                                    options={'maxiter': 100, 'ftol': 1e-10, 'maxls': 50})
                    z_res = res3.x
                    x_r = z_res[0::3][:N]
                    y_r = z_res[1::3][:N]
                    r_r = z_res[2::3][:N]
                    w_r = z_res[-1]
                    h_r = 2.0 - w_r

                    if _is_valid(x_r, y_r, r_r, w_r, h_r):
                        val_r = float(np.sum(r_r))
                        if val_r > best_value:
                            best_value = val_r
                            best_result = (x_r.copy(), y_r.copy(), r_r.copy(), w_r, h_r)
                except Exception:
                    pass

    # Phase 4: Polish phase
    if best_result is not None and time.time() < deadline - 10:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 5
        polished = _polish_positions(x_best, y_best, r_best, w_best, h_best, remaining)
        if polished is not None:
            x_p, y_p, r_p = polished
            val = float(np.sum(r_p))
            if val > best_value:
                best_value = val
                best_result = (x_p, y_p, r_p, w_best, h_best)

    # Phase 5: Final expansion
    if best_result is not None and time.time() < deadline - 3:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 1
        expanded = _expand_individual(x_best, y_best, r_best, w_best, h_best, remaining)
        if expanded is not None:
            x_e, y_e, r_e, _ = expanded
            val = float(np.sum(r_e))
            if val > best_value:
                best_value = val
                best_result = (x_e, y_e, r_e, w_best, h_best)

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result

    # Final safety: ensure strict validity
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
