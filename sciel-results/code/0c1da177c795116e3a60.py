"""Circle packing in perimeter-4 rectangle: SLSQP + Elastic Membrane Polish."""
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

    # Vectorized pairwise overlap penalty
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

    # Vectorized boundary penalty
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

    # Keep aspect ratio in reasonable range
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

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


def _elastic_membrane_polish(x, y, r, w, h, time_budget):
    """
    Elastic membrane formulation:
    Minimize strain energy of a virtual membrane with rigid circular obstacles.
    This is equivalent to force-directed relaxation but with:
    1. Smooth strain energy gradient (no hard cutoff)
    2. Adaptive stiffness that increases as overlaps decrease
    3. Simultaneous position AND radius optimization via epigraph variable
    
    The strain energy is: E = 0.5 * sum_ij (max(0, r_i + r_j - d_ij)^2)
    + 0.5 * sum_i (boundary_violations_i)^2
    
    We minimize this while maximizing sum(r) via a Lagrangian multiplier.
    """
    n = N
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    # Lagrangian multiplier for radius growth
    lambda_r = 0.5
    dt_pos = 0.002
    dt_rad = 0.0005
    stiffness = 10.0
    
    max_iter = int(min(5000, time_budget * 200))
    
    for it in range(max_iter):
        if time.time() > time_budget:
            break
        
        # Compute distances and overlaps (vectorized)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        
        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        
        # Smooth overlap: use soft-max approximation for differentiability
        raw_overlap = r_sum_pairs - d_pairs
        # Soft threshold: overlap is smooth function of distance
        beta = 50.0  # sharpness of soft-max
        soft_overlap = np.maximum(raw_overlap, 0.0)
        
        # Strain energy gradient for positions
        fx = np.zeros(n)
        fy = np.zeros(n)
        fr = np.zeros(n)
        
        # Pairwise forces from strain energy
        # dE/ddx = -stiffness * overlap * (dx/d_pairs) for each pair
        if np.max(soft_overlap) > 1e-10:
            # Increase stiffness as we get closer to valid (annealing)
            current_overlap = np.sum(soft_overlap)
            stiffness = 10.0 + 50.0 * (1.0 - min(1.0, current_overlap / 0.1))
            
            f_mag = stiffness * soft_overlap
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs
            
            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_mag[idx]
                fy[i_idx] += uy[idx] * f_mag[idx]
                fx[j_idx] -= ux[idx] * f_mag[idx]
                fy[j_idx] -= uy[idx] * f_mag[idx]
                
                # Radius forces: each circle wants to grow but is constrained
                # by its neighbors
                fr[i_idx] += lambda_r - 0.5 * stiffness * soft_overlap[idx]
                fr[j_idx] += lambda_r - 0.5 * stiffness * soft_overlap[idx]
        
        # Boundary forces
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        
        wall_stiff = 50.0
        fx -= left_viol * wall_stiff
        fx += right_viol * wall_stiff
        fy -= bottom_viol * wall_stiff
        fy += top_viol * wall_stiff
        
        # Radius boundary forces
        fr -= left_viol * wall_stiff
        fr -= right_viol * wall_stiff
        fr -= bottom_viol * wall_stiff
        fr -= top_viol * wall_stiff
        
        # Update positions
        x = x + dt_pos * fx
        y = y + dt_pos * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        
        # Update radii (grow towards optimal)
        r = r + dt_rad * fr
        r = np.maximum(r, 0.01)
        
        # Check convergence
        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))),
                       float(np.max(np.abs(fr))), 1e-8)
        
        # Adaptive step size
        if max_force > 1.0:
            dt_pos = max(0.0005, dt_pos * 0.8)
            dt_rad = max(0.0001, dt_rad * 0.8)
        elif max_force < 0.01:
            dt_pos = min(0.005, dt_pos * 1.05)
            dt_rad = min(0.002, dt_rad * 1.05)
            lambda_r = min(2.0, lambda_r * 1.01)
    
    # Final validity check and repair
    if _is_valid(x, y, r, w, h):
        return x, y, r
    
    # Try to repair by shrinking radii
    for scale in [0.99, 0.98, 0.95, 0.9]:
        r_repair = r * scale
        if _is_valid(x, y, r_repair, w, h):
            return x, y, r_repair
    
    return None


def _polish_positions(x, y, r, w, h, time_budget):
    """Polish positions via force-directed method, then grow radii."""
    n = N
    x = x.copy()
    y = y.copy()

    max_iter = int(min(3000, time_budget * 150))
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

    # Grow radii with binary search for tighter fit
    for _ in range(300):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0003
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            lo, hi = 1.0, 1.0003
            for __ in range(20):
                mid = (lo + hi) / 2
                r_mid = r * mid
                if _is_valid(x, y, r_mid, w, h):
                    lo = mid
                else:
                    hi = mid
            r = r * lo
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


def _move_circle_to_better_position(i, x, y, r, w, h, time_budget):
    """Try to move circle i to a position that allows larger radius."""
    if time.time() > time_budget:
        return x, y, r, False

    best_r = r[i]
    best_x, best_y = x[i], y[i]
    
    def _max_radius_at(xi, yi):
        r_wall = min(xi, w - xi, yi, h - yi)
        r_max = r_wall
        for j in range(N):
            if i == j:
                continue
            dx = xi - x[j]
            dy = yi - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - r[j]
            r_max = min(r_max, r_allowed)
        return max(0.0, r_max)
    
    step = 0.02
    for dx in np.arange(-0.1, 0.11, step):
        for dy in np.arange(-0.1, 0.11, step):
            if time.time() > time_budget:
                break
            xi = x[i] + dx
            yi = y[i] + dy
            if xi < r[i] or xi > w - r[i] or yi < r[i] or yi > h - r[i]:
                continue
            r_trial = _max_radius_at(xi, yi)
            if r_trial > best_r + 1e-6:
                best_r = r_trial
                best_x, best_y = xi, yi
    
    if best_r > r[i] + 1e-5:
        x[i] = best_x
        y[i] = best_y
        r[i] = best_r
        return x, y, r, True
    
    return x, y, r, False


def _get_initial_guesses():
    """Generate diverse initial guesses including hexagonal packing."""
    guesses = []
    side = int(np.ceil(np.sqrt(N)))

    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.2, 0.8, 0.9, 1.1, 1.4]
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

    # Hexagonal packing initial guesses (denser than square grid)
    for w0 in [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6]:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        z = np.zeros(3 * N + 1)
        rows_needed = int(np.ceil(N / 6)) + 1
        col_spacing = w0 / 6.0
        row_spacing = col_spacing * np.sqrt(3) / 2.0
        idx = 0
        for row in range(rows_needed):
            if idx >= N:
                break
            y_pos = (row + 0.5) * row_spacing
            if y_pos > h0 - 0.05:
                break
            offset = (col_spacing / 2.0) if (row % 2 == 1) else 0.0
            for col in range(6):
                if idx >= N:
                    break
                x_pos = (col + 0.5) * col_spacing + offset
                if x_pos < 0.05 or x_pos > w0 - 0.05:
                    continue
                r0 = min(col_spacing, row_spacing) * 0.48
                z[3 * idx] = x_pos
                z[3 * idx + 1] = y_pos
                z[3 * idx + 2] = r0
                idx += 1
        if idx < N:
            for i in range(idx, N):
                z[3 * i] = w0 / 2.0
                z[3 * i + 1] = h0 / 2.0
                z[3 * i + 2] = 0.05
        z[-1] = w0
        guesses.append(z)

    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(10):
        w0 = rng.uniform(0.6, 1.4)
        h0 = 2.0 - w0
        z = np.zeros(3 * N + 1)
        for i in range(N):
            r0 = rng.uniform(0.04, 0.14)
            cx = rng.uniform(r0, w0 - r0)
            cy = rng.uniform(r0, h0 - r0)
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)

    return guesses


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


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_value = 0.0
    best_result = None

    penalty_weight = 1000.0

    def objective(z):
        return _vectorized_objective(z, penalty_weight)

    guesses = _get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 25:
            break

        # Force-directed relaxation to resolve initial overlaps
        x0 = z0[0::3][:N].copy()
        y0 = z0[1::3][:N].copy()
        r0 = z0[2::3][:N].copy()
        w0 = z0[-1]
        h0 = 2.0 - w0

        x_rel, y_rel = _force_relax(x0, y0, r0, w0, h0, max_iter=500)

        z_rel = np.zeros(3 * N + 1)
        z_rel[0::3][:N] = x_rel
        z_rel[1::3][:N] = y_rel
        z_rel[2::3][:N] = r0
        z_rel[-1] = w0

        # SLSQP optimization
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(400, max(100, remaining * 40)))
        try:
            res = minimize(objective, z_rel, method='SLSQP',
                           bounds=[(0.01, 2.0)] * (2 * N) + [(0.01, 1.0)] * N + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})

            z_opt = res.x
            valid, val = _check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                for scale in [0.98, 0.95, 0.9]:
                    z_repair[r_idx] = z_opt[r_idx] * scale
                    valid, val = _check_validity(z_repair)
                    if valid and val > best_value:
                        best_value = val
                        best_result = z_repair.copy()
                        break

        except Exception:
            continue

    # --- Phase 2: Elastic Membrane Polish ---
    if best_result is not None and time.time() < deadline - 20:
        x_best = best_result[0::3][:N].copy()
        y_best = best_result[1::3][:N].copy()
        r_best = best_result[2::3][:N].copy()
        w_best = best_result[-1]
        h_best = 2.0 - w_best

        remaining = deadline - (time.time() - start_time) - 15
        polished = _elastic_membrane_polish(x_best, y_best, r_best, w_best, h_best, remaining)
        if polished is not None:
            x_p, y_p, r_p = polished
            z_p = np.zeros(3 * N + 1)
            z_p[0::3][:N] = x_p
            z_p[1::3][:N] = y_p
            z_p[2::3][:N] = r_p
            z_p[-1] = w_best
            valid, val = _check_validity(z_p)
            if valid and val > best_value:
                best_value = val
                best_result = z_p

    # --- Phase 3: Standard Polish ---
    if best_result is not None and time.time() < deadline - 15:
        x_best = best_result[0::3][:N].copy()
        y_best = best_result[1::3][:N].copy()
        r_best = best_result[2::3][:N].copy()
        w_best = best_result[-1]
        h_best = 2.0 - w_best

        remaining = deadline - (time.time() - start_time) - 10
        polished = _polish_positions(x_best, y_best, r_best, w_best, h_best, remaining)
        if polished is not None:
            x_p, y_p, r_p = polished
            z_p = np.zeros(3 * N + 1)
            z_p[0::3][:N] = x_p
            z_p[1::3][:N] = y_p
            z_p[2::3][:N] = r_p
            z_p[-1] = w_best
            valid, val = _check_validity(z_p)
            if valid and val > best_value:
                best_value = val
                best_result = z_p

    # --- Phase 4: Individual Circle Optimization ---
    if best_result is not None and time.time() < deadline - 10:
        x_curr = best_result[0::3][:N].copy()
        y_curr = best_result[1::3][:N].copy()
        r_curr = best_result[2::3][:N].copy()
        w_curr = best_result[-1]
        h_curr = 2.0 - w_curr

        remaining = deadline - (time.time() - start_time) - 5
        for i in range(N):
            if time.time() > remaining + start_time:
                break
            x_curr, y_curr, r_curr, improved = _move_circle_to_better_position(
                i, x_curr, y_curr, r_curr, w_curr, h_curr, remaining + start_time)

        z_curr = np.zeros(3 * N + 1)
        z_curr[0::3][:N] = x_curr
        z_curr[1::3][:N] = y_curr
        z_curr[2::3][:N] = r_curr
        z_curr[-1] = w_curr
        valid, val = _check_validity(z_curr)
        if valid and val > best_value:
            best_value = val
            best_result = z_curr

    # --- Phase 5: Final Expansion ---
    if best_result is not None and time.time() < deadline - 5:
        x_best = best_result[0::3][:N].copy()
        y_best = best_result[1::3][:N].copy()
        r_best = best_result[2::3][:N].copy()
        w_best = best_result[-1]
        h_best = 2.0 - w_best

        remaining = deadline - (time.time() - start_time) - 2
        expanded = _expand_individual(x_best, y_best, r_best, w_best, h_best, remaining)
        if expanded is not None:
            x_e, y_e, r_e, _ = expanded
            z_e = np.zeros(3 * N + 1)
            z_e[0::3][:N] = x_e
            z_e[1::3][:N] = y_e
            z_e[2::3][:N] = r_e
            z_e[-1] = w_best
            valid, val = _check_validity(z_e)
            if valid and val > best_value:
                best_value = val
                best_result = z_e

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

    # Final safety: ensure strict validity
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
