"""Circle packing in perimeter-4 rectangle: grid init + log-sum-exp L-BFGS-B optimization."""
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


def _force_relax(x, y, r, w, h, max_iter=1500, dt_init=0.003):
    """Force-directed relaxation to resolve overlaps and push circles to boundaries."""
    n = N
    dt = dt_init
    for it in range(max_iter):
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

        if np.max(overlap) < 1e-10:
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


def _log_sum_exp_penalty(z, w, h, beta=100.0):
    """Smooth log-sum-exp penalty for overlaps and boundary violations."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]

    # Pairwise overlap penalty via log-sum-exp
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
    overlap = r_sum[iu] - d_pairs

    # Log-sum-exp of positive overlaps
    max_ov = np.max(overlap)
    if max_ov > 0:
        penalty_overlap = max_ov + np.log(np.sum(np.exp(beta * (overlap - max_ov)))) / beta
    else:
        penalty_overlap = 0.0

    # Boundary violations
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)
    all_viol = np.concatenate([left_viol, right_viol, bottom_viol, top_viol])
    max_viol = np.max(all_viol)
    if max_viol > 0:
        penalty_boundary = max_viol + np.log(np.sum(np.exp(beta * (all_viol - max_viol)))) / beta
    else:
        penalty_boundary = 0.0

    return penalty_overlap + penalty_boundary


def _objective(z, w, h):
    """Objective: minimize -sum(r) + penalty."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]

    if np.any(r <= 0):
        return 1e8

    penalty = _log_sum_exp_penalty(z, w, h, beta=100.0)
    obj = -np.sum(r) + 50.0 * penalty

    # Aspect ratio constraint penalty
    if w < 0.4:
        obj += 500.0 * (0.4 - w) ** 2
    elif w > 1.6:
        obj += 500.0 * (w - 1.6) ** 2

    return obj


def _optimize_positions(z_init, w, h, time_budget):
    """Optimize positions and radii using L-BFGS-B with log-sum-exp penalty."""
    n = N
    maxiter = int(min(500, (time_budget - time.time()) * 80))
    if maxiter < 30:
        maxiter = 30

    bounds = [(0.01, 1.99)] * (2 * n) + [(0.01, 0.99)] * n

    try:
        res = minimize(
            lambda z: _objective(z, w, h),
            z_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50}
        )
        return res.x
    except Exception:
        return z_init


def _binary_search_radii(x, y, r, w, h, time_budget):
    """Binary search for maximum uniform scale factor on radii."""
    lo, hi = 1.0, 1.5
    for _ in range(40):
        if time.time() > time_budget:
            break
        mid = (lo + hi) / 2
        r_trial = r * mid
        if _is_valid(x, y, r_trial, w, h):
            lo = mid
        else:
            hi = mid
    return r * lo


def _expand_individual(x, y, r, w, h, time_budget):
    """Try to expand each circle individually."""
    n = N
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
    return r


def _grid_init(w, h, rows, cols):
    """Create grid-based initial layout."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)

    cell_w = w / cols
    cell_h = h / rows
    base_r = min(cell_w, cell_h) * 0.45

    for i in range(N):
        row = i // cols
        col = i % cols
        x[i] = (col + 0.5) * cell_w
        y[i] = (row + 0.5) * cell_h
        r[i] = base_r

    return x, y, r


def _perturb(x, y, r, w, h, scale=0.02, rng=None):
    """Add random perturbation to positions for multi-start diversity."""
    if rng is None:
        rng = np.random.default_rng()
    x_p = x + rng.normal(0, scale, N)
    y_p = y + rng.normal(0, scale, N)
    x_p = np.clip(x_p, r, w - r)
    y_p = np.clip(y_p, r, h - r)
    return x_p, y_p


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None
    rng = np.random.default_rng(42)

    # Grid configurations to try
    grid_configs = []
    for r in range(3, 8):
        for c in range(3, 8):
            if r * c >= N:
                w_opt = 2.0 * c / (r + c)
                h_opt = 2.0 * r / (r + c)
                for aspect_mult in [0.85, 0.95, 1.0, 1.05, 1.15]:
                    w_try = w_opt * aspect_mult
                    h_try = 2.0 - w_try
                    if h_try < 0.35 or w_try < 0.35:
                        continue
                    grid_configs.append((r, c, w_try, h_try))

    # Standard aspect ratios
    standard_aspects = [
        (1.0, 1.0), (0.85, 1.15), (1.15, 0.85), (0.75, 1.25), (1.25, 0.75),
        (0.65, 1.35), (1.35, 0.65)
    ]
    for w_s, h_s in standard_aspects:
        for r in range(3, 7):
            c = int(np.ceil(N / r))
            if r * c >= N:
                grid_configs.append((r, c, w_s, h_s))

    # Deduplicate
    seen = set()
    unique_configs = []
    for config in grid_configs:
        key = (config[0], config[1], round(config[2], 2))
        if key not in seen:
            seen.add(key)
            unique_configs.append(config)

    # Sort by expected value
    def expected_value(config):
        r, c, w, h = config
        cell_w = w / c
        cell_h = h / r
        inradius = min(cell_w, cell_h) / 2.0
        return N * inradius

    unique_configs.sort(key=expected_value, reverse=True)

    # Limit configs based on time
    max_configs = 12
    configs_to_try = unique_configs[:max_configs]

    for r_rows, c_cols, w, h in configs_to_try:
        if time.time() > deadline - 10:
            break

        time_budget = min(deadline - 3, time.time() + 10.0)

        # Get grid initial layout
        x0, y0, r0 = _grid_init(w, h, r_rows, c_cols)

        # Force relax to resolve initial overlaps
        x_relax, y_relax = _force_relax(x0, y0, r0, w, h, max_iter=500)

        if not _is_valid(x_relax, y_relax, r0, w, h):
            continue

        # Multi-start: try original and 2 perturbed versions
        starts = [(x_relax, y_relax)]
        for pert_scale in [0.01, 0.03]:
            xp, yp = _perturb(x_relax, y_relax, r0, w, h, scale=pert_scale, rng=rng)
            starts.append((xp, yp))

        for x_start, y_start in starts:
            if time.time() > deadline - 5:
                break

            # Build optimization vector
            z_init = np.zeros(3 * N)
            z_init[0::3][:N] = x_start
            z_init[1::3][:N] = y_start
            z_init[2::3][:N] = r0

            # Optimize
            z_opt = _optimize_positions(z_init, w, h, time_budget)

            x_opt = z_opt[0::3][:N]
            y_opt = z_opt[1::3][:N]
            r_opt = z_opt[2::3][:N]

            # Force relax to clean up
            x_clean, y_clean = _force_relax(x_opt, y_opt, r_opt, w, h, max_iter=300)

            if not _is_valid(x_clean, y_clean, r_opt, w, h):
                # Try shrinking radii slightly
                for shrink in [0.95, 0.90, 0.85]:
                    r_shrink = r_opt * shrink
                    if _is_valid(x_clean, y_clean, r_shrink, w, h):
                        r_opt = r_shrink
                        break
                else:
                    continue

            # Binary search for max uniform radius scale
            r_grown = _binary_search_radii(x_clean, y_clean, r_opt, w, h, time_budget)

            # Individual expansion
            r_expanded = _expand_individual(x_clean, y_clean, r_grown, w, h, time_budget)

            if _is_valid(x_clean, y_clean, r_expanded, w, h):
                value = np.sum(r_expanded)
                if value > best_value:
                    best_value = value
                    best_result = np.column_stack([x_clean, y_clean, r_expanded])
                    # Update for next iterations
                    r_opt = r_expanded.copy()

    # Final polish on best result if time remains
    if best_result is not None and time.time() < deadline - 8:
        x = best_result[:, 0].copy()
        y = best_result[:, 1].copy()
        r = best_result[:, 2].copy()

        min_x = np.min(x - r)
        max_x = np.max(x + r)
        min_y = np.min(y - r)
        max_y = np.max(y + r)
        bw = max_x - min_x
        bh = max_y - min_y

        # Center in bounding box
        x = x - min_x
        y = y - min_y

        time_budget = min(deadline - 2, time.time() + 12.0)

        # Force relax
        x_polished, y_polished = _force_relax(x, y, r, bw, bh, max_iter=500)

        if _is_valid(x_polished, y_polished, r, bw, bh):
            r_grown = _binary_search_radii(x_polished, y_polished, r, bw, bh, time_budget)
            r_expanded = _expand_individual(x_polished, y_polished, r_grown, bw, bh, time_budget)

            if _is_valid(x_polished, y_polished, r_expanded, bw, bh):
                value = np.sum(r_expanded)
                if value > best_value:
                    best_value = value
                    best_result = np.column_stack([x_polished, y_polished, r_expanded])

    # Final validation
    if best_result is not None:
        x = best_result[:, 0]
        y = best_result[:, 1]
        r = best_result[:, 2]

        min_x = np.min(x - r)
        max_x = np.max(x + r)
        min_y = np.min(y - r)
        max_y = np.max(y + r)
        bw = max_x - min_x
        bh = max_y - min_y
        perimeter = 2 * (bw + bh)

        if perimeter > 4.0 + 1e-6 or not _is_valid(x, y, r, bw, bh):
            # Fallback: simple grid
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                     for i in range(side) for j in range(side)]
            return np.array(cells[:N])

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_result
