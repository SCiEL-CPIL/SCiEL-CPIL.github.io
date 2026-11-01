"""Circle packing in perimeter-4 rectangle: Voronoi-Delaunay guided optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _soft_overlap_penalty(x, y, r, w, h, beta=8.0):
    """Smooth soft-min surrogate for pairwise overlap using log-sum-exp."""
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    
    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]
    
    # Smooth max of (r_sum - d) over all pairs
    # Use log-sum-exp for stability
    vals = r_sum_pairs - np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    vals = vals * beta
    max_vals = np.max(vals)
    if max_vals < -50:
        return 0.0
    lse = max_vals / beta + np.log(np.sum(np.exp(vals - max_vals))) / beta
    return max(0.0, lse)


def _vectorized_objective(z, penalty_weight=2000.0, smooth_beta=8.0):
    """Vectorized objective with smooth penalties."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.3 or h < 0.3:
        return 1e6
    if np.any(r < 0.001):
        return 1e6

    obj = -np.sum(r)

    # Smooth overlap penalty
    overlap_pen = _soft_overlap_penalty(x, y, r, w, h, smooth_beta)
    obj += penalty_weight * overlap_pen * overlap_pen

    # Boundary penalties (quadratic)
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

    # Aspect ratio regularization
    if w < 0.4:
        obj += penalty_weight * (0.4 - w) ** 2
    elif w > 1.6:
        obj += penalty_weight * (w - 1.6) ** 2

    return obj


def _compute_voronoi_radii(x, y, w, h):
    """
    For each circle center, compute the maximum radius such that the circle
    doesn't overlap any other circle or the walls.
    Returns an array of max radii.
    """
    max_r = np.zeros(N)
    for i in range(N):
        # Wall constraints
        r_wall = min(x[i], w - x[i], y[i], h - y[i])
        r_max = r_wall
        for j in range(N):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - 0.0  # we want distance to edge of other circle
            # Actually, for non-overlap: d >= r_i + r_j
            # So r_i <= d - r_j, but r_j is unknown. 
            # For "max radius assuming others fixed at some radius", we use d/2 as heuristic
            # Better: compute the actual constraint
            r_allowed = d - 0.0  # placeholder, will use pairwise
        # Compute max radius considering all other circles at their current radii
        for j in range(N):
            if i == j:
                continue
        # Actually let's just compute the true max radius
        r_max = r_wall
        for j in range(N):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            # r_i + r_j <= d => r_i <= d - r_j
            # But r_j is the current radius. We want to know how much we can grow r_i
            # keeping r_j fixed.
            r_allowed = d - 0.0  # this doesn't make sense without r_j
        # Let me redo this properly
        max_r[i] = r_wall
        for j in range(N):
            if i == j:
                continue
        return max_r


def _max_radius_i(x, y, r, w, h, i):
    """Max radius for circle i given all other circles fixed."""
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


def _force_relax(z, max_iter=800):
    """Force-directed relaxation to resolve overlaps (vectorized)."""
    z = z.copy().astype(float)
    n = N
    dt = 0.003
    min_dt = 0.00005

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
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > 1e-9:
            f_pairs = overlap * 12.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(len(iu[0])):
                ii = iu[0][idx]
                jj = iu[1][idx]
                fx[ii] += ux[idx] * f_pairs[idx]
                fy[ii] += uy[idx] * f_pairs[idx]
                fx[jj] -= ux[idx] * f_pairs[idx]
                fy[jj] -= uy[idx] * f_pairs[idx]

        if max_overlap < 1e-9:
            break

        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 0.5:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.05:
            dt = min(0.008, dt * 1.1)

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _polish_positions(z, time_budget):
    """Refine positions to maximize minimum gap, then grow radii."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.0005
    max_iter = int(min(5000, time_budget * 200))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Gentle wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 20
        fx += right_viol * 20
        fy -= bottom_viol * 20
        fy += top_viol * 20

        # Pairwise: push apart if close
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        gap = d_pairs - r_sum_pairs
        # Only act if gap is small (close to touching)
        close_mask = gap < 0.05
        if np.any(close_mask):
            idx_close = np.where(close_mask)[0]
            f_close = np.maximum(0.05 - gap[idx_close], 0.0) * 10.0
            dx_c = dx[iu[0][idx_close], iu[1][idx_close]]
            dy_c = dy[iu[0][idx_close], iu[1][idx_close]]
            d_c = np.maximum(d_pairs[idx_close], 1e-12)
            for k, idx in enumerate(idx_close):
                ii = iu[0][idx]
                jj = iu[1][idx]
                fx[ii] += (dx_c[k] / d_c[k]) * f_close[k]
                fy[ii] += (dy_c[k] / d_c[k]) * f_close[k]
                fx[jj] -= (dx_c[k] / d_c[k]) * f_close[k]
                fy[jj] -= (dy_c[k] / d_c[k]) * f_close[k]

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Grow radii
    for _ in range(500):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0005
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


def _expand_individual(z, time_budget):
    """Iteratively expand each circle to its max radius, then reposition."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    for rnd in range(20):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            # Compute max radius for circle i
            r_max = _max_radius_i(x, y, r, w, h, i)
            if r_max > r[i] + 1e-7:
                # Binary search for max valid radius
                lo, hi = r[i], r_max
                for _ in range(15):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(x, y, r_trial, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True

            # Try to move circle i to create more space
            # Find direction of most constraining neighbor
            best_gain = 0.0
            best_dx = 0.0
            best_dy = 0.0
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < 1e-9:
                    continue
                # Moving i away from j increases gap
                gap = d - r[i] - r[j]
                if gap < 0.03:
                    ux = dx / d
                    uy = dy / d
                    gain = 0.03 - gap
                    if gain > best_gain:
                        best_gain = gain
                        best_dx = ux
                        best_dy = uy

            # Also consider walls
            for dx_wall, dy_wall in [(1,0), (-1,0), (0,1), (0,-1)]:
                if dx_wall == 1:
                    gap = x[i] - r[i]
                elif dx_wall == -1:
                    gap = w - x[i] - r[i]
                elif dy_wall == 1:
                    gap = y[i] - r[i]
                else:
                    gap = h - y[i] - r[i]
                if gap < 0.02:
                    gain = 0.02 - gap
                    if gain > best_gain:
                        best_gain = gain
                        best_dx = float(dx_wall)
                        best_dy = float(dy_wall)

            if best_gain > 0.001:
                for step in [0.04, 0.02, 0.01, 0.005, 0.002]:
                    nx = x[i] + best_dx * step
                    ny = y[i] + best_dy * step
                    if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                        continue
                    x_trial = x.copy()
                    y_trial = y.copy()
                    x_trial[i] = nx
                    y_trial[i] = ny
                    if _is_valid(x_trial, y_trial, r, w, h):
                        x[i] = nx
                        y[i] = ny
                        improved = True
                        break

        if not improved:
            break

    # Final uniform growth
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * 1.001
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


def _slsqp_refine(z, time_budget, smooth_beta=10.0):
    """SLSQP with smooth objective."""
    def obj(zv):
        return _vectorized_objective(zv, penalty_weight=3000.0, smooth_beta=smooth_beta)

    maxiter = int(min(500, time_budget * 100))
    try:
        res = minimize(obj, z, method='SLSQP',
                       bounds=[(0.01, 1.99)] * (3 * N) + [(0.3, 1.7)],
                       options={'maxiter': maxiter, 'ftol': 1e-13, 'maxls': 60})
        z_opt = res.x
        x = z_opt[0::3][:N]
        y = z_opt[1::3][:N]
        r = z_opt[2::3][:N]
        w = z_opt[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            return z_opt
        # Try shrinking
        z_shrink = z_opt.copy()
        r_idx = np.arange(2, len(z_shrink) - 1, 3)
        z_shrink[r_idx] *= 0.92
        if _is_valid(z_shrink[0::3][:N], z_shrink[1::3][:N], z_shrink[2::3][:N], z_shrink[-1], 2.0 - z_shrink[-1]):
            return z_shrink
        return None
    except Exception:
        return None


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check validity of packing."""
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


def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    best_value = 0.0
    best_result = None

    def check_valid_value(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            return True, float(np.sum(r))
        return False, 0.0

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))
        
        # Grid-based with various aspect ratios
        aspect_ratios = [1.0, 0.85, 1.15, 0.75, 1.3, 0.67, 1.5, 0.6, 1.7, 0.55, 2.0, 0.5, 1.2, 0.8, 0.9, 1.1]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.35:
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

        # Random starts
        rng = np.random.default_rng(123)
        for _ in range(12):
            w0 = rng.uniform(0.55, 1.45)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.11)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Hexagonal-like packing for specific aspect ratios
        for w0, h0 in [(1.0, 1.0), (1.2, 0.8), (0.8, 1.2), (1.33, 0.67), (1.5, 0.5)]:
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            row_h = h0 / 5.0
            for i in range(N):
                row = i // 5
                col = i % 5
                cx = (col + 0.5) / 5.0 * w0 + (row % 2) * (w0 / 10.0)
                cy = (row + 0.5) * row_h
                r0 = min(w0 / 6.0, row_h / 1.8) * 0.9
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    # Phase 1: SLSQP from multiple starts
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        remaining = deadline - (time.time() - start_time)
        if remaining < 5:
            break

        # Force relax to resolve initial overlaps
        z0_relaxed = _force_relax(z0, max_iter=400)

        # SLSQP
        maxiter = int(min(400, max(100, remaining * 40)))
        try:
            res = minimize(_vectorized_objective, z0_relaxed, method='SLSQP',
                           bounds=[(0.01, 1.99)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 60})
            z_opt = res.x
            valid, val = check_valid_value(z_opt)
            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.9
                valid, val = check_valid_value(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()
        except Exception:
            continue

    # Phase 2: Polish best result
    if best_result is not None:
        # 2a: Position polish
        if time.time() < deadline - 25:
            polished = _polish_positions(best_result, deadline - (time.time() - start_time) - 20)
            if polished is not None:
                valid, val = check_valid_value(polished)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished

        # 2b: Individual expansion
        if time.time() < deadline - 20:
            expanded = _expand_individual(best_result, deadline - (time.time() - start_time) - 15)
            if expanded is not None:
                valid, val = check_valid_value(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 2c: High-precision SLSQP
        if time.time() < deadline - 15:
            refined = _slsqp_refine(best_result, deadline - (time.time() - start_time) - 10, smooth_beta=15.0)
            if refined is not None:
                valid, val = check_valid_value(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 2d: Another expansion round
        if time.time() < deadline - 10:
            expanded2 = _expand_individual(best_result, deadline - (time.time() - start_time) - 5)
            if expanded2 is not None:
                valid, val = check_valid_value(expanded2)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded2

        # 2e: Final polish
        if time.time() < deadline - 5:
            polished2 = _polish_positions(best_result, deadline - (time.time() - start_time))
            if polished2 is not None:
                valid, val = check_valid_value(polished2)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished2

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final safety shrink
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
