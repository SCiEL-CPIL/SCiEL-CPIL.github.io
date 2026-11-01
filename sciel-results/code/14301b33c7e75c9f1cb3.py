"""Circle packing in perimeter-4 rectangle: SLSQP + Advanced Expansion Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


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


def construct():
    start_time = time.time()
    deadline = start_time + 137.0  # Safety margin before 150s kill

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

        val = float(np.sum(r))
        return True, val

    # Generate diverse initial guesses
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
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Hexagonal packing initial guesses (denser than square grid)
        for w0 in [1.0, 0.9, 1.1, 0.8, 1.2, 0.7]:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            # Hexagonal lattice: rows offset by half spacing
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
            # Fill remaining circles if any
            if idx < N:
                for i in range(idx, N):
                    z[3 * i] = w0 / 2.0
                    z[3 * i + 1] = h0 / 2.0
                    z[3 * i + 2] = 0.05
            z[-1] = w0
            guesses.append(z)

        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(12):
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

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 20:
            break

        # Force-directed relaxation to resolve initial overlaps
        z0_relaxed = _force_relax_unified(z0, max_iter=500)

        # SLSQP optimization
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
                # Try repair: shrink radii slightly
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            continue

    # --- Phase 2: Advanced Polish on best result ---
    if best_result is not None and time.time() < deadline - 20:
        # 1. Force-directed position polish to ensure we are in a good basin
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 12)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Vectorized Expansion/Relocation Polish
        if best_result is not None and time.time() < deadline - 14:
            expanded = _expansion_relaxation_polish_v2(best_result, deadline - (time.time() - start_time) - 6)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 3. Newton-Raphson contact graph solver
        if best_result is not None and time.time() < deadline - 12:
            newton_result = _newton_contact_polish(best_result, deadline - (time.time() - start_time) - 4)
            if newton_result is not None:
                valid, val = check_validity(newton_result)
                if valid and val > best_value:
                    best_value = val
                    best_result = newton_result

        # 4. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 10:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 4)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 5. Smooth epigraph polish
        if best_result is not None and time.time() < deadline - 5:
            smoothed = _smooth_epigraph_polish(best_result, deadline - (time.time() - start_time) - 1)
            if smoothed is not None:
                valid, val = check_validity(smoothed)
                if valid and val > best_value:
                    best_value = val
                    best_result = smoothed

        # 6. Second Newton pass after other refinements
        if best_result is not None and time.time() < deadline - 4:
            newton_result2 = _newton_contact_polish(best_result, deadline - (time.time() - start_time) - 1)
            if newton_result2 is not None:
                valid, val = check_validity(newton_result2)
                if valid and val > best_value:
                    best_value = val
                    best_result = newton_result2

        # 7. Final quick polish
        if best_result is not None and time.time() < deadline - 2:
            polished2 = _polish_result(best_result, deadline - (time.time() - start_time))
            if polished2 is not None:
                valid, val = check_validity(polished2)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished2

    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Final safety: ensure strict validity with minimal shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))


def _force_relax_unified(z, max_iter=500):
    """Force-directed relaxation in unified parameter space with adaptive step (vectorized)."""
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

        # Wall repulsion (vectorized)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

        # Pairwise repulsion (vectorized)
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

        # Check convergence
        if max_overlap < convergence_tol:
            break

        # Adaptive step size
        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.01, dt * 1.1)

        # Update positions
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        # Clamp positions
        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _polish_result(z, time_budget):
    """Polish a valid packing with force-directed position adjustment and radius growth (vectorized)."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.001
    max_iter = int(min(5000, time_budget * 250))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion (vectorized)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 35
        fx += right_viol * 35
        fy -= bottom_viol * 35
        fy += top_viol * 35

        # Pairwise repulsion (vectorized)
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

        # Adaptive step
        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.1:
            dt = 0.0005
        else:
            dt = 0.001

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Try to grow radii uniformly
    grow_factor = 1.0005
    for _ in range(500):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    # Reconstruct z
    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w

    if _is_valid(x, y, r, w, h):
        return z_new
    return None


def _expansion_relaxation_polish(z, time_budget):
    """
    Advanced polish: Iteratively expand circles and shift positions to create space.
    Improved: multi-step moves, binary search for max radius, more rounds.
    """
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
            # 1. Try to expand radius i to max possible using binary search
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
                # Binary search for the true max valid radius
                lo = r[i]
                hi = r_max
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    if _is_valid_move(x, y, np.where(np.arange(n) == i, mid, r), w, h, i, x[i], y[i]):
                        lo = mid
                    else:
                        hi = mid
                new_r = lo
                if new_r > r[i] + 1e-7:
                    r[i] = new_r
                    improved = True
            
            # 2. If expansion is limited, try to shift circle i to create space
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
            
            # If touching a constraint, try multi-step moves away
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
                
                # Try multiple step sizes, from larger to smaller
                for step in [0.05, 0.03, 0.02, 0.01, 0.005, 0.002]:
                    new_x = x[i] + move_x * step
                    new_y = y[i] + move_y * step
                    if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                        x[i] = new_x
                        y[i] = new_y
                        improved = True
                        break
                else:
                    # Try perpendicular directions too
                    perp_x = -move_y
                    perp_y = move_x
                    for step in [0.03, 0.01, 0.005]:
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

    # Final uniform expansion with binary search
    grow_factor = 1.001
    for _ in range(200):
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


def _expansion_relaxation_polish_v2(z, time_budget):
    """
    Vectorized expansion polish: computes max radii for all circles simultaneously,
    then performs coordinated position shifts. Much faster than per-circle loop.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    iu = np.triu_indices(n, k=1)

    max_rounds = 60
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        # Vectorized: compute max possible radius for each circle
        dx_all = x[:, None] - x[None, :]
        dy_all = y[:, None] - y[None, :]
        d_all = np.sqrt(dx_all * dx_all + dy_all * dy_all)
        np.fill_diagonal(d_all, np.inf)

        # r_max[i] = min over j!=i of (d[i,j] - r[j]), also bounded by wall
        r_from_others = d_all - r[None, :]  # shape (n, n), r_from_others[i,j] = d[i,j] - r[j]
        np.fill_diagonal(r_from_others, np.inf)
        r_from_neigh = np.min(r_from_others, axis=1)  # shape (n,)

        r_wall = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))
        r_max = np.maximum(0.0, np.minimum(r_wall, r_from_neigh))

        # Expand all circles that can grow
        grow_mask = r_max > r + 1e-7
        if np.any(grow_mask):
            # Use 95% of available to stay safe
            r_new = np.where(grow_mask, np.minimum(r_max * 0.95, r + (r_max - r) * 0.5), r)
            # Verify validity
            if _is_valid(x, y, r_new, w, h):
                r = r_new
            else:
                # Fallback: expand one at a time
                for i in np.where(grow_mask)[0]:
                    r_trial = r.copy()
                    r_trial[i] = r_max[i] * 0.95
                    if _is_valid(x, y, r_trial, w, h):
                        r = r_trial

        # Vectorized: find circles that are "stuck" (close to constraint)
        margin_left = x - r
        margin_right = w - x - r
        margin_bottom = y - r
        margin_top = h - y - r

        # Pairwise gaps
        d_pairs = d_all[iu]
        r_sum_pairs = r[iu[0]] + r[iu[1]]
        gap_pairs = d_pairs - r_sum_pairs

        # For each circle, find minimum gap
        min_gap = np.full(n, 0.05)  # threshold
        for idx in range(len(iu[0])):
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            g = gap_pairs[idx]
            if g < min_gap[i_idx]:
                min_gap[i_idx] = g
            if g < min_gap[j_idx]:
                min_gap[j_idx] = g

        min_margin = np.minimum(np.minimum(margin_left, margin_right),
                                np.minimum(margin_bottom, margin_top))
        min_constraint = np.minimum(min_gap, min_margin)

        stuck = min_constraint < 0.015
        if not np.any(stuck):
            break

        # Move stuck circles away from their tightest constraint
        improved = False
        for i in np.where(stuck)[0]:
            # Determine direction to move
            ml = margin_left[i]
            mr = margin_right[i]
            mb = margin_bottom[i]
            mt = margin_top[i]
            mg = min_gap[i]

            move_x = 0.0
            move_y = 0.0

            if mg <= ml and mg <= mr and mg <= mb and mg <= mt:
                # Nearest neighbor is the constraint
                # Find which neighbor
                best_j = -1
                best_d = float('inf')
                for j in range(n):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < best_d:
                        best_d = d
                        best_j = j
                if best_j >= 0:
                    dx = x[i] - x[best_j]
                    dy = y[i] - y[best_j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d > 1e-8:
                        move_x = dx / d
                        move_y = dy / d
            elif ml <= mr and ml <= mb and ml <= mt:
                move_x = 1.0
            elif mr <= ml and mr <= mb and mr <= mt:
                move_x = -1.0
            elif mb <= ml and mb <= mr and mb <= mt:
                move_y = 1.0
            else:
                move_y = -1.0

            # Try moves
            for step in [0.04, 0.02, 0.01, 0.005, 0.002]:
                new_x = x[i] + move_x * step
                new_y = y[i] + move_y * step
                if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                    x[i] = new_x
                    y[i] = new_y
                    improved = True
                    break
            else:
                # Perpendicular
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
    grow_factor = 1.001
    for _ in range(300):
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


def _newton_contact_polish(z, time_budget):
    """
    Newton-Raphson contact graph solver: identify active contacts (near-touching pairs
    and wall contacts), then solve the nonlinear system where all active constraints
    are exactly satisfied while maximizing sum of radii.
    
    Approach:
    1. Identify contact pairs (distance - (r_i + r_j) < tolerance) and wall contacts
    2. Build Jacobian for the system of equations:
       - For each contact pair (i,j): (x_i-x_j)^2 + (y_i-y_j)^2 = (r_i+r_j)^2
       - For each wall contact: x_i = r_i, x_i = w-r_i, y_i = r_i, y_i = h-r_i
    3. Solve for small perturbations in positions and radii that maintain contacts
       while increasing sum(r)
    4. Iterate: update, re-identify contacts, repeat
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    contact_tol = 0.005  # threshold for "active" contact
    max_iter = int(min(200, time_budget * 30))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        # Identify active contacts
        dx_all = x[:, None] - x[None, :]
        dy_all = y[:, None] - y[None, :]
        d_all = np.sqrt(dx_all * dx_all + dy_all * dy_all)
        np.fill_diagonal(d_all, np.inf)
        r_sum_all = r[:, None] + r[None, :]

        # Pair contacts: d[i,j] - (r[i]+r[j]) < contact_tol
        gap_all = d_all - r_sum_all
        np.fill_diagonal(gap_all, np.inf)

        # Wall contacts
        wall_left = x - r
        wall_right = w - x - r
        wall_bottom = y - r
        wall_top = h - y - r

        # Collect active constraints
        # Variables: dx[0..n-1], dy[0..n-1], dr[0..n-1] -> 3n variables
        # We want to find delta that keeps contacts active while increasing sum(r)
        
        # Build the system: for each active contact, the constraint residual should remain 0
        # After perturbation: (x_i+dx_i - x_j-dx_j)^2 + (y_i+dy_i - y_j-dy_j)^2 = (r_i+dr_i + r_j+dr_j)^2
        
        # Linearized: 2*(x_i-x_j)*(dx_i-dx_j) + 2*(y_i-y_j)*(dy_i-dy_j) = 2*(r_i+r_j)*(dr_i+dr_j)
        
        # Objective: maximize sum(dr_i) subject to contact constraints being maintained
        # and no new violations
        
        # Use a simpler approach: for each contact, compute the direction that
        # increases radii while maintaining the contact distance
        
        # For pair contact (i,j):
        #   (x_i-x_j)*dx_i + (y_i-y_j)*dy_i - (x_i-x_j)*dx_j - (y_i-y_j)*dy_j = (r_i+r_j)*(dr_i+dr_j)
        # To maximize dr_i + dr_j, we need to move i and j apart along the center line
        # while growing both radii.
        
        # Strategy: for each contact pair, compute the "growth direction"
        # and apply a small step. For wall contacts, grow radius toward the wall.
        
        growth = np.zeros(n)
        move_x = np.zeros(n)
        move_y = np.zeros(n)

        # Pair contacts
        iu = np.triu_indices(n, k=1)
        gap_pairs = gap_all[iu]
        active_pairs = gap_pairs < contact_tol

        for idx in np.where(active_pairs)[0]:
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            dx_ij = x[i_idx] - x[j_idx]
            dy_ij = y[i_idx] - y[j_idx]
            d_ij = np.sqrt(dx_ij * dx_ij + dy_ij * dy_ij)
            if d_ij < 1e-10:
                continue
            ux = dx_ij / d_ij
            uy = dy_ij / d_ij
            
            # Move i away from j, j away from i
            move_x[i_idx] += ux * 0.5
            move_y[i_idx] += uy * 0.5
            move_x[j_idx] -= ux * 0.5
            move_y[j_idx] -= uy * 0.5
            growth[i_idx] += 0.5
            growth[j_idx] += 0.5

        # Wall contacts
        for i in range(n):
            if wall_left[i] < contact_tol:
                move_x[i] += 0.5
                growth[i] += 0.5
            if wall_right[i] < contact_tol:
                move_x[i] -= 0.5
                growth[i] += 0.5
            if wall_bottom[i] < contact_tol:
                move_y[i] += 0.5
                growth[i] += 0.5
            if wall_top[i] < contact_tol:
                move_y[i] -= 0.5
                growth[i] += 0.5

        # Apply small step
        step_size = 0.002
        new_x = x + step_size * move_x
        new_y = y + step_size * move_y
        new_r = r + step_size * growth

        # Clamp
        new_x = np.clip(new_x, new_r, w - new_r)
        new_y = np.clip(new_y, new_r, h - new_r)
        new_r = np.maximum(new_r, 1e-4)

        # Check validity
        if _is_valid(new_x, new_y, new_r, w, h):
            x, y, r = new_x, new_y, new_r
        else:
            # Try smaller step
            for ss in [0.001, 0.0005, 0.0001]:
                new_x = x + ss * move_x
                new_y = y + ss * move_y
                new_r = r + ss * growth
                new_x = np.clip(new_x, new_r, w - new_r)
                new_y = np.clip(new_y, new_r, h - new_r)
                new_r = np.maximum(new_r, 1e-4)
                if _is_valid(new_x, new_y, new_r, w, h):
                    x, y, r = new_x, new_y, new_r
                    break
            else:
                break

    # Final uniform expansion
    grow_factor = 1.0005
    for _ in range(500):
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


def _smooth_epigraph_polish(z, time_budget):
    """
    Smooth epigraph formulation: introduce auxiliary variables for pairwise distances
    and wall margins, optimize with smooth constraints via SLSQP.
    This can escape local plateaus that discrete radius-grow gets stuck in.
    """
    z = z.copy().astype(float)
    n = N
    x0 = z[0::3][:n].copy()
    y0 = z[1::3][:n].copy()
    r0 = z[2::3][:n].copy()
    w0 = z[-1]
    h0 = 2.0 - w0

    # Variables: x(n), y(n), r(n), w(1)
    # Objective: maximize sum(r)
    # Constraints: smooth approximations of non-overlap and wall constraints

    def objective(zv):
        r = zv[2::3][:n]
        return -np.sum(r)

    bounds = [(0.01, 1.99)] * (2 * n) + [(0.005, 0.5)] * n + [(0.4, 1.6)]

    # Wall constraints (smooth): x_i - r_i >= 0, etc.
    # We use the standard inequality form

    def make_wall_left(i):
        return lambda p: p[3*i] - p[3*i+2]
    def make_wall_right(i):
        return lambda p: p[-1] - p[3*i] - p[3*i+2]
    def make_wall_bottom(i):
        return lambda p: p[3*i+1] - p[3*i+2]
    def make_wall_top(i):
        return lambda p: (2.0 - p[-1]) - p[3*i+1] - p[3*i+2]

    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': make_wall_left(i)})
        constraints.append({'type': 'ineq', 'fun': make_wall_right(i)})
        constraints.append({'type': 'ineq', 'fun': make_wall_bottom(i)})
        constraints.append({'type': 'ineq', 'fun': make_wall_top(i)})

    # Pairwise non-overlap: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    # Smooth version: use d^2 - (r1+r2)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            def make_overlap(idx1, idx2):
                return lambda p: (
                    (p[3*idx1] - p[3*idx2])**2 +
                    (p[3*idx1+1] - p[3*idx2+1])**2 -
                    (p[3*idx1+2] + p[3*idx2+2])**2
                )
            constraints.append({'type': 'ineq', 'fun': make_overlap(i, j)})

    maxiter = int(min(500, time_budget * 100))
    try:
        res = minimize(objective, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 120})
        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if _is_valid(x, y, r, w, h):
            return z_opt
        return None
    except Exception:
        return None

def _max_radius_for_circle(x, y, r, w, h, i):
    """Calculate maximum possible radius for circle i given other positions."""
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    for j in range(N):
        if i == j: continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx*dx + dy*dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)

def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    """Check if moving circle i to new_x, new_y is valid."""
    # Wall check
    if new_x < r[i] or new_x > w - r[i]:
        return False
    if new_y < r[i] or new_y > h - r[i]:
        return False
    
    # Neighbor check
    for j in range(N):
        if i == j: continue
        dx = new_x - x[j]
        dy = new_y - y[j]
        d2 = dx*dx + dy*dy
        r_sum = r[i] + r[j]
        if d2 < r_sum*r_sum - 1e-9:
            return False
    return True


def _constraint_refine(z, time_budget):
    """High-precision SLSQP refinement using explicit constraints."""
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

    def make_boundary_constraint(idx, axis, side):
        if axis == 0:
            if side == 0:
                return lambda p: p[3 * idx] - p[3 * idx + 2]
            else:
                return lambda p: p[-1] - p[3 * idx] - p[3 * idx + 2]
        else:
            if side == 0:
                return lambda p: p[3 * idx + 1] - p[3 * idx + 2]
            else:
                return lambda p: (2.0 - p[-1]) - p[3 * idx + 1] - p[3 * idx + 2]

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 0)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 1)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 0)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 1)})

    for i in range(n):
        for j in range(i + 1, n):
            def make_overlap_constraint(idx1, idx2):
                return lambda p: (
                    (p[3 * idx1] - p[3 * idx2]) ** 2 +
                    (p[3 * idx1 + 1] - p[3 * idx2 + 1]) ** 2 -
                    (p[3 * idx1 + 2] + p[3 * idx2 + 2]) ** 2
                )
            constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})

    maxiter = int(min(800, time_budget * 150))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if _is_valid(x, y, r, w, h):
            return z_opt
        return None
    except Exception:
        return None


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
