"""Circle packing in perimeter-4 rectangle: SLSQP + Enhanced Polish with Smooth Surrogate + Contact Graph Solver."""
import numpy as np
import time
from scipy.optimize import minimize, fsolve

N = 21


def _soft_max(v, beta=200.0):
    """Smooth approximation of max(v, 0) using log-sum-exp."""
    m = np.maximum(v, 0.0)
    return m + np.logaddexp(0.0, beta * (v - m)) / beta


def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth (soft-max) penalties for differentiability."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Vectorized pairwise overlap penalty (smooth)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap = _soft_max(r_sum_pairs - d_pairs, beta=200.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)

    obj += overlap_penalty

    # Vectorized boundary penalty (smooth)
    left_viol = _soft_max(r - x, beta=200.0)
    right_viol = _soft_max(x + r - w, beta=200.0)
    bottom_viol = _soft_max(r - y, beta=200.0)
    top_viol = _soft_max(y + r - h, beta=200.0)

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
    deadline = start_time + 145.0  # Safety margin before 150s kill

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

        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(10):
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

        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 25:
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

    # --- Phase 2: Enhanced Polish on best result ---
    if best_result is not None and time.time() < deadline - 30:
        # 0. Contact graph exact tangency solver
        contact_result = _contact_graph_solve(best_result, deadline - (time.time() - start_time) - 25)
        if contact_result is not None:
            valid, val = check_validity(contact_result)
            if valid and val > best_value:
                best_value = val
                best_result = contact_result
        
        # 0b. Contact graph solver v2 (SLSQP with equality constraints)
        if best_result is not None and time.time() < deadline - 22:
            contact_result2 = _contact_graph_solve_v2(best_result, deadline - (time.time() - start_time) - 20)
            if contact_result2 is not None:
                valid, val = check_validity(contact_result2)
                if valid and val > best_value:
                    best_value = val
                    best_result = contact_result2
        # 1. Force-directed position polish to ensure we are in a good basin
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 20)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Enhanced Expansion/Relocation Polish
        if best_result is not None and time.time() < deadline - 20:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 3. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 12:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 6)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 4. Final high-precision polish (maximize radii to limits)
        if best_result is not None and time.time() < deadline - 5:
            polished2 = _polish_result_tight(best_result, deadline - (time.time() - start_time) - 1)
            if polished2 is not None:
                valid, val = check_validity(polished2)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished2

        # 5. Multi-start local search: perturb and maximize radii
        if best_result is not None and time.time() < deadline - 1:
            best_local = _multi_start_local_search(best_result, deadline - (time.time() - start_time) - 0.5)
            if best_local is not None:
                valid, val = check_validity(best_local)
                if valid and val > best_value:
                    best_value = val
                    best_result = best_local

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


def _contact_graph_solve_v2(z, time_budget):
    """
    Alternative contact graph solver using scipy minimize with equality constraints.
    Maximizes sum of radii subject to exact tangency equations.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Identify contacts
    contact_tol = 0.01
    cc_contacts = []
    cw_contacts = []
    
    for i in range(n):
        if x[i] - r[i] < contact_tol:
            cw_contacts.append((i, 0))
        if w - x[i] - r[i] < contact_tol:
            cw_contacts.append((i, 1))
        if y[i] - r[i] < contact_tol:
            cw_contacts.append((i, 2))
        if h - y[i] - r[i] < contact_tol:
            cw_contacts.append((i, 3))
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            gap = d - (r[i] + r[j])
            if gap < contact_tol:
                cc_contacts.append((i, j))
    
    n_eq = len(cc_contacts) + len(cw_contacts)
    if n_eq < 3 * n - 2:
        return None

    def objective(vars):
        vr = vars[2*n:3*n]
        return -np.sum(vr)

    constraints = []
    for i, j in cc_contacts:
        def make_cc_eq(idx1, idx2):
            return lambda v: (v[idx1] - v[idx2]) ** 2 + (v[n+idx1] - v[n+idx2]) ** 2 - (v[2*n+idx1] + v[2*n+idx2]) ** 2
        constraints.append({'type': 'eq', 'fun': make_cc_eq(i, j)})
    
    for i, wall in cw_contacts:
        if wall == 0:
            def make_cw_eq(idx, w_=0):
                return lambda v: v[idx] - v[2*n+idx]
            constraints.append({'type': 'eq', 'fun': make_cw_eq(i)})
        elif wall == 1:
            def make_cw_eq(idx, w_=w):
                return lambda v: v[idx] + v[2*n+idx] - w_
            constraints.append({'type': 'eq', 'fun': make_cw_eq(i, w)})
        elif wall == 2:
            def make_cw_eq(idx, h_=0):
                return lambda v: v[n+idx] - v[2*n+idx]
            constraints.append({'type': 'eq', 'fun': make_cw_eq(i)})
        elif wall == 3:
            def make_cw_eq(idx, h_=h):
                return lambda v: v[n+idx] + v[2*n+idx] - h_
            constraints.append({'type': 'eq', 'fun': make_cw_eq(i, h)})

    bounds = []
    for i in range(n):
        bounds.append((0.01, w - 0.01))
    for i in range(n):
        bounds.append((0.01, h - 0.01))
    for i in range(n):
        bounds.append((0.01, min(w, h) * 0.45))

    vars0 = np.concatenate([x, y, r])

    try:
        res = minimize(objective, vars0, method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 500, 'ftol': 1e-14, 'maxls': 100})
        
        if res.success:
            sol_x = res.x[:n]
            sol_y = res.x[n:2*n]
            sol_r = res.x[2*n:3*n]
            
            if _is_valid(sol_x, sol_y, sol_r, w, h, tol=1e-7):
                val = float(np.sum(sol_r))
                orig_val = float(np.sum(r))
                if val > orig_val + 1e-8:
                    z_new = np.empty(3 * N + 1)
                    z_new[0::3][:n] = sol_x
                    z_new[1::3][:n] = sol_y
                    z_new[2::3][:n] = sol_r
                    z_new[-1] = w
                    return z_new
    except Exception:
        pass
    
    return None


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
    max_iter = int(min(3000, time_budget * 150))

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
    grow_factor = 1.0003
    for _ in range(300):
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
    Advanced polish: Iteratively expand circles with position optimization.
    Uses 8-direction local search to reposition circles for larger radii.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # 8 directions for local search
    directions = [
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (0.7071, 0.7071), (-0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, -0.7071)
    ]

    max_rounds = 40
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break

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
                for _ in range(25):
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
            # Find the most constraining direction
            min_gap = float('inf')
            best_dx, best_dy = 0.0, 0.0

            # Check walls
            d_left = x[i] - r[i]
            d_right = w - x[i] - r[i]
            d_bottom = y[i] - r[i]
            d_top = h - y[i] - r[i]

            if d_left < min_gap:
                min_gap = d_left
                best_dx, best_dy = 1.0, 0.0
            if d_right < min_gap:
                min_gap = d_right
                best_dx, best_dy = -1.0, 0.0
            if d_bottom < min_gap:
                min_gap = d_bottom
                best_dx, best_dy = 0.0, 1.0
            if d_top < min_gap:
                min_gap = d_top
                best_dx, best_dy = 0.0, -1.0

            # Check other circles
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_gap:
                    min_gap = gap
                    if d > 1e-6:
                        best_dx = dx / d
                        best_dy = dy / d

            # 3. 8-direction local search to find a better position
            if min_gap < 0.05:
                # Try moving in 8 directions with multiple step sizes
                found_better = False
                for dx_dir, dy_dir in directions:
                    if found_better:
                        break
                    for step in [0.04, 0.02, 0.01, 0.005, 0.002, 0.001]:
                        new_x = x[i] + dx_dir * step
                        new_y = y[i] + dy_dir * step
                        if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                            # Check if this move allows larger radius
                            old_r_max = _max_radius_for_circle(x, y, r, w, h, i)
                            x[i], y[i] = new_x, new_y
                            new_r_max = _max_radius_for_circle(x, y, r, w, h, i)
                            if new_r_max > old_r_max + 1e-6:
                                improved = True
                                found_better = True
                                break
                            else:
                                # Revert if no benefit
                                x[i], y[i] = x[i] - dx_dir * step, y[i] - dy_dir * step

                # 4. After repositioning, try to expand again
                if found_better:
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
                        for _ in range(25):
                            mid = (lo + hi) / 2.0
                            if _is_valid_move(x, y, np.where(np.arange(n) == i, mid, r), w, h, i, x[i], y[i]):
                                lo = mid
                            else:
                                hi = mid
                        new_r = lo
                        if new_r > r[i] + 1e-7:
                            r[i] = new_r
                            improved = True

        if not improved:
            break

    # Second pass: more aggressive expansion with position optimization
    if time.time() < time_budget - 5:
        for rnd in range(20):
            if time.time() > time_budget:
                break

            improved = False
            for i in range(n):
                if time.time() > time_budget:
                    break

                # Try to expand radius i
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

                if r_max > r[i] + 1e-5:
                    # Binary search for the true max valid radius
                    lo = r[i]
                    hi = r_max
                    for _ in range(30):
                        mid = (lo + hi) / 2.0
                        if _is_valid_move(x, y, np.where(np.arange(n) == i, mid, r), w, h, i, x[i], y[i]):
                            lo = mid
                        else:
                            hi = mid
                    new_r = lo
                    if new_r > r[i] + 1e-6:
                        r[i] = new_r
                        improved = True

            if not improved:
                break

    # Final uniform expansion with binary search
    grow_factor = 1.0005
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


def _max_radius_for_circle(x, y, r, w, h, i):
    """Calculate maximum possible radius for circle i given other positions."""
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


def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    """Check if moving circle i to new_x, new_y is valid."""
    # Wall check
    if new_x < r[i] or new_x > w - r[i]:
        return False
    if new_y < r[i] or new_y > h - r[i]:
        return False

    # Neighbor check
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
        h = z_opt[-1]

        if _is_valid(x, y, r, w, h):
            return z_opt
        return None
    except Exception:
        return None


def _polish_result_tight(z, time_budget):
    """High-precision final polish: maximize each circle to its limit, then local search."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Phase 1: Maximize each circle's radius to its limit (iterate until convergence)
    for outer_iter in range(50):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            # Compute max valid radius for circle i given all other positions/radii
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
            if r_max > r[i] + 1e-10:
                # Binary search for exact max valid radius
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
                if lo > r[i] + 1e-11:
                    r[i] = lo
                    improved = True
        if not improved:
            break

    # Phase 2: Local search - try moving circles to create more room
    directions = [
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (0.7071, 0.7071), (-0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, -0.7071)
    ]
    steps = [0.04, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]

    for ls_round in range(40):
        if time.time() > time_budget:
            break
        round_improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            # Current max radius for circle i
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max_cur = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max_cur:
                    r_max_cur = r_allowed
            r_max_cur = max(0.0, r_max_cur)

            # Try small moves to see if we can get a larger max radius
            best_new_r_max = r_max_cur
            best_new_pos = None
            for dx_dir, dy_dir in directions:
                for step in steps:
                    new_x = x[i] + dx_dir * step
                    new_y = y[i] + dy_dir * step
                    if new_x < r[i] or new_x > w - r[i] or new_y < r[i] or new_y > h - r[i]:
                        continue
                    # Check validity of move
                    valid_move = True
                    for j in range(n):
                        if i == j:
                            continue
                        dx = new_x - x[j]
                        dy = new_y - y[j]
                        if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                            valid_move = False
                            break
                    if not valid_move:
                        continue
                    # Compute new max radius at this position
                    r_wall_new = min(new_x, w - new_x, new_y, h - new_y)
                    r_max_new = r_wall_new
                    for j in range(n):
                        if i == j:
                            continue
                        dx = new_x - x[j]
                        dy = new_y - y[j]
                        d = np.sqrt(dx * dx + dy * dy)
                        r_allowed = d - r[j]
                        if r_allowed < r_max_new:
                            r_max_new = r_allowed
                    r_max_new = max(0.0, r_max_new)
                    if r_max_new > best_new_r_max:
                        best_new_r_max = r_max_new
                        best_new_pos = (new_x, new_y)

            if best_new_pos is not None and best_new_r_max > r_max_cur + 1e-8:
                x[i], y[i] = best_new_pos
                # Now expand to the new max
                lo = r[i]
                hi = best_new_r_max
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-10:
                    r[i] = lo
                    round_improved = True

        if not round_improved:
            break

    # Phase 3: Final uniform expansion
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0001
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


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid (vectorized)."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    return np.all(d_sq[iu] >= r_sum[iu] ** 2 - tol)


def _contact_graph_solve(z, time_budget):
    """
    Solve for exact tangency conditions using a contact graph approach.
    
    The idea: given a valid packing, identify which circles are in contact
    (nearly tangent) and which touch walls. Then solve the nonlinear system
    where all identified contacts are EXACT (distance = sum of radii, or
    center - radius = wall). This finds the exact local optimum for that
    contact configuration.
    
    Variables: x_i, y_i, r_i for each circle i (3N variables)
    Equations for each contact:
      - Circle-circle: (x_i - x_j)^2 + (y_i - y_j)^2 = (r_i + r_j)^2
      - Circle-wall (left): x_i = r_i
      - Circle-wall (right): x_i + r_i = w
      - Circle-wall (bottom): y_i = r_i
      - Circle-wall (top): y_i + r_i = h
    
    We use fsolve to find the exact solution, then verify validity.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Identify contacts based on proximity (within tolerance)
    contact_tol = 0.005  # Circles within this gap are considered "in contact"
    
    # Build contact list
    cc_contacts = []  # (i, j) circle-circle contacts
    cw_contacts = []  # (i, wall) circle-wall contacts, wall in {0:left, 1:right, 2:bottom, 3:top}
    
    for i in range(n):
        # Wall contacts
        if x[i] - r[i] < contact_tol:
            cw_contacts.append((i, 0))
        if w - x[i] - r[i] < contact_tol:
            cw_contacts.append((i, 1))
        if y[i] - r[i] < contact_tol:
            cw_contacts.append((i, 2))
        if h - y[i] - r[i] < contact_tol:
            cw_contacts.append((i, 3))
        
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            gap = d - (r[i] + r[j])
            if gap < contact_tol:
                cc_contacts.append((i, j))
    
    # We need at least 3N equations for 3N unknowns (or fewer, in which case
    # the system is underdetermined and we fix some variables)
    n_eq = len(cc_contacts) + len(cw_contacts)
    
    if n_eq < 3 * n - 2:  # Need at least 3N-2 equations (w and h are determined by perimeter)
        # Add more contacts: lower the threshold
        contact_tol = 0.02
        cc_contacts = []
        cw_contacts = []
        for i in range(n):
            if x[i] - r[i] < contact_tol:
                cw_contacts.append((i, 0))
            if w - x[i] - r[i] < contact_tol:
                cw_contacts.append((i, 1))
            if y[i] - r[i] < contact_tol:
                cw_contacts.append((i, 2))
            if h - y[i] - r[i] < contact_tol:
                cw_contacts.append((i, 3))
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < contact_tol:
                    cc_contacts.append((i, j))
        n_eq = len(cc_contacts) + len(cw_contacts)
    
    if n_eq < 3 * n - 2:
        # Not enough contacts; try with even larger tolerance
        contact_tol = 0.05
        cc_contacts = []
        cw_contacts = []
        for i in range(n):
            if x[i] - r[i] < contact_tol:
                cw_contacts.append((i, 0))
            if w - x[i] - r[i] < contact_tol:
                cw_contacts.append((i, 1))
            if y[i] - r[i] < contact_tol:
                cw_contacts.append((i, 2))
            if h - y[i] - r[i] < contact_tol:
                cw_contacts.append((i, 3))
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < contact_tol:
                    cc_contacts.append((i, j))
        n_eq = len(cc_contacts) + len(cw_contacts)
    
    if n_eq < 3 * n - 2:
        return None  # Not enough contact structure to solve
    
    # Variables: x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1} (3N vars)
    # w and h are fixed (perimeter constraint: w + h = 2)
    
    def equations(vars):
        vx = vars[:n]
        vy = vars[n:2*n]
        vr = vars[2*n:3*n]
        eqs = []
        
        for i, j in cc_contacts:
            dx = vx[i] - vx[j]
            dy = vy[i] - vy[j]
            eqs.append(dx * dx + dy * dy - (vr[i] + vr[j]) ** 2)
        
        for i, wall in cw_contacts:
            if wall == 0:  # left
                eqs.append(vx[i] - vr[i])
            elif wall == 1:  # right
                eqs.append(vx[i] + vr[i] - w)
            elif wall == 2:  # bottom
                eqs.append(vy[i] - vr[i])
            elif wall == 3:  # top
                eqs.append(vy[i] + vr[i] - h)
        
        return eqs
    
    # Initial guess
    vars0 = np.concatenate([x, y, r])
    
    # Solve
    try:
        sol, info, ier, msg = fsolve(equations, vars0, full_output=True, xtol=1e-12, maxfev=10000)
        
        if ier != 1:
            # Try with different initial perturbation
            rng = np.random.default_rng(7)
            for attempt in range(3):
                if time.time() > time_budget:
                    break
                perturbed = vars0 + rng.normal(0, 0.01, size=vars0.shape)
                # Clamp to valid range
                for i in range(n):
                    perturbed[i] = np.clip(perturbed[i], 0.01, w - 0.01)
                    perturbed[n + i] = np.clip(perturbed[n + i], 0.01, h - 0.01)
                    perturbed[2*n + i] = np.clip(perturbed[2*n + i], 0.01, min(w, h) * 0.4)
                sol, info, ier, msg = fsolve(equations, perturbed, full_output=True, xtol=1e-12, maxfev=5000)
                if ier == 1:
                    break
        
        if ier != 1:
            return None
        
        sol_x = sol[:n]
        sol_y = sol[n:2*n]
        sol_r = sol[2*n:3*n]
        
        # Check validity
        if _is_valid(sol_x, sol_y, sol_r, w, h, tol=1e-7):
            val = float(np.sum(sol_r))
            orig_val = float(np.sum(r))
            if val > orig_val + 1e-8:
                z_new = np.empty(3 * N + 1)
                z_new[0::3][:n] = sol_x
                z_new[1::3][:n] = sol_y
                z_new[2::3][:n] = sol_r
                z_new[-1] = w
                return z_new
        else:
            # Try to repair by shrinking slightly
            for factor in [0.999, 0.998, 0.995, 0.99, 0.98]:
                sol_r_shrunk = sol_r * factor
                if _is_valid(sol_x, sol_y, sol_r_shrunk, w, h, tol=1e-7):
                    val = float(np.sum(sol_r_shrunk))
                    orig_val = float(np.sum(r))
                    if val > orig_val + 1e-8:
                        z_new = np.empty(3 * N + 1)
                        z_new[0::3][:n] = sol_x
                        z_new[1::3][:n] = sol_y
                        z_new[2::3][:n] = sol_r_shrunk
                        z_new[-1] = w
                        return z_new
                    break
    except Exception:
        pass
    
    return None


def _multi_start_local_search(z, time_budget):
    """Multi-start local search: perturb and maximize radii."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    best_val = float(np.sum(r))
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    rng = np.random.default_rng(123)
    directions = [
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (0.7071, 0.7071), (-0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, -0.7071)
    ]

    for trial in range(30):
        if time.time() > time_budget:
            break

        x_p = x.copy()
        y_p = y.copy()
        r_p = r.copy()

        # Perturb a random subset
        n_perturb = rng.integers(2, 7)
        indices = rng.choice(n, size=n_perturb, replace=False)
        for idx in indices:
            dx_dir, dy_dir = directions[rng.integers(0, 8)]
            step = rng.uniform(0.005, 0.03)
            new_x = x_p[idx] + dx_dir * step
            new_y = y_p[idx] + dy_dir * step
            if _is_valid_move(x_p, y_p, r_p, w, h, idx, new_x, new_y):
                x_p[idx], y_p[idx] = new_x, new_y

        # Maximize all radii (iterate to convergence)
        for outer_iter in range(20):
            improved = False
            for i in range(n):
                if time.time() > time_budget:
                    break
                r_wall = min(x_p[i], w - x_p[i], y_p[i], h - y_p[i])
                r_max = r_wall
                for j in range(n):
                    if i == j:
                        continue
                    dx = x_p[i] - x_p[j]
                    dy = y_p[i] - y_p[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    r_allowed = d - r_p[j]
                    if r_allowed < r_max:
                        r_max = r_allowed
                r_max = max(0.0, r_max)
                if r_max > r_p[i] + 1e-10:
                    lo = r_p[i]
                    hi = r_max
                    for _ in range(15):
                        mid = (lo + hi) / 2.0
                        r_test = r_p.copy()
                        r_test[i] = mid
                        if _is_valid(x_p, y_p, r_test, w, h):
                            lo = mid
                        else:
                            hi = mid
                    if lo > r_p[i] + 1e-11:
                        r_p[i] = lo
                        improved = True
            if not improved:
                break

        val_p = float(np.sum(r_p))
        if _is_valid(x_p, y_p, r_p, w, h) and val_p > best_val:
            best_val = val_p
            best_x, best_y, best_r = x_p.copy(), y_p.copy(), r_p.copy()

    if best_val > float(np.sum(r)) + 1e-12:
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = best_x
        z_new[1::3][:n] = best_y
        z_new[2::3][:n] = best_r
        z_new[-1] = w
        return z_new
    return None
