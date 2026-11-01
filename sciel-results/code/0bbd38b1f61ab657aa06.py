"""Circle packing in perimeter-4 rectangle: SLSQP + QP Relaxation + Epigraph + Polish."""
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
    deadline = start_time + 140.0  # Safety margin before 150s kill

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

    # --- QP Relaxation: Linearized convex surrogate for better starting points ---
    if time.time() < deadline - 40:
        qp_starts = _qp_relaxation_starts(guesses[:3], deadline - (time.time() - start_time) - 30)
        for z_qp in qp_starts:
            if time.time() > deadline - 30:
                break
            z_qp_relaxed = _force_relax_unified(z_qp, max_iter=300)
            remaining = deadline - (time.time() - start_time)
            maxiter = int(min(400, max(100, remaining * 40)))
            try:
                res = minimize(objective, z_qp_relaxed, method='SLSQP',
                               bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                               options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})
                z_opt = res.x
                valid, val = check_validity(z_opt)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_opt.copy()
            except Exception:
                continue

    # --- Phase 2: Enhanced Polish on best result ---
    if best_result is not None and time.time() < deadline - 25:
        # 1. Force-directed position polish to ensure we are in a good basin
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 15)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Epigraph-based smooth refinement (balances circle sizes)
        if best_result is not None and time.time() < deadline - 18:
            epi = _epigraph_refine(best_result, deadline - (time.time() - start_time) - 10)
            if epi is not None:
                valid, val = check_validity(epi)
                if valid and val > best_value:
                    best_value = val
                    best_result = epi

        # 3. Expansion/Relocation Polish with enhanced local search
        if best_result is not None and time.time() < deadline - 12:
            expanded = _expansion_relaxation_polish_enhanced(best_result, deadline - (time.time() - start_time) - 6)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 4. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 8:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 4)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 5. Greedy expansion polish (aggressive per-circle)
        if best_result is not None and time.time() < deadline - 5:
            greedy = _greedy_expansion(best_result, deadline - (time.time() - start_time) - 2)
            if greedy is not None:
                valid, val = check_validity(greedy)
                if valid and val > best_value:
                    best_value = val
                    best_result = greedy

        # 6. Final quick polish
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
    for _ in range(20):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    # Final per-circle expansion pass with very tight tolerance
    # This is the key improvement: squeeze out remaining gains by expanding each circle as much as possible
    for i in range(N):
        if time.time() > deadline - 1:
            break
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
        r_max = max(0.0, r_max)
        if r_max > r[i] + 1e-8:
            lo = r[i]
            hi = r_max
            # Increase binary search steps for precision
            for _ in range(30):
                mid = (lo + hi) / 2.0
                if _is_valid_move(x, y, np.where(np.arange(N) == i, mid, r), w, h, i, x[i], y[i]):
                    lo = mid
                else:
                    hi = mid
            new_r = lo
            if new_r > r[i] + 1e-9:
                r[i] = new_r

    return np.column_stack((x, y, r))


def _qp_relaxation_starts(initial_guesses, time_budget):
    """
    Generate improved starting points via linearized QP relaxation.
    
    For each initial guess, we linearize the circle-circle non-overlap constraints
    via first-order Taylor expansion around the current positions, creating a convex
    QP: maximize sum(r_i) subject to linearized distance constraints and bounds.
    The solution gives a better starting point for SLSQP.
    """
    starts = []
    n = N
    
    for z0 in initial_guesses:
        if time.time() > time_budget:
            break
        
        x = z0[0::3][:n].copy()
        y = z0[1::3][:n].copy()
        r = z0[2::3][:n].copy()
        w = z0[-1]
        h = 2.0 - w
        
        if w < 0.2 or h < 0.2:
            continue
        
        # Linearize circle-circle constraints around current positions
        # Constraint: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        # Linearize around (x_i0, y_i0, r_i0, x_j0, y_j0, r_j0):
        # Let dx = x_i - x_j, dy = y_i - y_j, d = sqrt(dx^2+dy^2)
        # Linearized: dx0*(x_i-x_j) + dy0*(y_i-y_j) + d0*(r_i+r_j) >= (r_i0+r_j0)^2 + d0*(r_i0+r_j0)
        # Simplified: dx0*(x_i-x_j) + dy0*(y_i-y_j) + d0*(r_i+r_j) >= d0*(r_i0+r_j0) + (r_i0+r_j0)^2 - d0^2
        # Actually simpler: project onto the tangent plane
        # dx0/d0 * (x_i - x_j) + dy0/d0 * (y_i - y_j) + (r_i + r_j) >= (r_i0 + r_j0) + (r_i0+r_j0)^2/d0 - d0
        # Let's use: u_ij*(x_i-x_j) + v_ij*(y_i-y_j) + r_i + r_j >= c_ij
        # where u_ij = (x_i0-x_j0)/d0, v_ij = (y_i0-y_j0)/d0, c_ij = (r_i0+r_j0) + (r_i0+r_j0)^2/d0 - d0
        
        # Build QP: minimize -sum(r_i) subject to linear constraints
        # Variables: [x_0, y_0, r_0, x_1, y_1, r_1, ..., x_n-1, y_n-1, r_n-1]
        # n_vars = 3*n
        n_vars = 3 * n
        
        # Objective: minimize -sum(r_i)
        c_obj = np.zeros(n_vars)
        for i in range(n):
            c_obj[3*i + 2] = -1.0
        
        # Inequality constraints: A_ineq * vars >= b_ineq (scipy uses A @ x <= b, so flip)
        # We'll use scipy minimize with SLSQP and linear constraints
        
        constraints = []
        
        # Circle-circle linearized constraints
        for i in range(n):
            for j in range(i + 1, n):
                dx0 = x[i] - x[j]
                dy0 = y[i] - y[j]
                d0 = np.sqrt(dx0*dx0 + dy0*dy0)
                if d0 < 1e-6:
                    d0 = 1e-6
                    ux, uy = 1.0, 0.0
                else:
                    ux = dx0 / d0
                    uy = dy0 / d0
                
                r_sum0 = r[i] + r[j]
                # Linearized: ux*(x_i-x_j) + uy*(y_i-y_j) + r_i + r_j >= r_sum0 + r_sum0^2/d0 - d0
                # i.e., ux*x_i - ux*x_j + uy*y_i - uy*y_j + r_i + r_j >= c
                # scipy wants A @ x <= b, so: -ux*x_i + ux*x_j - uy*y_i + uy*y_j - r_i - r_j <= -c
                c_val = r_sum0 + r_sum0*r_sum0/d0 - d0
                a_row = np.zeros(n_vars)
                a_row[3*i] = -ux
                a_row[3*i+1] = -uy
                a_row[3*i+2] = -1.0
                a_row[3*j] = ux
                a_row[3*j+1] = uy
                a_row[3*j+2] = 1.0
                constraints.append({'type': 'ineq', 'fun': lambda v, a=a_row, c=c_val: c - a @ v})
        
        # Wall constraints: x_i >= r_i, x_i <= w - r_i, y_i >= r_i, y_i <= h - r_i
        for i in range(n):
            # x_i - r_i >= 0  =>  -(x_i - r_i) <= 0
            a = np.zeros(n_vars)
            a[3*i] = -1.0
            a[3*i+2] = 1.0
            constraints.append({'type': 'ineq', 'fun': lambda v, a=a: 0.0 - a @ v})
            
            # w - x_i - r_i >= 0  =>  x_i + r_i - w <= 0
            a = np.zeros(n_vars)
            a[3*i] = 1.0
            a[3*i+2] = 1.0
            constraints.append({'type': 'ineq', 'fun': lambda v, a=a, w=w: a @ v - w})
            
            # y_i - r_i >= 0
            a = np.zeros(n_vars)
            a[3*i+1] = -1.0
            a[3*i+2] = 1.0
            constraints.append({'type': 'ineq', 'fun': lambda v, a=a: 0.0 - a @ v})
            
            # h - y_i - r_i >= 0
            a = np.zeros(n_vars)
            a[3*i+1] = 1.0
            a[3*i+2] = 1.0
            constraints.append({'type': 'ineq', 'fun': lambda v, a=a, h=h: a @ v - h})
        
        # Bounds
        bounds = []
        for i in range(n):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((0.001, 1.0))
        
        # Solve QP via SLSQP (it handles linear constraints well for convex problems)
        try:
            res = minimize(lambda v: c_obj @ v, z0[:3*n], method='SLSQP',
                           bounds=bounds,
                           constraints=constraints,
                           options={'maxiter': 200, 'ftol': 1e-10, 'maxls': 50})
            
            z_qp = res.x.copy()
            z_qp = np.concatenate([z_qp, [w]])
            
            # Check if this gives a valid packing (linearized constraints may be violated)
            x_q = z_qp[0::3][:n]
            y_q = z_qp[1::3][:n]
            r_q = z_qp[2::3][:n]
            
            if _is_valid(x_q, y_q, r_q, w, h):
                starts.append(z_qp.copy())
            else:
                # Use it as a starting point even if slightly invalid; force_relax will fix
                starts.append(z_qp.copy())
                
        except Exception:
            continue
    
    return starts


def _epigraph_refine(z, time_budget):
    """
    Epigraph formulation: minimize t subject to r_i <= t for all i,
    plus all packing constraints. This balances circle sizes and can escape
    local optima where some circles are much smaller than others.
    
    We use a smooth log-sum-exp penalty for the epigraph constraint.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    
    # Use log-sum-exp to create a smooth surrogate for max(r_i)
    # LSE(r/s) * s approximates max(r_i) when s is large
    s = 50.0  # sharpness parameter
    
    def lse_penalty(rv):
        """Smooth max via log-sum-exp."""
        rv_clipped = np.clip(rv, 1e-8, None)
        max_rv = np.max(rv_clipped)
        return max_rv + np.log(np.sum(np.exp((rv_clipped - max_rv) / s))) * s / n
    
    def epi_objective(zv):
        xv = zv[0::3][:n]
        yv = zv[1::3][:n]
        rv = zv[2::3][:n]
        wv = zv[-1]
        hv = 2.0 - wv
        
        if wv < 0.2 or hv < 0.2 or np.any(rv < 0):
            return 1e6
        
        # Primary: minimize sum(r)
        obj = -np.sum(rv)
        
        # Secondary: penalize imbalance via LSE (encourages uniform sizes)
        # We want to minimize max(r_i) as a tiebreaker, but actually we want to MAXIMIZE min(r_i)
        # So we penalize small circles: use -LSE(-r/s)*s which approximates -max(-r) = min(r)
        # Actually, let's use a different approach: penalize variance of radii
        r_mean = np.mean(rv)
        variance_penalty = 0.01 * np.sum((rv - r_mean)**2)
        obj += variance_penalty
        
        # Overlap penalty
        dx = xv[:, None] - xv[None, :]
        dy = yv[:, None] - yv[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = rv[:, None] + rv[None, :]
        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
        overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
        obj += 1000.0 * np.sum(overlap * overlap)
        
        # Boundary penalty
        left_viol = np.maximum(rv - xv, 0.0)
        right_viol = np.maximum(xv + rv - wv, 0.0)
        bottom_viol = np.maximum(rv - yv, 0.0)
        top_viol = np.maximum(yv + rv - hv, 0.0)
        obj += 1000.0 * (np.sum(left_viol**2) + np.sum(right_viol**2) + 
                         np.sum(bottom_viol**2) + np.sum(top_viol**2))
        
        return obj
    
    try:
        res = minimize(epi_objective, z, method='SLSQP',
                       bounds=[(0, 2)] * (3 * n) + [(0.3, 1.7)],
                       options={'maxiter': 300, 'ftol': 1e-12, 'maxls': 80})
        
        z_opt = res.x
        x_e = z_opt[0::3][:n]
        y_e = z_opt[1::3][:n]
        r_e = z_opt[2::3][:n]
        w_e = z_opt[-1]
        h_e = 2.0 - w_e
        
        if _is_valid(x_e, y_e, r_e, w_e, h_e):
            return z_opt
        return None
    except Exception:
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


def _expansion_relaxation_polish_enhanced(z, time_budget):
    """
    Enhanced polish: Iteratively expand circles with improved position optimization.
    Uses multi-direction local search with adaptive step sizes for better exploration.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Enhanced directions: 12 directions for better coverage
    directions = [
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (0.7071, 0.7071), (-0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, -0.7071),
        (0.866, 0.5), (-0.866, 0.5), (0.866, -0.5), (-0.866, -0.5)
    ]

    max_rounds = 50
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
                for _ in range(30):
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
            
            # 3. Enhanced 12-direction local search to find a better position
            if min_gap < 0.05:
                # Try moving in 12 directions with multiple step sizes
                found_better = False
                for dx_dir, dy_dir in directions:
                    if found_better:
                        break
                    for step in [0.05, 0.03, 0.015, 0.008, 0.004, 0.002, 0.001]:
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
                        for _ in range(30):
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
        for rnd in range(30):
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
                    for _ in range(35):
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
    """High-precision SLSQP refinement using explicit constraints with tighter tolerances."""
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


def _greedy_expansion(z, time_budget):
    """
    Greedy per-circle expansion with position optimization.
    Iteratively tries to expand each circle as much as possible,
    using local position search when expansion is blocked.
    More aggressive than the enhanced polish for final gains.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    
    # Sort circles by current radius (expand small ones first)
    order = np.argsort(r)
    
    max_rounds = 80
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        
        improved = False
        for idx in order:
            i = idx
            if time.time() > time_budget:
                break
            
            # Compute max possible radius for circle i
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
                # Binary search for max valid radius
                lo = r[i]
                hi = r_max
                for _ in range(35):
                    mid = (lo + hi) / 2.0
                    if _is_valid_move(x, y, np.where(np.arange(n) == i, mid, r), w, h, i, x[i], y[i]):
                        lo = mid
                    else:
                        hi = mid
                new_r = lo
                if new_r > r[i] + 1e-7:
                    r[i] = new_r
                    improved = True
            
            # If expansion is limited, try to move circle i to free space
            min_gap = float('inf')
            best_dx, best_dy = 0.0, 0.0
            
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
            
            # Try 8 directions with adaptive steps
            if min_gap < 0.03:
                directions = [(1,0),(-1,0),(0,1),(0,-1),(0.707,0.707),(-0.707,0.707),(0.707,-0.707),(-0.707,-0.707)]
                for dx_dir, dy_dir in directions:
                    for step in [0.02, 0.01, 0.005, 0.002, 0.001]:
                        new_x = x[i] + dx_dir * step
                        new_y = y[i] + dy_dir * step
                        if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                            old_r_max = _max_radius_for_circle(x, y, r, w, h, i)
                            x[i], y[i] = new_x, new_y
                            new_r_max = _max_radius_for_circle(x, y, r, w, h, i)
                            if new_r_max > old_r_max + 1e-5:
                                improved = True
                                # Expand after move
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
                                    for _ in range(30):
                                        mid = (lo + hi) / 2.0
                                        if _is_valid_move(x, y, np.where(np.arange(n) == i, mid, r), w, h, i, x[i], y[i]):
                                            lo = mid
                                        else:
                                            hi = mid
                                    r[i] = lo
                                break
                            else:
                                x[i], y[i] = x[i] - dx_dir * step, y[i] - dy_dir * step
    
        if not improved:
            break
    
    # Final uniform expansion
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0003
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
