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

    # PBD-based initializer: produces valid packings with good contact structure
    def pbd_init(w0, h0, target_r, max_iters=2000):
        """Position-Based Dynamics: iteratively project to satisfy constraints."""
        n = N
        side = int(np.ceil(np.sqrt(n)))
        x = np.zeros(n)
        y = np.zeros(n)
        r = np.full(n, target_r)

        for i in range(n):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side * w0
            y[i] = (row + 0.5) / side * h0

        # PBD iterations
        for it in range(max_iters):
            # Wall constraints (project)
            for i in range(n):
                if x[i] < r[i]:
                    x[i] = r[i]
                elif x[i] > w0 - r[i]:
                    x[i] = w0 - r[i]
                if y[i] < r[i]:
                    y[i] = r[i]
                elif y[i] > h0 - r[i]:
                    y[i] = h0 - r[i]

            # Pairwise overlap constraints (project)
            max_overlap = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    r_sum = r[i] + r[j]
                    if d < r_sum - 1e-12:
                        if d < 1e-12:
                            # Degenerate: separate arbitrarily
                            x[i] += 0.01
                            y[i] += 0.01
                            x[j] -= 0.01
                            y[j] -= 0.01
                        else:
                            overlap = r_sum - d
                            ux = dx / d
                            uy = dy / d
                            x[i] += 0.5 * overlap * ux
                            y[i] += 0.5 * overlap * uy
                            x[j] -= 0.5 * overlap * ux
                            y[j] -= 0.5 * overlap * uy
                        max_overlap = max(max_overlap, overlap)
            if max_overlap < 1e-10:
                break

        # Clamp final positions
        for i in range(n):
            x[i] = np.clip(x[i], r[i], w0 - r[i])
            y[i] = np.clip(y[i], r[i], h0 - r[i])

        if _is_valid(x, y, r, w0, h0, tol=1e-9):
            z = np.empty(3 * N + 1)
            z[0::3][:n] = x
            z[1::3][:n] = y
            z[2::3][:n] = r
            z[-1] = w0
            return z
        return None

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

        # PBD-based initializations with various aspect ratios and radii
        pbd_configs = [
            (1.0, 0.10), (0.9, 0.10), (1.1, 0.10), (0.8, 0.09), (1.2, 0.09),
            (1.0, 0.11), (0.9, 0.11), (1.1, 0.11), (0.85, 0.10), (1.15, 0.10),
            (1.0, 0.12), (0.95, 0.11), (1.05, 0.11),
        ]
        for w0, tr in pbd_configs:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z_pbd = pbd_init(w0, h0, tr)
            if z_pbd is not None:
                guesses.append(z_pbd)

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

        # 3. Smooth log-sum-exp epigraph refinement (better-conditioned, can escape non-convex traps)
        if best_result is not None and time.time() < deadline - 18:
            lse_result = _lse_refine(best_result, deadline - (time.time() - start_time) - 12)
            if lse_result is not None:
                valid, val = check_validity(lse_result)
                if valid and val > best_value:
                    best_value = val
                    best_result = lse_result

        # 3b. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 10:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 5)
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

        # 6. Final high-precision radius maximization
        if best_result is not None and time.time() < deadline - 0.5:
            x = best_result[0::3][:N]
            y = best_result[1::3][:N]
            r = best_result[2::3][:N]
            w = best_result[-1]
            h = 2.0 - w
            # Aggressive per-circle radius maximization
            for _iter in range(30):
                if time.time() > deadline - 0.5:
                    break
                improved = False
                for i in range(N):
                    if time.time() > deadline - 0.5:
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
                    if r_max > r[i] + 1e-12:
                        lo, hi = r[i], r_max
                        for _ in range(30):
                            mid = (lo + hi) / 2.0
                            r_test = r.copy()
                            r_test[i] = mid
                            if _is_valid(x, y, r_test, w, h, tol=1e-10):
                                lo = mid
                            else:
                                hi = mid
                        if lo > r[i] + 1e-13:
                            r[i] = lo
                            improved = True
                if not improved:
                    break
            if _is_valid(x, y, r, w, h, tol=1e-10):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    z_new = np.empty(3 * N + 1)
                    z_new[0::3][:N] = x
                    z_new[1::3][:N] = y
                    z_new[2::3][:N] = r
                    z_new[-1] = w
                    best_result = z_new

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
        if _is_valid(x, y, r_trial, w, h, tol=1e-10):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))


def _lse_refine(z, time_budget):
    """
    Smooth log-sum-exp epigraph refinement with adaptive penalty.
    Uses a two-phase approach: first relax, then tighten.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    iu = np.triu_indices(n, k=1)

    def lse_objective(v, penalty):
        vx = v[0:n]
        vy = v[n:2*n]
        vr = v[2*n:3*n]
        vw = v[3*n]
        vh = 2.0 - vw

        if vw < 0.2 or vh < 0.2:
            return 1e6

        obj = -np.sum(vr)

        # Pairwise: smooth penalty
        dx = vx[iu[0]] - vx[iu[1]]
        dy = vy[iu[0]] - vy[iu[1]]
        d = np.sqrt(dx*dx + dy*dy + 1e-12)
        rsum = vr[iu[0]] + vr[iu[1]]
        viol = rsum - d
        beta = 60.0
        smooth_viol = np.maximum(viol, 0.0)
        obj += penalty * np.sum(smooth_viol * smooth_viol)

        # Wall penalties (smooth)
        lw = np.maximum(vr - vx, 0.0)
        rw = np.maximum(vx + vr - vw, 0.0)
        bw = np.maximum(vr - vy, 0.0)
        tw = np.maximum(vy + vr - vh, 0.0)
        obj += penalty * (np.sum(lw**2) + np.sum(rw**2) + np.sum(bw**2) + np.sum(tw**2))

        # Aspect ratio penalty
        if vw < 0.5:
            obj += penalty * (0.5 - vw)**2
        elif vw > 1.5:
            obj += penalty * (vw - 1.5)**2

        return obj

    bounds = []
    for i in range(n):
        bounds.append((0.01, 1.99))
    for i in range(n):
        bounds.append((0.01, 1.99))
    for i in range(n):
        bounds.append((0.01, 0.6))
    bounds.append((0.4, 1.6))

    v0 = np.concatenate([x, y, r, [w]])
    best_v = v0.copy()
    best_val = float(np.sum(r))

    # Phase 1: moderate penalty
    penalty1 = 300.0
    maxiter1 = int(min(400, time_budget * 50))
    try:
        res = minimize(lambda v: lse_objective(v, penalty1), v0, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter1, 'ftol': 1e-13, 'maxls': 60})
        v1 = res.x
        if _is_valid(v1[:n], v1[n:2*n], v1[2*n:3*n], v1[3*n], 2.0 - v1[3*n], tol=1e-7):
            val1 = float(np.sum(v1[2*n:3*n]))
            if val1 > best_val:
                best_val = val1
                best_v = v1.copy()
    except Exception:
        pass

    # Phase 2: high penalty from best phase 1 result
    if time.time() < time_budget - 3:
        penalty2 = 2000.0
        maxiter2 = int(min(300, (time_budget - (time.time() - (time_budget - time_budget))) * 40))
        maxiter2 = int(min(300, time_budget * 30))
        try:
            res2 = minimize(lambda v: lse_objective(v, penalty2), best_v, method='SLSQP',
                            bounds=bounds,
                            options={'maxiter': maxiter2, 'ftol': 1e-14, 'maxls': 60})
            v2 = res2.x
            if _is_valid(v2[:n], v2[n:2*n], v2[2*n:3*n], v2[3*n], 2.0 - v2[3*n], tol=1e-8):
                val2 = float(np.sum(v2[2*n:3*n]))
                if val2 > best_val:
                    best_val = val2
                    best_v = v2.copy()
        except Exception:
            pass

    if best_val > float(np.sum(r)) + 1e-9:
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = best_v[:n]
        z_new[1::3][:n] = best_v[n:2*n]
        z_new[2::3][:n] = best_v[2*n:3*n]
        z_new[-1] = best_v[3*n]
        return z_new
    return None


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
        fx += left_viol * 100
        fx -= right_viol * 100
        fy += bottom_viol * 100
        fy -= top_viol * 100

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
    Advanced polish: coordinate ascent with push-neighbors and adaptive repositioning.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    def _max_r(xi, yi, x_arr, y_arr, r_arr, W, H):
        r_wall = min(xi, W - xi, yi, H - yi)
        r_max = r_wall
        for j in range(n):
            dx = xi - x_arr[j]
            dy = yi - y_arr[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - r_arr[j]
            if r_allowed < r_max:
                r_max = r_allowed
        return max(0.0, r_max)

    def _valid_fast(xi, yi, ri, x_arr, y_arr, r_arr, W, H, skip):
        if xi < ri or xi > W - ri or yi < ri or yi > H - ri:
            return False
        for j in range(n):
            if j == skip:
                continue
            dx = xi - x_arr[j]
            dy = yi - y_arr[j]
            rs = ri + r_arr[j]
            if dx * dx + dy * dy < rs * rs - 1e-9:
                return False
        return True

    # Phase 1: Coordinate ascent with repositioning
    max_rounds = 35
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            # Expand radius i
            r_max = _max_r(x[i], y[i], x, y, r, w, h)
            if r_max > r[i] + 1e-8:
                lo, hi = r[i], r_max
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    if _valid_fast(x[i], y[i], mid, x, y, r, w, h, i):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-9:
                    r[i] = lo
                    improved = True

            # Find most constraining direction and try to move
            r_max_cur = _max_r(x[i], y[i], x, y, r, w, h)
            min_gap = float('inf')
            best_dx, best_dy = 0.0, 0.0
            for j in range(n):
                if j == i:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_gap:
                    min_gap = gap
                    if d > 1e-6:
                        best_dx, best_dy = dx / d, dy / d
            for dx_w, dy_w in [(1,0),(-1,0),(0,1),(0,-1)]:
                if dx_w > 0: g = x[i] - r[i]
                elif dx_w < 0: g = w - x[i] - r[i]
                elif dy_w > 0: g = y[i] - r[i]
                else: g = h - y[i] - r[i]
                if g < min_gap:
                    min_gap = g
                    best_dx, best_dy = float(dx_w), float(dy_w)

            if min_gap < 0.04:
                for step in [0.03, 0.015, 0.008, 0.004, 0.002, 0.001]:
                    nx = x[i] + best_dx * step
                    ny = y[i] + best_dy * step
                    if _valid_fast(nx, ny, r[i], x, y, r, w, h, i):
                        new_r_max = _max_r(nx, ny, x, y, r, w, h)
                        if new_r_max > r_max_cur + 1e-7:
                            x[i], y[i] = nx, ny
                            lo, hi = r[i], new_r_max
                            for _ in range(18):
                                mid = (lo + hi) / 2.0
                                if _valid_fast(x[i], y[i], mid, x, y, r, w, h, i):
                                    lo = mid
                                else:
                                    hi = mid
                            if lo > r[i] + 1e-9:
                                r[i] = lo
                                improved = True
                            break

        if not improved:
            break

    # Phase 2: Multi-direction search for additional gains
    directions = [
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (0.7071, 0.7071), (-0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, -0.7071)
    ]
    steps = [0.04, 0.02, 0.01, 0.005, 0.002, 0.001]

    for ls_round in range(25):
        if time.time() > time_budget:
            break
        round_improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            r_max_cur = _max_r(x[i], y[i], x, y, r, w, h)
            best_gain = 0.0
            best_pos = None
            for dx_dir, dy_dir in directions:
                for step in steps:
                    nx = x[i] + dx_dir * step
                    ny = y[i] + dy_dir * step
                    if _valid_fast(nx, ny, r[i], x, y, r, w, h, i):
                        new_r_max = _max_r(nx, ny, x, y, r, w, h)
                        if new_r_max - r_max_cur > best_gain:
                            best_gain = new_r_max - r_max_cur
                            best_pos = (nx, ny)
            if best_pos is not None and best_gain > 1e-7:
                x[i], y[i] = best_pos
                lo, hi = r[i], r_max_cur + best_gain
                for _ in range(18):
                    mid = (lo + hi) / 2.0
                    if _valid_fast(x[i], y[i], mid, x, y, r, w, h, i):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-9:
                    r[i] = lo
                    round_improved = True
        if not round_improved:
            break

    # Phase 3: Final uniform expansion
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0003
        if _is_valid(x, y, r_trial, w, h, tol=1e-9):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h, tol=1e-9):
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
    """High-precision SLSQP refinement using smooth penalty + boundary constraints only."""
    z = z.copy().astype(float)
    n = N

    def obj_refine(zv):
        vx = zv[0::3][:n]
        vy = zv[1::3][:n]
        vr = zv[2::3][:n]
        vw = zv[-1]
        vh = 2.0 - vw
        if vw < 0.2 or vh < 0.2 or np.any(vr < 0.001):
            return 1e6
        obj = -np.sum(vr)
        # Smooth pairwise overlap penalty
        dx = vx[:, None] - vx[None, :]
        dy = vy[:, None] - vy[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = vr[:, None] + vr[None, :]
        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        viol = r_sum[iu] - d_pairs
        beta = 100.0
        smooth_viol = np.maximum(viol, 0.0)
        obj += 2000.0 * np.sum(smooth_viol * smooth_viol)
        # Wall penalties (smooth)
        lw = np.maximum(vr - vx, 0.0)
        rw = np.maximum(vx + vr - vw, 0.0)
        bw = np.maximum(vr - vy, 0.0)
        tw = np.maximum(vy + vr - vh, 0.0)
        obj += 2000.0 * (np.sum(lw**2) + np.sum(rw**2) + np.sum(bw**2) + np.sum(tw**2))
        return obj

    bounds = []
    for i in range(n):
        bounds.extend([(0.01, 1.99), (0.01, 1.99), (0.005, 0.5)])
    bounds.append((0.4, 1.6))

    maxiter = int(min(600, time_budget * 80))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 80})
        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h, tol=1e-8):
            return z_opt
        # Try slight shrink to fix numerical issues
        for factor in [0.9999, 0.9995, 0.999]:
            r_shrunk = r * factor
            if _is_valid(x, y, r_shrunk, w, h, tol=1e-8):
                z_new = z_opt.copy()
                z_new[2::3][:n] = r_shrunk
                return z_new
        return None
    except Exception:
        return None


def _polish_result_tight(z, time_budget):
    """High-precision final polish: coordinate ascent with push-neighbors and binary search."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    def _max_r_i(xi, yi, ri, x_arr, y_arr, r_arr, W, H):
        """Max radius for circle at (xi,yi) given others."""
        r_wall = min(xi, W - xi, yi, H - yi)
        r_max = r_wall
        for j in range(n):
            dx = xi - x_arr[j]
            dy = yi - y_arr[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - r_arr[j]
            if r_allowed < r_max:
                r_max = r_allowed
        return max(0.0, r_max)

    def _valid_move_fast(xi, yi, ri, x_arr, y_arr, r_arr, W, H, skip_i):
        """Fast validity check for moving circle i."""
        if xi < ri or xi > W - ri or yi < ri or yi > H - ri:
            return False
        for j in range(n):
            if j == skip_i:
                continue
            dx = xi - x_arr[j]
            dy = yi - y_arr[j]
            rs = ri + r_arr[j]
            if dx * dx + dy * dy < rs * rs - 1e-9:
                return False
        return True

    # Phase 1: Coordinate ascent - maximize each radius, then reposition
    for outer_iter in range(40):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            # Max radius at current position
            r_max = _max_r_i(x[i], y[i], r[i], x, y, r, w, h)
            if r_max > r[i] + 1e-10:
                lo, hi = r[i], r_max
                for _ in range(18):
                    mid = (lo + hi) / 2.0
                    if _valid_move_fast(x[i], y[i], mid, x, y, r, w, h, i):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-11:
                    r[i] = lo
                    improved = True

            # Try to reposition circle i to allow larger radius
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max_cur = min(r_wall, r_max)
            # Find most constraining neighbor
            min_gap = float('inf')
            best_dx, best_dy = 0.0, 0.0
            for j in range(n):
                if j == i:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_gap:
                    min_gap = gap
                    if d > 1e-6:
                        best_dx, best_dy = dx / d, dy / d
            # Wall gaps
            for dx_w, dy_w in [(1,0),(-1,0),(0,1),(0,-1)]:
                if dx_w > 0:
                    g = x[i] - r[i]
                elif dx_w < 0:
                    g = w - x[i] - r[i]
                elif dy_w > 0:
                    g = y[i] - r[i]
                else:
                    g = h - y[i] - r[i]
                if g < min_gap:
                    min_gap = g
                    best_dx, best_dy = dx_w, dy_w

            if min_gap < 0.03:
                # Try moving away from constraining direction
                for step in [0.03, 0.015, 0.008, 0.004, 0.002]:
                    nx = x[i] + best_dx * step
                    ny = y[i] + best_dy * step
                    if _valid_move_fast(nx, ny, r[i], x, y, r, w, h, i):
                        new_r_max = _max_r_i(nx, ny, r[i], x, y, r, w, h)
                        if new_r_max > r_max_cur + 1e-7:
                            x[i], y[i] = nx, ny
                            # Expand to new max
                            lo, hi = r[i], new_r_max
                            for _ in range(15):
                                mid = (lo + hi) / 2.0
                                if _valid_move_fast(x[i], y[i], mid, x, y, r, w, h, i):
                                    lo = mid
                                else:
                                    hi = mid
                            if lo > r[i] + 1e-11:
                                r[i] = lo
                                improved = True
                            break

        if not improved:
            break

    # Phase 2: Multi-direction local search for stuck circles
    directions = [
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (0.7071, 0.7071), (-0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, -0.7071)
    ]
    steps = [0.03, 0.015, 0.008, 0.004, 0.002, 0.001]

    for ls_round in range(30):
        if time.time() > time_budget:
            break
        round_improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            r_max_cur = _max_r_i(x[i], y[i], r[i], x, y, r, w, h)
            best_gain = 0.0
            best_pos = None
            for dx_dir, dy_dir in directions:
                for step in steps:
                    nx = x[i] + dx_dir * step
                    ny = y[i] + dy_dir * step
                    if _valid_move_fast(nx, ny, r[i], x, y, r, w, h, i):
                        new_r_max = _max_r_i(nx, ny, r[i], x, y, r, w, h)
                        gain = new_r_max - r_max_cur
                        if gain > best_gain:
                            best_gain = gain
                            best_pos = (nx, ny)
            if best_pos is not None and best_gain > 1e-7:
                x[i], y[i] = best_pos
                lo, hi = r[i], r_max_cur + best_gain
                for _ in range(15):
                    mid = (lo + hi) / 2.0
                    if _valid_move_fast(x[i], y[i], mid, x, y, r, w, h, i):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-11:
                    r[i] = lo
                    round_improved = True
        if not round_improved:
            break

    # Phase 3: Final uniform expansion
    for _ in range(150):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0001
        if _is_valid(x, y, r_trial, w, h, tol=1e-9):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h, tol=1e-9):
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
    """Multi-start local search: perturb, coordinate-ascent maximize, and reposition."""
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

    def _max_r(xi, yi, x_arr, y_arr, r_arr, W, H):
        r_wall = min(xi, W - xi, yi, H - yi)
        r_max = r_wall
        for j in range(n):
            dx = xi - x_arr[j]
            dy = yi - y_arr[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - r_arr[j]
            if r_allowed < r_max:
                r_max = r_allowed
        return max(0.0, r_max)

    def _valid_fast(xi, yi, ri, x_arr, y_arr, r_arr, W, H, skip):
        if xi < ri or xi > W - ri or yi < ri or yi > H - ri:
            return False
        for j in range(n):
            if j == skip:
                continue
            dx = xi - x_arr[j]
            dy = yi - y_arr[j]
            rs = ri + r_arr[j]
            if dx * dx + dy * dy < rs * rs - 1e-9:
                return False
        return True

    for trial in range(25):
        if time.time() > time_budget:
            break

        x_p = x.copy()
        y_p = y.copy()
        r_p = r.copy()

        # Perturb a random subset
        n_perturb = rng.integers(2, 8)
        indices = rng.choice(n, size=n_perturb, replace=False)
        for idx in indices:
            dx_dir, dy_dir = directions[rng.integers(0, 8)]
            step = rng.uniform(0.005, 0.035)
            new_x = x_p[idx] + dx_dir * step
            new_y = y_p[idx] + dy_dir * step
            if _valid_fast(new_x, new_y, r_p[idx], x_p, y_p, r_p, w, h, idx):
                x_p[idx], y_p[idx] = new_x, new_y

        # Coordinate ascent: maximize each radius then reposition
        for outer_iter in range(15):
            if time.time() > time_budget:
                break
            improved = False
            for i in range(n):
                if time.time() > time_budget:
                    break
                r_max = _max_r(x_p[i], y_p[i], x_p, y_p, r_p, w, h)
                if r_max > r_p[i] + 1e-10:
                    lo, hi = r_p[i], r_max
                    for _ in range(14):
                        mid = (lo + hi) / 2.0
                        if _valid_fast(x_p[i], y_p[i], mid, x_p, y_p, r_p, w, h, i):
                            lo = mid
                        else:
                            hi = mid
                    if lo > r_p[i] + 1e-11:
                        r_p[i] = lo
                        improved = True

                # Try reposition
                r_max_cur = _max_r(x_p[i], y_p[i], x_p, y_p, r_p, w, h)
                min_gap = float('inf')
                best_dx, best_dy = 0.0, 0.0
                for j in range(n):
                    if j == i:
                        continue
                    dx = x_p[i] - x_p[j]
                    dy = y_p[i] - y_p[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    gap = d - (r_p[i] + r_p[j])
                    if gap < min_gap:
                        min_gap = gap
                        if d > 1e-6:
                            best_dx, best_dy = dx / d, dy / d
                for dx_w, dy_w in [(1,0),(-1,0),(0,1),(0,-1)]:
                    if dx_w > 0: g = x_p[i] - r_p[i]
                    elif dx_w < 0: g = w - x_p[i] - r_p[i]
                    elif dy_w > 0: g = y_p[i] - r_p[i]
                    else: g = h - y_p[i] - r_p[i]
                    if g < min_gap:
                        min_gap = g
                        best_dx, best_dy = float(dx_w), float(dy_w)

                if min_gap < 0.03:
                    for step in [0.02, 0.01, 0.005, 0.002]:
                        nx = x_p[i] + best_dx * step
                        ny = y_p[i] + best_dy * step
                        if _valid_fast(nx, ny, r_p[i], x_p, y_p, r_p, w, h, i):
                            new_r_max = _max_r(nx, ny, x_p, y_p, r_p, w, h)
                            if new_r_max > r_max_cur + 1e-7:
                                x_p[i], y_p[i] = nx, ny
                                lo, hi = r_p[i], new_r_max
                                for _ in range(12):
                                    mid = (lo + hi) / 2.0
                                    if _valid_fast(x_p[i], y_p[i], mid, x_p, y_p, r_p, w, h, i):
                                        lo = mid
                                    else:
                                        hi = mid
                                if lo > r_p[i] + 1e-11:
                                    r_p[i] = lo
                                    improved = True
                                break

            if not improved:
                break

        val_p = float(np.sum(r_p))
        if _is_valid(x_p, y_p, r_p, w, h, tol=1e-9) and val_p > best_val:
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
