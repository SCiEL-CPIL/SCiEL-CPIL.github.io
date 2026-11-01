"""Circle packing in perimeter-4 rectangle: SLSQP + Enhanced Polish + Contact Graph + Global Search."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

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

    # --- Phase 2: Enhanced Polish on best result ---
    if best_result is not None and time.time() < deadline - 25:
        # 1. Force-directed position polish to ensure we are in a good basin
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 15)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Expansion/Relocation Polish with enhanced local search
        if best_result is not None and time.time() < deadline - 20:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 3. Contact-graph local optimization (direct radius maximization)
        if best_result is not None and time.time() < deadline - 15:
            cg_result = _contact_graph_optimize(best_result, deadline - (time.time() - start_time) - 8)
            if cg_result is not None:
                valid, val = check_validity(cg_result)
                if valid and val > best_value:
                    best_value = val
                    best_result = cg_result

        # 4. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 12:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 6)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 4b. LSE epigraph refinement (smooth surrogate, better-conditioned)
        if best_result is not None and time.time() < deadline - 10:
            lse_result = _lse_epigraph_refine(best_result, deadline - (time.time() - start_time) - 5)
            if lse_result is not None:
                valid, val = check_validity(lse_result)
                if valid and val > best_value:
                    best_value = val
                    best_result = lse_result

        # 5. Final high-precision coordinate descent polish
        if best_result is not None and time.time() < deadline - 5:
            final_polish = _coordinate_descent_polish(best_result, deadline - (time.time() - start_time) - 2)
            if final_polish is not None:
                valid, val = check_validity(final_polish)
                if valid and val > best_value:
                    best_value = val
                    best_result = final_polish

        # 5b. Greedy per-circle expansion pass
        if best_result is not None and time.time() < deadline - 2:
            greedy = _greedy_expansion_pass(best_result, deadline - (time.time() - start_time))
            if greedy is not None:
                valid, val = check_validity(greedy)
                if valid and val > best_value:
                    best_value = val
                    best_result = greedy

    # --- Phase 3: Global search with Delaunay-based perturbation ---
    if best_result is not None and time.time() < deadline - 10:
        # Try a few global perturbations to escape local minima
        rng_global = np.random.default_rng(123)
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            z_perturbed = _global_perturb(best_result, rng_global, deadline - (time.time() - start_time) - 5)
            if z_perturbed is not None:
                valid, val = check_validity(z_perturbed)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_perturbed

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
    Advanced polish: Iteratively expand circles and shift positions to create space.
    Uses systematic local search with multiple strategies.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Strategy 1: Greedy expansion with binary search
    max_rounds = 20
    for rnd in range(max_rounds):
        if time.time() > time_budget - 5:
            break
        
        improved = False
        # Process circles in order of smallest current radius first (they have most room to grow)
        order = np.argsort(r)
        for i in order:
            if time.time() > time_budget - 5:
                break
            # Calculate max possible radius for circle i
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
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid
                new_r = lo
                if new_r > r[i] + 1e-7:
                    r[i] = new_r
                    improved = True
        
        if not improved:
            break

    # Strategy 2: Position optimization via local search
    # Try to move each circle to a better position that allows larger radius
    for rnd in range(15):
        if time.time() > time_budget - 3:
            break
        
        improved = False
        for i in range(n):
            if time.time() > time_budget - 3:
                break
            
            # Find the direction of the tightest constraint
            min_gap = float('inf')
            best_move = None
            best_new_r = r[i]
            
            # Check walls
            d_left = x[i] - r[i]
            d_right = w - x[i] - r[i]
            d_bottom = y[i] - r[i]
            d_top = h - y[i] - r[i]
            
            # Try moving in 8 directions
            for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                for step in [0.08, 0.05, 0.03, 0.02, 0.01, 0.005]:
                    new_x = x[i] + step * np.cos(angle)
                    new_y = y[i] + step * np.sin(angle)
                    
                    # Check if valid
                    if new_x < r[i] or new_x > w - r[i]:
                        continue
                    if new_y < r[i] or new_y > h - r[i]:
                        continue
                    
                    valid = True
                    for j in range(n):
                        if i == j:
                            continue
                        dx = new_x - x[j]
                        dy = new_y - y[j]
                        d2 = dx * dx + dy * dy
                        r_sum = r[i] + r[j]
                        if d2 < r_sum * r_sum - 1e-9:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Calculate new max radius from this position
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
                    
                    if r_max_new > best_new_r + 1e-5:
                        best_new_r = r_max_new
                        best_move = (new_x, new_y)
            
            # Apply best move if found
            if best_move is not None and best_new_r > r[i] + 1e-5:
                x[i] = best_move[0]
                y[i] = best_move[1]
                r[i] = best_new_r
                improved = True
        
        if not improved:
            break

    # Strategy 3: Uniform expansion with binary search
    lo = 1.0
    hi = 1.05
    for _ in range(30):
        if time.time() > time_budget:
            break
        mid = (lo + hi) / 2.0
        r_trial = r * mid
        if _is_valid(x, y, r_trial, w, h):
            lo = mid
        else:
            hi = mid
    r = r * lo

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


def _lse_epigraph_refine(z, time_budget):
    """
    Smooth log-sum-exp epigraph refinement.
    
    Instead of non-convex distance constraints (d_ij >= r_i + r_j), we use
    a smooth surrogate: t >= soft_max(r_i + r_j - d_ij) for all pairs,
    with t constrained to be <= 0. This is better-conditioned for SLSQP
    and can escape local minima that trap the plain constraint refine.
    
    Variables: x_i, y_i, r_i for i in [0..N-1], plus t, plus w.
    Objective: maximize sum(r_i) - penalty * max(0, t)
    Constraints: t >= log-sum-exp((r_i + r_j - d_ij)^2 / sigma^2) for all pairs
                 t <= 0 (enforced via penalty)
                 Boundary constraints (smooth)
    """
    z = z.copy().astype(float)
    n = N
    x0 = z[0::3][:n].copy()
    y0 = z[1::3][:n].copy()
    r0 = z[2::3][:n].copy()
    w0 = z[-1]
    h0 = 2.0 - w0

    # Variables: [x0..x_{n-1}, y0..y_{n-1}, r0..r_{n-1}, t, w]
    # Total: 3*n + 2
    def lse_obj(params):
        x = params[0:n]
        y = params[n:2*n]
        r = params[2*n:3*n]
        t = params[3*n]
        w = params[3*n + 1]
        h = 2.0 - w

        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6

        obj = -np.sum(r)

        # Pairwise: soft max of (r_i + r_j - d_ij)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        d = np.sqrt(np.maximum(d_sq, 1e-12))
        r_sum = r[:, None] + r[None, :]
        overlap = r_sum - d  # positive means overlap

        # Use log-sum-exp for smooth max
        sigma = 0.01
        iu = np.triu_indices(n, k=1)
        ov_pairs = overlap[iu]
        # Soft max: sigma * log(sum(exp(ov/sigma)))
        max_ov = np.max(ov_pairs)
        if max_ov > -5 * sigma:
            lse_val = sigma * np.log(np.sum(np.exp((ov_pairs - max_ov) / sigma))) + max_ov
        else:
            lse_val = max_ov

        # Penalty for t > 0 and for t < lse_val
        # We want t >= lse_val and t <= 0
        # Penalty: (max(0, t))^2 + (max(0, lse_val - t))^2
        pen_t = max(0.0, t) ** 2 + max(0.0, lse_val - t) ** 2
        obj += 500.0 * pen_t

        # Boundary penalties (smooth)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        boundary_pen = 500.0 * (
            np.sum(left_viol ** 2) + np.sum(right_viol ** 2) +
            np.sum(bottom_viol ** 2) + np.sum(top_viol ** 2)
        )
        obj += boundary_pen

        return obj

    bounds_lse = [(0, 2)] * n + [(0, 2)] * n + [(0.001, 1.0)] * n + [(-0.1, 0.1)] + [(0.3, 1.7)]

    # Initial t: estimate from current overlaps
    x = x0.copy()
    y = y0.copy()
    r = r0.copy()
    w = w0
    h = h0
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    d = np.sqrt(np.maximum(d_sq, 1e-12))
    r_sum = r[:, None] + r[None, :]
    overlap = r_sum - d
    iu = np.triu_indices(n, k=1)
    t_init = max(0.0, float(np.max(overlap[iu]))) + 0.001

    z_lse = np.concatenate([x, y, r, [t_init], [w]])

    maxiter = int(min(1000, time_budget * 200))
    try:
        res = minimize(lse_obj, z_lse, method='SLSQP',
                       bounds=bounds_lse,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})

        z_opt = res.x
        x = z_opt[0:n]
        y = z_opt[n:2*n]
        r = z_opt[2*n:3*n]
        w = z_opt[3*n + 1]
        h = 2.0 - w

        if _is_valid(x, y, r, w, h):
            z_new = np.empty(3 * N + 1)
            z_new[0::3][:n] = x
            z_new[1::3][:n] = y
            z_new[2::3][:n] = r
            z_new[-1] = w
            return z_new
        return None
    except Exception:
        return None


def _greedy_expansion_pass(z, time_budget):
    """
    Final per-circle greedy binary-search expansion.
    For each circle (in order of smallest radius first), binary-search
    the maximum valid radius given current positions, then try to
    reposition to allow further growth.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Multiple rounds of per-circle expansion
    for rnd in range(10):
        if time.time() > time_budget:
            break

        improved = False
        order = np.argsort(r)  # smallest first

        for i in order:
            if time.time() > time_budget:
                break

            # Compute max radius at current position
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

            if r_max > r[i] + 1e-7:
                # Binary search for true max valid radius
                lo = r[i]
                hi = r_max
                for _ in range(20):
                    if time.time() > time_budget:
                        break
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True

            # Try to reposition circle i to allow larger radius
            # Search in a small neighborhood
            best_r_i = r[i]
            best_pos = (x[i], y[i])
            for dx in [-0.01, -0.005, 0.005, 0.01]:
                for dy in [-0.01, -0.005, 0.005, 0.01]:
                    if time.time() > time_budget:
                        break
                    nx = x[i] + dx
                    ny = y[i] + dy
                    if nx < r[i] or nx > w - r[i]:
                        continue
                    if ny < r[i] or ny > h - r[i]:
                        continue

                    # Check validity at current radius
                    valid = True
                    for j in range(n):
                        if i == j:
                            continue
                        ddx = nx - x[j]
                        ddy = ny - y[j]
                        if ddx * ddx + ddy * ddy < (r[i] + r[j]) ** 2 - 1e-9:
                            valid = False
                            break
                    if not valid:
                        continue

                    # Max radius at new position
                    r_wall_new = min(nx, w - nx, ny, h - ny)
                    r_max_new = r_wall_new
                    for j in range(n):
                        if i == j:
                            continue
                        ddx = nx - x[j]
                        ddy = ny - y[j]
                        d = np.sqrt(ddx * ddx + ddy * ddy)
                        r_allowed = d - r[j]
                        if r_allowed < r_max_new:
                            r_max_new = r_allowed
                    r_max_new = max(0.0, r_max_new)

                    if r_max_new > best_r_i + 1e-6:
                        best_r_i = r_max_new
                        best_pos = (nx, ny)

            if best_r_i > r[i] + 1e-6:
                x[i] = best_pos[0]
                y[i] = best_pos[1]
                r[i] = best_r_i
                improved = True

        if not improved:
            break

    # Final uniform expansion
    lo, hi = 1.0, 1.03
    for _ in range(25):
        if time.time() > time_budget:
            break
        mid = (lo + hi) / 2.0
        r_trial = r * mid
        if _is_valid(x, y, r_trial, w, h):
            lo = mid
        else:
            hi = mid
    r = r * lo

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


def _contact_graph_optimize(z, time_budget):
    """
    Contact-graph-based local optimizer: builds the contact graph (pairs touching
    or nearly touching), then performs a local Newton-style solve to maximize
    sum of radii subject to separation and boundary constraints.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Build contact graph: pairs where distance < r_i + r_j + eps
    eps = 0.005
    contacts = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < r[i] + r[j] + eps:
                contacts.append((i, j))

    # Also include wall contacts
    wall_contacts = []  # (circle_idx, wall: 0=left, 1=right, 2=bottom, 3=top)
    for i in range(n):
        if x[i] - r[i] < eps:
            wall_contacts.append((i, 0))
        if w - x[i] - r[i] < eps:
            wall_contacts.append((i, 1))
        if y[i] - r[i] < eps:
            wall_contacts.append((i, 2))
        if h - y[i] - r[i] < eps:
            wall_contacts.append((i, 3))

    # Local optimization: try small perturbations of positions and radii
    # that maintain all constraints while increasing sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_val = float(np.sum(r))
    step_size = 0.002

    for iteration in range(200):
        if time.time() > time_budget:
            break

        improved = False

        # Try growing each circle slightly and adjusting positions
        for i in range(n):
            if time.time() > time_budget:
                break

            # Try to grow circle i
            grow = step_size * 0.5
            r_trial = best_r.copy()
            r_trial[i] += grow

            if not _is_valid(best_x, best_y, r_trial, w, h):
                # Try adjusting position of circle i to make room
                for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                    for step in [0.003, 0.005, 0.008, 0.012]:
                        new_x = best_x[i] + step * np.cos(angle)
                        new_y = best_y[i] + step * np.sin(angle)

                        if new_x < r_trial[i] or new_x > w - r_trial[i]:
                            continue
                        if new_y < r_trial[i] or new_y > h - r_trial[i]:
                            continue

                        x_trial = best_x.copy()
                        y_trial = best_y.copy()
                        x_trial[i] = new_x
                        y_trial[i] = new_y

                        if _is_valid(x_trial, y_trial, r_trial, w, h):
                            best_x = x_trial
                            best_y = y_trial
                            best_r = r_trial
                            improved = True
                            break
                    if improved:
                        break

        if not improved:
            # Try pairwise position adjustments for contact pairs
            for (i, j) in contacts:
                if time.time() > time_budget:
                    break

                dx = best_x[i] - best_x[j]
                dy = best_y[i] - best_y[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < 1e-10:
                    continue

                # Try moving both circles apart slightly
                for factor in [0.002, 0.004, 0.006]:
                    ux = dx / d
                    uy = dy / d
                    x_trial = best_x.copy()
                    y_trial = best_y.copy()
                    x_trial[i] += factor * ux
                    y_trial[i] += factor * uy
                    x_trial[j] -= factor * ux
                    y_trial[j] -= factor * uy

                    # Clamp to walls
                    x_trial[i] = np.clip(x_trial[i], best_r[i], w - best_r[i])
                    y_trial[i] = np.clip(y_trial[i], best_r[i], h - best_r[i])
                    x_trial[j] = np.clip(x_trial[j], best_r[j], w - best_r[j])
                    y_trial[j] = np.clip(y_trial[j], best_r[j], h - best_r[j])

                    if _is_valid(x_trial, y_trial, best_r, w, h):
                        # Now try to grow both circles
                        r_trial = best_r.copy()
                        grow = factor * 0.3
                        r_trial[i] += grow
                        r_trial[j] += grow

                        if _is_valid(x_trial, y_trial, r_trial, w, h):
                            best_x = x_trial
                            best_y = y_trial
                            best_r = r_trial
                            improved = True
                            break
                        else:
                            best_x = x_trial
                            best_y = y_trial
                            improved = True
                            break

            if not improved:
                # Try uniform small growth
                r_trial = best_r * (1 + 0.0001)
                if _is_valid(best_x, best_y, r_trial, w, h):
                    best_r = r_trial
                    improved = True

        if not improved:
            # Decrease step and try finer search
            step_size *= 0.7
            if step_size < 0.0001:
                break

    # Final uniform expansion
    lo, hi = 1.0, 1.02
    for _ in range(20):
        if time.time() > time_budget:
            break
        mid = (lo + hi) / 2.0
        r_trial = best_r * mid
        if _is_valid(best_x, best_y, r_trial, w, h):
            lo = mid
        else:
            hi = mid
    best_r = best_r * lo

    if _is_valid(best_x, best_y, best_r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = best_x
        z_new[1::3][:n] = best_y
        z_new[2::3][:n] = best_r
        z_new[-1] = w
        return z_new
    return None


def _coordinate_descent_polish(z, time_budget):
    """
    High-precision coordinate descent: alternate between optimizing each
    circle's position (x, y) and radius (r) with tight tolerances.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    max_rounds = 50
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False

        # Phase A: Optimize positions (fix radii)
        for i in range(n):
            if time.time() > time_budget:
                break

            # Golden-section search in x and y for max feasible radius
            best_r_i = r[i]
            best_pos = (x[i], y[i])

            # Try a grid of nearby positions (finer grid)
            for dx in [-0.03, -0.02, -0.015, -0.01, -0.005, 0, 0.005, 0.01, 0.015, 0.02, 0.03]:
                for dy in [-0.03, -0.02, -0.015, -0.01, -0.005, 0, 0.005, 0.01, 0.015, 0.02, 0.03]:
                    nx = x[i] + dx
                    ny = y[i] + dy

                    if nx < r[i] or nx > w - r[i]:
                        continue
                    if ny < r[i] or ny > h - r[i]:
                        continue

                    # Check validity at current radius
                    valid = True
                    for j in range(n):
                        if i == j:
                            continue
                        ddx = nx - x[j]
                        ddy = ny - y[j]
                        if ddx * ddx + ddy * ddy < (r[i] + r[j]) ** 2 - 1e-9:
                            valid = False
                            break

                    if not valid:
                        continue

                    # Max radius at this position
                    r_max = min(nx, w - nx, ny, h - ny)
                    for j in range(n):
                        if i == j:
                            continue
                        ddx = nx - x[j]
                        ddy = ny - y[j]
                        d = np.sqrt(ddx * ddx + ddy * ddy)
                        r_allowed = d - r[j]
                        if r_allowed < r_max:
                            r_max = r_allowed
                    r_max = max(0.0, r_max)

                    if r_max > best_r_i + 1e-8:
                        best_r_i = r_max
                        best_pos = (nx, ny)

            if best_pos != (x[i], y[i]) or best_r_i > r[i] + 1e-8:
                x[i] = best_pos[0]
                y[i] = best_pos[1]
                r[i] = max(r[i], best_r_i)
                improved = True

        # Phase B: Optimize radii (fix positions) - binary search per circle
        for i in range(n):
            if time.time() > time_budget:
                break

            # Max possible radius
            r_max = min(x[i], w - x[i], y[i], h - y[i])
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

            if r_max > r[i] + 1e-8:
                # Binary search
                lo, hi = r[i], r_max
                for _ in range(15):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid
                r[i] = lo
                improved = True

        if not improved:
            break

    # Final uniform expansion
    lo, hi = 1.0, 1.01
    for _ in range(15):
        if time.time() > time_budget:
            break
        mid = (lo + hi) / 2.0
        r_trial = r * mid
        if _is_valid(x, y, r_trial, w, h):
            lo = mid
        else:
            hi = mid
    r = r * lo

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _global_perturb(z, rng, time_budget):
    """
    Global perturbation using Delaunay-based neighborhood exploration.
    This attempts to escape local minima by making larger structural changes
    to the packing configuration.
    """
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Build Delaunay triangulation to identify neighbor relationships
    points = np.column_stack((x, y))
    try:
        tri = Delaunay(points)
    except Exception:
        return None

    # For each circle, try moving it to a random location in the rectangle
    # and then re-optimize locally
    best_perturbed = None
    best_val = 0.0

    for _ in range(5):
        if time.time() > time_budget:
            break

        x_new = x.copy()
        y_new = y.copy()
        r_new = r.copy()

        # Randomly select 3-5 circles to relocate
        n_relocate = rng.integers(3, 6)
        indices = rng.choice(n, size=n_relocate, replace=False)

        for idx in indices:
            # New random position
            r_i = r_new[idx]
            x_new[idx] = rng.uniform(r_i, w - r_i)
            y_new[idx] = rng.uniform(r_i, h - r_i)

        # Check validity
        if _is_valid(x_new, y_new, r_new, w, h):
            # Try to expand radii
            for _ in range(50):
                if time.time() > time_budget:
                    break
                r_trial = r_new * 1.001
                if _is_valid(x_new, y_new, r_trial, w, h):
                    r_new = r_trial
                else:
                    break

            val = float(np.sum(r_new))
            if val > best_val:
                best_val = val
                best_perturbed = np.empty(3 * N + 1)
                best_perturbed[0::3][:n] = x_new
                best_perturbed[1::3][:n] = y_new
                best_perturbed[2::3][:n] = r_new
                best_perturbed[-1] = w

    if best_perturbed is not None:
        # Polish the perturbed result
        polished = _polish_result(best_perturbed, time_budget)
        if polished is not None:
            return polished

    return None
