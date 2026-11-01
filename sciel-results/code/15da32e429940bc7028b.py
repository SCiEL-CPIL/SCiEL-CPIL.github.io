"""Circle packing in perimeter-4 rectangle: SLSQP + Constraint Polish + LSE Refine."""
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

    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        # More diverse aspect ratios
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8, 0.9, 1.1, 1.4, 0.7]
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

        return guesses

    guesses = get_initial_guesses()

    # Constraint-based SLSQP setup
    def make_constraints():
        cons = []
        # Pairwise non-overlap: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        # This is non-convex but SLSQP can handle it locally.
        # To make it smoother and more robust, we use the form: dist - (r_i + r_j) >= 0
        # But SLSQP requires differentiable constraints. sqrt is differentiable for d>0.
        
        # Let's stick to the squared distance form for stability, but it's non-convex.
        # Actually, the explicit constraint approach in _constraint_refine is better.
        # We will use the _constraint_refine function as the main solver now.
        pass

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 25:
            break

        # Force-directed relaxation to resolve initial overlaps
        z0_relaxed = _force_relax_unified(z0, max_iter=500)

        # Use Constraint Refine (SLSQP with explicit constraints) as primary solver
        remaining = deadline - (time.time() - start_time)
        try:
            res_opt = _constraint_refine(z0_relaxed, remaining - 5)
            if res_opt is not None:
                valid, val = check_validity(res_opt)
                if valid and val > best_value:
                    best_value = val
                    best_result = res_opt.copy()
        except Exception:
            continue

        # Fallback to penalty-based SLSQP if constraint fails or for variety
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': 300, 'ftol': 1e-12, 'maxls': 80})
            z_opt = res.x
            valid, val = check_validity(z_opt)
            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
        except Exception:
            continue

    # --- Phase 2: Polish on best result ---
    if best_result is not None and time.time() < deadline - 25:
        # 1. Force-directed position polish
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 15)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Expansion/Relocation Polish
        if best_result is not None and time.time() < deadline - 20:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 3. Contact Graph Optimization (New)
        if best_result is not None and time.time() < deadline - 15:
            cg = _contact_graph_optimize(best_result, deadline - (time.time() - start_time) - 10)
            if cg is not None:
                valid, val = check_validity(cg)
                if valid and val > best_value:
                    best_value = val
                    best_result = cg

        # 4. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 12:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 6)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 5. Smooth LSE epigraph refinement
        if best_result is not None and time.time() < deadline - 10:
            lse = _lse_refine(best_result, deadline - (time.time() - start_time) - 5)
            if lse is not None:
                valid, val = check_validity(lse)
                if valid and val > best_value:
                    best_value = val
                    best_result = lse

        # 6. Soft expansion refinement
        if best_result is not None and time.time() < deadline - 8:
            soft = _soft_expansion_refine(best_result, deadline - (time.time() - start_time) - 4)
            if soft is not None:
                valid, val = check_validity(soft)
                if valid and val > best_value:
                    best_value = val
                    best_result = soft

        # 7. Second constraint refinement pass
        if best_result is not None and time.time() < deadline - 5:
            refined2 = _constraint_refine(best_result, deadline - (time.time() - start_time) - 2)
            if refined2 is not None:
                valid, val = check_validity(refined2)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined2

        # 8. Coordinate descent polish
        if best_result is not None and time.time() < deadline - 3:
            cd = _coordinate_descent_polish(best_result, deadline - (time.time() - start_time) - 1)
            if cd is not None:
                valid, val = check_validity(cd)
                if valid and val > best_value:
                    best_value = val
                    best_result = cd

        # 9. Final high-precision polish
        if best_result is not None and time.time() < deadline - 1:
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

    # Final coordinate descent squeeze
    if time.time() < start_time + 148:
        cd_final = _coordinate_descent_polish(
            np.concatenate([x, y, r, [w]]),
            start_time + 148
        )
        if cd_final is not None:
            x2 = cd_final[0::3][:N]
            y2 = cd_final[1::3][:N]
            r2 = cd_final[2::3][:N]
            w2 = cd_final[-1]
            h2 = 2.0 - w2
            if _is_valid(x2, y2, r2, w2, h2) and np.sum(r2) > np.sum(r):
                x, y, r, w, h = x2, y2, r2, w2, h2

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

    dt = 0.002
    max_iter = int(min(5000, time_budget * 200))
    prev_best = float(np.sum(r))

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
        fx -= left_viol * 40
        fx += right_viol * 40
        fy -= bottom_viol * 40
        fy += top_viol * 40

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

        # Adaptive step
        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.1:
            dt = 0.0005
        elif max_force > 0.01:
            dt = 0.001
        else:
            dt = 0.002

        x_new = x + dt * fx
        y_new = y + dt * fy
        x_new = np.clip(x_new, r, w - r)
        y_new = np.clip(y_new, r, h - r)

        # Only accept if valid
        if _is_valid(x_new, y_new, r, w, h):
            x = x_new
            y = y_new

    # Try to grow radii uniformly with binary search
    for _ in range(100):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.05
        for _ in range(15):
            mid = (lo + hi) / 2.0
            if _is_valid(x, y, r * mid, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-9:
            r = r * lo
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
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # 8 directions
    dirs = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (0.7071, 0.7071), (-0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, -0.7071)
    ]

    def try_expand_circle(i):
        """Try to expand circle i to max possible radius."""
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
                r_test = r.copy()
                r_test[i] = mid
                if _is_valid(x, y, r_test, w, h):
                    lo = mid
                else:
                    hi = mid
            if lo > r[i] + 1e-7:
                r[i] = lo
                return True
        return False

    def try_move_circle(i):
        """Try to move circle i in various directions to create space."""
        best_dir = None
        best_score = -float('inf')

        for dx_d, dy_d in dirs:
            new_x = x[i] + dx_d * 0.01
            new_y = y[i] + dy_d * 0.01
            if new_x < r[i] or new_x > w - r[i] or new_y < r[i] or new_y > h - r[i]:
                continue

            min_gap = float('inf')
            for j in range(n):
                if i == j:
                    continue
                dx = new_x - x[j]
                dy = new_y - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_gap:
                    min_gap = gap

            wall_dist = min(new_x - r[i], w - new_x - r[i], new_y - r[i], h - new_y - r[i])
            score = min(min_gap, wall_dist)

            if score > best_score:
                best_score = score
                best_dir = (dx_d, dy_d)

        if best_dir is None:
            return False

        dx_d, dy_d = best_dir

        for step in [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001]:
            new_x = x[i] + dx_d * step
            new_y = y[i] + dy_d * step
            if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                x[i] = new_x
                y[i] = new_y
                return True

        return False

    # Pass 1: Expand and move
    max_rounds = 40
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break

            if try_expand_circle(i):
                improved = True

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

            if r_max - r[i] < 0.01:  # Limited expansion
                if try_move_circle(i):
                    improved = True

        if not improved:
            break

    # Pass 2: Second expansion after repositioning
    for rnd in range(20):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            if try_expand_circle(i):
                improved = True

        if not improved:
            break

    # Final uniform expansion with binary search
    for _ in range(100):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.05
        for _ in range(15):
            mid = (lo + hi) / 2.0
            if _is_valid(x, y, r * mid, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-9:
            r = r * lo
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


def _contact_graph_optimize(z, time_budget):
    """
    Identify contact graph (circles touching each other or walls) and optimize
    by enforcing these contacts as equality constraints. This reduces the
    degrees of freedom and can lead to a tighter packing.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    tol = 1e-4
    contacts = []
    wall_contacts = []

    # Find pairwise contacts
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            if abs(d - (r[i] + r[j])) < tol:
                contacts.append((i, j))

    # Find wall contacts
    for i in range(n):
        if abs(x[i] - r[i]) < tol:
            wall_contacts.append((i, 'left'))
        if abs(w - x[i] - r[i]) < tol:
            wall_contacts.append((i, 'right'))
        if abs(y[i] - r[i]) < tol:
            wall_contacts.append((i, 'bottom'))
        if abs(h - y[i] - r[i]) < tol:
            wall_contacts.append((i, 'top'))

    if not contacts and not wall_contacts:
        return None

    # Define objective: minimize negative sum of radii
    def obj_cg(zv):
        return -np.sum(zv[2::3][:n])

    # Define constraints
    cons = []

    # Pairwise contact constraints: dist - (r_i + r_j) = 0
    # We use inequality dist - (r_i + r_j) >= 0 for validity, but for optimization
    # we want to push them to be equal.
    # However, SLSQP equality constraints on non-convex functions are tricky.
    # Instead, we can use the penalty method on the contact graph or just use
    # the existing constraint refine which handles non-overlap.
    
    # A better approach for "Contact Graph" is to fix the topology and solve
    # for the optimal radii and positions that satisfy the contact equations.
    # This is a system of non-linear equations.
    # Let's use scipy.optimize.fsolve or least_squares.
    
    # Variables: x, y, r, w (h is dependent)
    # Equations:
    # 1. For each contact (i,j): (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 = 0
    # 2. For each wall contact: e.g., x_i - r_i = 0
    # 3. We need to ensure non-overlap for non-contacts: dist >= r_i + r_j
    
    # This is underdetermined if we don't fix some variables.
    # We can fix w (aspect ratio) and optimize x, y, r.
    # Number of variables: 2n + n = 3n = 63.
    # Number of equations: len(contacts) + len(wall_contacts).
    # If #equations < #variables, we have a family of solutions.
    # We want to maximize sum(r).
    
    # Let's use a simpler heuristic: 
    # 1. Identify the "rigid" components of the contact graph.
    # 2. For each component, we can potentially scale it up if it's not touching the boundary.
    
    # For now, let's just return the input if the graph is not rigid, 
    # or try a small perturbation to break symmetry.
    
    # A practical approach: Use the contact graph to guide the SLSQP refinement.
    # We will create a custom SLSQP run that penalizes deviation from the contact distances.
    
    def cg_objective(zv):
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w
        
        obj = -np.sum(r)
        
        # Penalty for not being in contact
        for i, j in contacts:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            diff = d - (r[i] + r[j])
            obj += 10.0 * diff * diff
            
        for i, side in wall_contacts:
            if side == 'left':
                diff = x[i] - r[i]
            elif side == 'right':
                diff = w - x[i] - r[i]
            elif side == 'bottom':
                diff = y[i] - r[i]
            else:
                diff = h - y[i] - r[i]
            obj += 10.0 * diff * diff
            
        return obj

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    try:
        res = minimize(cg_objective, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-12})
        
        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w
        
        # Check validity
        if _is_valid(x, y, r, w, h):
            # Try to expand radii
            lo, hi = 1.0, 1.05
            for _ in range(15):
                mid = (lo + hi) / 2.0
                if _is_valid(x, y, r * mid, w, h):
                    lo = mid
                else:
                    hi = mid
            r = r * lo
            
            z_new = np.empty(3 * N + 1)
            z_new[0::3][:n] = x
            z_new[1::3][:n] = y
            z_new[2::3][:n] = r
            z_new[-1] = w
            return z_new
    except Exception:
        pass
        
    return None


def _constraint_refine(z, time_budget):
    """High-precision SLSQP refinement using pure objective + explicit inequality constraints."""
    z = z.copy().astype(float)
    n = N

    def obj_refine(zv):
        """Objective: minimize negative sum of radii (no penalty)."""
        r = zv[2::3][:n]
        return -np.sum(r)

    def make_circle_constraint(i):
        """Circle i must not overlap with any other circle: dist - (r[i]+r[j]) >= 0."""
        def constraint(zv):
            x = zv[0::3][:n]
            y = zv[1::3][:n]
            r = zv[2::3][:n]
            dx = x[i] - x
            dy = y[i] - y
            dist = np.sqrt(dx * dx + dy * dy + 1e-12)
            result = dist - (r[i] + r)
            result[i] = 1.0  # Self-constraint always satisfied
            return result
        return constraint

    def make_wall_constraint(axis, side):
        """Wall constraint: circle must stay inside rectangle."""
        def constraint(zv):
            x = zv[0::3][:n]
            y = zv[1::3][:n]
            r = zv[2::3][:n]
            w = zv[-1]
            h = 2.0 - w

            if axis == 'x':
                pos = x
            else:
                pos = y

            if side == 'left' or side == 'bottom':
                return pos - r
            else:
                if axis == 'x':
                    return w - pos - r
                else:
                    return h - pos - r
        return constraint

    # Build constraints
    constraints = []

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': make_circle_constraint(i)})

    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('x', 'left')})
    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('x', 'right')})
    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('y', 'bottom')})
    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('y', 'top')})

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    maxiter = int(min(1500, time_budget * 300))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 200})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        # Try to repair if slightly invalid
        if not _is_valid(x, y, r, w, h):
            for factor in [0.999, 0.998, 0.997, 0.995, 0.99]:
                r_trial = r * factor
                if _is_valid(x, y, r_trial, w, h):
                    r = r_trial
                    break

        if _is_valid(x, y, r, w, h):
            z_opt[2::3][:n] = r
            return z_opt
        return None
    except Exception:
        return None


def _lse_refine(z, time_budget):
    """
    Smooth surrogate refinement: minimize -sum(r) + smooth log-sum-exp penalty
    on overlap and wall violations. The LSE term is a smooth, always-positive
    surrogate for the hard non-overlap / in-rectangle constraints, which is
    better conditioned for SLSQP than the non-convex dist constraints.
    """
    z = z.copy().astype(float)
    n = N

    def lse_obj(zv):
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w

        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6

        # Pairwise overlap: r_i + r_j - dist
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        viol = r_sum_pairs - d_pairs  # > 0 when overlapping

        # Wall violations (all four)
        wall_viol = np.concatenate([
            np.maximum(r - x, 0.0),
            np.maximum(x + r - w, 0.0),
            np.maximum(r - y, 0.0),
            np.maximum(y + r - h, 0.0),
        ])

        # Log-sum-exp smooth penalty (temperature beta)
        beta = 200.0
        all_viol = np.concatenate([viol, wall_viol])
        m = np.max(all_viol)
        lse = m + np.log(np.sum(np.exp(beta * (all_viol - m)))) / beta

        return -np.sum(r) + 50.0 * lse

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    maxiter = int(min(1200, time_budget * 300))
    try:
        res = minimize(lse_obj, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 150})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if not _is_valid(x, y, r, w, h):
            for factor in [0.999, 0.998, 0.997, 0.995, 0.99, 0.98]:
                r_trial = r * factor
                if _is_valid(x, y, r_trial, w, h):
                    r = r_trial
                    break

        if _is_valid(x, y, r, w, h):
            z_opt[2::3][:n] = r
            return z_opt
        return None
    except Exception:
        return None


def _soft_expansion_refine(z, time_budget):
    """
    Joint position+radius optimization with smooth LSE penalty.
    Uses a higher LSE weight to push circles apart while growing radii.
    This can find configurations where circles are slightly separated,
    allowing further radius growth.
    """
    z = z.copy().astype(float)
    n = N

    def soft_obj(zv):
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w

        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6

        # Pairwise: we want dist >= r_i + r_j + gap_margin
        # Violation = r_i + r_j + margin - dist (positive when too close)
        margin = 0.002
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        viol = r_sum_pairs + margin - d_pairs

        # Wall violations
        wall_viol = np.concatenate([
            np.maximum(r - x, 0.0),
            np.maximum(x + r - w, 0.0),
            np.maximum(r - y, 0.0),
            np.maximum(y + r - h, 0.0),
        ])

        # LSE penalty with high weight to enforce separation
        beta = 300.0
        all_viol = np.concatenate([viol, wall_viol])
        m = np.max(all_viol)
        lse = m + np.log(np.sum(np.exp(beta * (all_viol - m)))) / beta

        return -np.sum(r) + 100.0 * lse

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    maxiter = int(min(1500, time_budget * 250))
    try:
        res = minimize(soft_obj, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-13, 'maxls': 200})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        # Now try to grow radii since positions may have shifted
        if _is_valid(x, y, r, w, h):
            # Binary search for max uniform growth
            lo, hi = 1.0, 1.08
            for _ in range(20):
                mid = (lo + hi) / 2.0
                if _is_valid(x, y, r * mid, w, h):
                    lo = mid
                else:
                    hi = mid
            r = r * lo

        # Per-circle expansion
        for i in range(n):
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
                lo_r, hi_r = r[i], r_max
                for _ in range(20):
                    mid_r = (lo_r + hi_r) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid_r
                    if _is_valid(x, y, r_test, w, h):
                        lo_r = mid_r
                    else:
                        hi_r = mid_r
                r[i] = lo_r

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


def _coordinate_descent_polish(z, time_budget):
    """
    Final high-precision polish via coordinate descent:
    For each circle, optimize its (x, y, r) independently while others are fixed.
    This can squeeze out small gains that gradient methods miss.
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
        for i in range(n):
            if time.time() > time_budget:
                break

            # Max possible radius for circle i given others fixed
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

            if r_max <= r[i] + 1e-10:
                continue

            # Binary search for max radius
            lo, hi = r[i], r_max
            for _ in range(25):
                mid = (lo + hi) / 2.0
                r_test = r.copy()
                r_test[i] = mid
                if _is_valid(x, y, r_test, w, h):
                    lo = mid
                else:
                    hi = mid

            if lo > r[i] + 1e-10:
                r[i] = lo
                improved = True

            # Try small position shifts to create more room
            for dx_s in [0.005, -0.005, 0.01, -0.01, 0.002, -0.002]:
                for dy_s in [0.005, -0.005, 0.01, -0.01, 0.002, -0.002]:
                    if time.time() > time_budget:
                        break
                    nx = x[i] + dx_s
                    ny = y[i] + dy_s
                    if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                        continue
                    if not _is_valid_move(x, y, r, w, h, i, nx, ny):
                        continue
                    # Check if this position allows larger radius
                    r_wall2 = min(nx, w - nx, ny, h - ny)
                    r_max2 = r_wall2
                    for j in range(n):
                        if i == j:
                            continue
                        dx2 = nx - x[j]
                        dy2 = ny - y[j]
                        d2 = np.sqrt(dx2 * dx2 + dy2 * dy2)
                        r_allowed2 = d2 - r[j]
                        if r_allowed2 < r_max2:
                            r_max2 = r_allowed2
                    r_max2 = max(0.0, r_max2)
                    if r_max2 > r[i] + 1e-8:
                        x[i] = nx
                        y[i] = ny
                        improved = True
                        break

        if not improved:
            break

    # Final uniform growth
    lo, hi = 1.0, 1.03
    for _ in range(15):
        mid = (lo + hi) / 2.0
        if _is_valid(x, y, r * mid, w, h):
            lo = mid
        else:
            hi = mid
    if lo > 1.0 + 1e-9:
        r = r * lo

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    """Check if moving circle i to new_x, new_y is valid."""
    if new_x < r[i] or new_x > w - r[i]:
        return False
    if new_y < r[i] or new_y > h - r[i]:
        return False

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
