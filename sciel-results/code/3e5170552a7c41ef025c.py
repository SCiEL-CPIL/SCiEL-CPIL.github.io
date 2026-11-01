"""Circle packing in perimeter-4 rectangle: SLSQP + smooth pressure-field growth + PBD + SLSQP refine."""
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

    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
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


def _force_relax_unified(z, max_iter=500):
    """Force-directed relaxation in unified parameter space with adaptive step."""
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

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

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

        if max_overlap < convergence_tol:
            break

        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.01, dt * 1.1)

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _soft_min_clearance(x, y, r, w, h, i, soft_len=0.03):
    """Compute smooth soft-min of clearance distances for circle i using log-sum-exp."""
    # Wall clearances
    wall_dists = np.array([
        x[i] - r[i],
        w - x[i] - r[i],
        y[i] - r[i],
        h - y[i] - r[i]
    ])
    
    # Neighbor clearances
    neighbor_dists = []
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        clearance = d - r[i] - r[j]
        neighbor_dists.append(clearance)
    neighbor_dists = np.array(neighbor_dists)
    
    # Combine all clearances
    all_dists = np.concatenate([wall_dists, neighbor_dists])
    
    # Soft-min via log-sum-exp (negative trick)
    # soft_min(a) = -soft_max(-a)
    neg_dists = -all_dists
    max_neg = np.max(neg_dists)
    if max_neg > 50:  # numerical stability
        max_neg = 50
    exp_neg = np.exp((neg_dists - max_neg) / soft_len)
    soft_max_neg = max_neg + soft_len * np.log(np.sum(exp_neg))
    soft_min_val = -soft_max_neg
    
    return soft_min_val


def _pressure_field_growth(z, time_budget, n_rounds=15):
    """
    Sequential radial growth using smooth pressure field.
    Grows circles most-constrained first, interleaved with force relaxation.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    
    soft_len = 0.03
    growth_rate = 0.002
    
    for rnd in range(n_rounds):
        if time.time() > time_budget:
            break
        
        # Compute pressure (negative clearance) for each circle
        pressures = np.zeros(n)
        for i in range(n):
            clearance = _soft_min_clearance(x, y, r, w, h, i, soft_len)
            pressures[i] = -clearance  # higher pressure = more constrained
        
        # Grow circles in order of highest pressure (most constrained first)
        order = np.argsort(-pressures)  # descending pressure
        
        for i in order:
            if time.time() > time_budget:
                break
            
            # Try to grow circle i
            new_r = r[i] * (1.0 + growth_rate)
            
            # Check if growth is feasible
            if _is_valid(x, y, np.concatenate([r[:i], [new_r], r[i+1:]]), w, h):
                r[i] = new_r
            else:
                # Binary search for max feasible growth
                lo = r[i]
                hi = new_r
                for _ in range(10):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid
                r[i] = lo
        
        # Force relaxation to create space after growth
        z_temp = np.empty(3 * N + 1)
        z_temp[0::3][:n] = x
        z_temp[1::3][:n] = y
        z_temp[2::3][:n] = r
        z_temp[-1] = w
        
        z_relaxed = _force_relax_unified(z_temp, max_iter=100)
        
        # Extract positions and radii (keep original radii, just update positions)
        x = z_relaxed[0::3][:n].copy()
        y = z_relaxed[1::3][:n].copy()
        # Keep our radii, but clamp positions
        for i in range(n):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])
    
    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w
    
    if _is_valid(x, y, r, w, h):
        return z_new
    return None


def _pbd_polish(z, time_budget):
    """Position-Based Dynamics polish: resolve overlaps, then grow radii."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    max_pbd_iters = 2000
    for it in range(max_pbd_iters):
        if time.time() > time_budget:
            break

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

        for idx in range(len(iu[0])):
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            if overlap[idx] < 1e-12:
                continue

            dx_ij = x[i_idx] - x[j_idx]
            dy_ij = y[i_idx] - y[j_idx]
            d_ij = d_pairs[idx]
            if d_ij < 1e-12:
                angle = np.random.uniform(0, 2 * np.pi)
                dx_ij = np.cos(angle)
                dy_ij = np.sin(angle)
                d_ij = 1.0

            correction = overlap[idx] * 0.5
            nx = dx_ij / d_ij
            ny = dy_ij / d_ij

            x[i_idx] += nx * correction
            y[i_idx] += ny * correction
            x[j_idx] -= nx * correction
            y[j_idx] -= ny * correction

        for i in range(n):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])

    # Grow radii uniformly
    grow_factor = 1.0003
    for _ in range(300):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
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


def _expansion_relaxation_polish(z, time_budget):
    """Iterative per-circle expansion with binary search."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Per-circle expansion
    for rnd in range(20):
        if time.time() > time_budget - 3:
            break
        
        improved = False
        order = np.argsort(r)
        for i in order:
            if time.time() > time_budget - 3:
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
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            
            if r_max > r[i] + 1e-6:
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
                if lo > r[i] + 1e-7:
                    r[i] = lo
                    improved = True
        
        if not improved:
            break

    # Position optimization via local search
    for rnd in range(20):
        if time.time() > time_budget - 2:
            break
        
        improved = False
        order = np.argsort(r)
        for i in order:
            if time.time() > time_budget - 2:
                break
            
            best_move = None
            best_new_r = r[i]
            
            for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                for step in [0.1, 0.06, 0.035, 0.018, 0.009]:
                    new_x = x[i] + step * np.cos(angle)
                    new_y = y[i] + step * np.sin(angle)
                    
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
            
            if best_move is not None and best_new_r > r[i] + 1e-5:
                x[i] = best_move[0]
                y[i] = best_move[1]
                r[i] = best_new_r
                improved = True
        
        if not improved:
            break

    # Final uniform scaling
    lo = 1.0
    hi = 1.05
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

    maxiter = int(min(1000, time_budget * 200))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 150})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = z_opt[-1]
        h = 2.0 - w

        if _is_valid(x, y, r, w, h):
            return z_opt
        z_repair = z_opt.copy()
        r_idx = np.arange(2, len(z_repair) - 1, 3)
        z_repair[r_idx] *= 0.999
        x_r = z_repair[0::3][:n]
        y_r = z_repair[1::3][:n]
        r_r = z_repair[2::3][:n]
        if _is_valid(x_r, y_r, r_r, w, h):
            return z_repair
        return None
    except Exception:
        return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

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

        # Hexagonal Lattice Initializations
        aspect_ratios_hex = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45]

        for w0 in aspect_ratios_hex:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue

            z = np.zeros(3 * N + 1)
            z[-1] = w0

            r_est = np.sqrt(w0 * h0 / (N * np.pi * 0.9))

            points = []
            y_pos = r_est
            row_idx = 0
            while y_pos < h0 - r_est * 0.5:
                x_offset = 0 if row_idx % 2 == 0 else r_est
                x_pos = r_est + x_offset
                while x_pos < w0 - r_est * 0.5:
                    if x_pos >= r_est and x_pos <= w0 - r_est and y_pos >= r_est and y_pos <= h0 - r_est:
                        points.append((x_pos, y_pos))
                    x_pos += 2 * r_est
                y_pos += r_est * np.sqrt(3)
                row_idx += 1

            if len(points) >= N:
                for i in range(N):
                    cx, cy = points[i]
                    z[3 * i] = cx
                    z[3 * i + 1] = cy
                    z[3 * i + 2] = r_est
                guesses.append(z)
            else:
                for scale in [0.9, 0.8, 0.7]:
                    r_est_new = r_est * scale
                    points = []
                    y_pos = r_est_new
                    row_idx = 0
                    while y_pos < h0 - r_est_new * 0.5:
                        x_offset = 0 if row_idx % 2 == 0 else r_est_new
                        x_pos = r_est_new + x_offset
                        while x_pos < w0 - r_est_new * 0.5:
                            if x_pos >= r_est_new and x_pos <= w0 - r_est_new and y_pos >= r_est_new and y_pos <= h0 - r_est_new:
                                points.append((x_pos, y_pos))
                            x_pos += 2 * r_est_new
                        y_pos += r_est_new * np.sqrt(3)
                        row_idx += 1

                    if len(points) >= N:
                        for i in range(N):
                            cx, cy = points[i]
                            z[3 * i] = cx
                            z[3 * i + 1] = cy
                            z[3 * i + 2] = r_est_new
                        guesses.append(z)
                        break

        # Square Grid Initializations
        side = int(np.ceil(np.sqrt(N)))
        aspect_ratios_sq = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8]
        for w0 in aspect_ratios_sq:
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

        z0_relaxed = _force_relax_unified(z0, max_iter=500)

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
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            continue

    # Phase 2: Polish pipeline
    if best_result is not None and time.time() < deadline - 25:
        # 1. Pressure-field growth (new: smooth sequential radial growth)
        if time.time() < deadline - 22:
            grown = _pressure_field_growth(best_result, deadline - (time.time() - start_time) - 18, n_rounds=12)
            if grown is not None:
                valid, val = check_validity(grown)
                if valid and val > best_value:
                    best_value = val
                    best_result = grown

        # 2. PBD polish
        if time.time() < deadline - 18:
            polished = _pbd_polish(best_result, deadline - (time.time() - start_time) - 12)
            if polished is not None:
                valid, val = check_validity(polished)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished

        # 3. Expansion polish
        if time.time() < deadline - 12:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 8)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 4. Constraint-based SLSQP refinement
        if time.time() < deadline - 8:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 4)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 5. Final expansion pass
        if time.time() < deadline - 4:
            expanded2 = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 1)
            if expanded2 is not None:
                valid, val = check_validity(expanded2)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded2

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

    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
