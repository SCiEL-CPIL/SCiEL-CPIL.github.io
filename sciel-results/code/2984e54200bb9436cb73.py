"""Circle packing in perimeter-4 rectangle: Delaunay construction + SLSQP + Nelder-Mead + greedy expansion."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

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


def _force_relax(x, y, r, w, h, max_iter=800, dt_init=0.003):
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


def _compute_inradii(x, y, w, h):
    """Compute maximum inscribed circle radius for each point using Delaunay triangulation."""
    n = N
    boundary_pts = np.array([
        [0, 0], [w, 0], [w, h], [0, h],
        [w/2, 0], [w, h/2], [w/2, h], [0, h/2],
        [w/4, 0], [3*w/4, 0], [w, h/4], [w, 3*h/4],
        [3*w/4, h], [w/4, h], [0, 3*h/4], [0, h/4],
    ])
    all_pts = np.vstack([np.column_stack([x, y]), boundary_pts])
    
    try:
        tri = Delaunay(all_pts)
    except Exception:
        return np.full(n, min(w, h) / 6.0)
    
    inradii = np.zeros(n)
    for i in range(n):
        min_dist = min(x[i], w - x[i], y[i], h - y[i])
        
        for simplex in tri.simplices:
            if i in simplex:
                pts = all_pts[simplex]
                ax, ay = pts[0]
                bx, by = pts[1]
                cx, cy = pts[2]
                
                d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                if abs(d) < 1e-12:
                    continue
                ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
                uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d
                
                dist = np.sqrt((x[i] - ux)**2 + (y[i] - uy)**2)
                min_dist = min(min_dist, dist)
        
        inradii[i] = min_dist
    
    return inradii


def _delaunay_construction(w0, h0, max_iter=200, dt=0.002):
    """Construct initial packing using Delaunay-based inradius optimization."""
    n = N
    
    rows = 4
    cols = 6
    x = np.zeros(n)
    y = np.zeros(n)
    r = np.zeros(n)
    
    for i in range(n):
        row = i // cols
        col = i % cols
        x[i] = (col + 0.5) * (w0 / cols) + np.random.uniform(-0.1, 0.1) * w0 / cols
        y[i] = (row + 0.5) * (h0 / rows) + np.random.uniform(-0.1, 0.1) * h0 / rows
        x[i] = np.clip(x[i], 0.05, w0 - 0.05)
        y[i] = np.clip(y[i], 0.05, h0 - 0.05)
        r[i] = 0.05
    
    for it in range(max_iter):
        inradii = _compute_inradii(x, y, w0, h0)
        r = np.maximum(inradii, 0.01)
        
        boundary_pts = np.array([
            [0, 0], [w0, 0], [w0, h0], [0, h0],
            [w0/2, 0], [w0, h0/2], [w0/2, h0], [0, h0/2],
        ])
        all_pts = np.vstack([np.column_stack([x, y]), boundary_pts])
        
        try:
            tri = Delaunay(all_pts)
        except Exception:
            break
        
        new_x = x.copy()
        new_y = y.copy()
        
        for i in range(n):
            cx_sum = 0.0
            cy_sum = 0.0
            weight_sum = 0.0
            
            for simplex in tri.simplices:
                if i in simplex:
                    pts = all_pts[simplex]
                    ax, ay = pts[0]
                    bx, by = pts[1]
                    cx, cy = pts[2]
                    
                    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                    if abs(d) < 1e-12:
                        continue
                    ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
                    uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d
                    
                    dist = np.sqrt((x[i] - ux)**2 + (y[i] - uy)**2)
                    weight = 1.0 / max(dist, 0.01)
                    cx_sum += ux * weight
                    cy_sum += uy * weight
                    weight_sum += weight
            
            if weight_sum > 0:
                new_x[i] = cx_sum / weight_sum
                new_y[i] = cy_sum / weight_sum
        
        x = x + dt * (new_x - x)
        y = y + dt * (new_y - y)
        x = np.clip(x, 0.05, w0 - 0.05)
        y = np.clip(y, 0.05, h0 - 0.05)
    
    r = _compute_inradii(x, y, w0, h0)
    x, y = _force_relax(x, y, r, w0, h0, max_iter=300)
    
    if _is_valid(x, y, r, w0, h0):
        return x, y, r
    else:
        r *= 0.9
        if _is_valid(x, y, r, w0, h0):
            return x, y, r
    return None


def _slsqp_objective(z, penalty_weight=5000.0):
    """Objective for SLSQP with smooth log-sum-exp penalties."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.1 or h < 0.1 or np.any(r <= 0):
        return 1e8

    obj = -np.sum(r)

    # Smooth pairwise overlap penalty using log-sum-exp
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
    violations = r_sum[iu] - d_pairs
    
    # Smooth penalty: use soft-max to avoid hard max
    beta = 20.0
    v_max = np.max(violations)
    if v_max > 0:
        smooth_penalty = np.log(np.sum(np.exp(beta * violations))) / beta
        obj += penalty_weight * smooth_penalty
    else:
        # All non-positive, use quadratic on the most violated
        obj += penalty_weight * np.sum(np.maximum(violations, 0.0)**2)

    # Wall violations with smooth penalty
    lv = r - x
    rv = x + r - w
    bv = r - y
    tv = y + r - h
    all_viol = np.concatenate([lv, rv, bv, tv])
    v_max_w = np.max(all_viol)
    if v_max_w > 0:
        smooth_wall = np.log(np.sum(np.exp(beta * all_viol))) / beta
        obj += penalty_weight * smooth_wall
    else:
        obj += penalty_weight * np.sum(np.maximum(all_viol, 0.0)**2)

    # Aspect ratio constraints
    if w < 0.4:
        obj += penalty_weight * (0.4 - w) ** 2
    elif w > 1.6:
        obj += penalty_weight * (w - 1.6) ** 2

    return obj


def _nm_objective(z, penalty_weight=5000.0):
    """Objective for Nelder-Mead with penalties."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.3 or h < 0.3 or np.any(r < 0.001):
        return 1e8

    obj = -np.sum(r)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    lv = np.maximum(r - x, 0.0)
    rv = np.maximum(x + r - w, 0.0)
    bv = np.maximum(r - y, 0.0)
    tv = np.maximum(y + r - h, 0.0)
    obj += penalty_weight * (np.sum(lv*lv) + np.sum(rv*rv) + np.sum(bv*bv) + np.sum(tv*tv))

    if w < 0.6:
        obj += penalty_weight * (0.6 - w) ** 2
    elif w > 1.4:
        obj += penalty_weight * (w - 1.4) ** 2

    return obj


def _slsqp_optimize(x, y, r, w, h, time_budget):
    """SLSQP optimization with smooth objective."""
    n = N
    z = np.zeros(3 * n + 1)
    z[0::3][:n] = x
    z[1::3][:n] = y
    z[2::3][:n] = r
    z[-1] = w

    bounds = [(0.01, 2.0)] * (2 * n) + [(0.01, 1.0)] * n + [(0.3, 1.7)]
    remaining = time_budget - time.time()
    if remaining < 5:
        return None
    
    maxiter = int(min(400, max(100, remaining * 30)))
    
    try:
        res = minimize(_slsqp_objective, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})
        z_opt = res.x
        x_opt = z_opt[0::3][:n]
        y_opt = z_opt[1::3][:n]
        r_opt = z_opt[2::3][:n]
        w_opt = z_opt[-1]
        h_opt = 2.0 - w_opt

        if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
            return x_opt, y_opt, r_opt, w_opt, h_opt
        else:
            r_repair = r_opt * 0.98
            if _is_valid(x_opt, y_opt, r_repair, w_opt, h_opt):
                return x_opt, y_opt, r_repair, w_opt, h_opt
    except Exception:
        pass
    return None


def _nelder_mead_optimize(x, y, r, w, h, time_budget):
    """Nelder-Mead optimization (gradient-free)."""
    n = N
    z = np.zeros(3 * n + 1)
    z[0::3][:n] = x
    z[1::3][:n] = y
    z[2::3][:n] = r
    z[-1] = w

    remaining = time_budget - time.time()
    if remaining < 5:
        return None
    
    maxiter = int(min(3000, max(200, remaining * 40)))
    
    try:
        res = minimize(_nm_objective, z, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-9,
                                'adaptive': True})
        z_opt = res.x
        x_opt = z_opt[0::3][:n]
        y_opt = z_opt[1::3][:n]
        r_opt = z_opt[2::3][:n]
        w_opt = z_opt[-1]
        h_opt = 2.0 - w_opt

        if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
            return x_opt, y_opt, r_opt, w_opt, h_opt
        else:
            r_repair = r_opt * 0.9
            if _is_valid(x_opt, y_opt, r_repair, w_opt, h_opt):
                return x_opt, y_opt, r_repair, w_opt, h_opt
    except Exception:
        pass
    return None


def _greedy_expand(x, y, r, w, h, time_budget):
    """Iteratively expand each circle to its maximum possible radius."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    improved = True
    rounds = 0
    while improved and rounds < 50:
        if time.time() > time_budget:
            break
        improved = False
        rounds += 1
        for i in range(N):
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-8:
                r[i] = r_max
                improved = True
    return x, y, r


def _local_search_positions(x, y, r, w, h, time_budget):
    """Jiggle-based local search: perturb positions, maintain feasibility, grow radii."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    rng = np.random.default_rng(42)
    best_sum = float(np.sum(r))
    
    # Adaptive step size
    step = 0.003
    
    for it in range(8000):
        if time.time() > time_budget:
            break
        
        # Jiggle: move 1-3 circles simultaneously
        n_move = 1 + (it % 3)
        idxs = rng.choice(N, size=n_move, replace=False)
        
        old_x = x[idxs].copy()
        old_y = y[idxs].copy()
        
        # Try random displacement
        dxs = rng.normal(0, step, size=n_move)
        dys = rng.normal(0, step, size=n_move)
        x[idxs] += dxs
        y[idxs] += dys
        
        # Clamp to walls
        x[idxs] = np.clip(x[idxs], r[idxs], w - r[idxs])
        y[idxs] = np.clip(y[idxs], r[idxs], h - r[idxs])
        
        # Check feasibility
        valid = True
        for i in idxs:
            if x[i] < r[i] - 1e-9 or x[i] > w - r[i] + 1e-9:
                valid = False
                break
            if y[i] < r[i] - 1e-9 or y[i] > h - r[i] + 1e-9:
                valid = False
                break
            for j in range(N):
                if i == j:
                    continue
                ddx = x[i] - x[j]
                ddy = y[i] - y[j]
                if ddx*ddx + ddy*ddy < (r[i] + r[j])**2 - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Try to grow radii
            x2, y2, r2 = _greedy_expand(x, y, r, w, h, time_budget)
            new_sum = float(np.sum(r2))
            if new_sum > best_sum:
                best_sum = new_sum
                r = r2
                x = x2
                y = y2
                # Increase step slightly on success
                step = min(0.01, step * 1.05)
            else:
                # Revert
                x[idxs] = old_x
                y[idxs] = old_y
                step = max(0.0005, step * 0.95)
        else:
            # Revert
            x[idxs] = old_x
            y[idxs] = old_y
            step = max(0.0005, step * 0.9)
    
    return x, y, r


def _polish_positions(x, y, r, w, h, time_budget):
    """High-precision polish: force-directed relaxation + iterative radius growth."""
    n = N
    x = x.copy()
    y = y.copy()
    r = r.copy()

    # Phase 1: Force-directed relaxation with adaptive step
    max_iter = int(min(3000, time_budget * 150))
    dt = 0.0005
    for it in range(max_iter):
        if time.time() > time_budget:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall forces with stronger coefficient
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

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

        x = x + dt * fx
        y = y + dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Phase 2: Iterative radius growth with binary search
    for _ in range(300):
        if time.time() > time_budget:
            break
        # Try to grow each circle individually
        improved = False
        for i in range(n):
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                ddx = x[i] - x[j]
                ddy = y[i] - y[j]
                d = np.sqrt(ddx*ddx + ddy*ddy)
                r_allowed = d - r[j]
                r_max = min(r_max, r_allowed)
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-10:
                r[i] = r_max
                improved = True
        if not improved:
            break

    # Phase 3: Final uniform scaling via binary search
    lo, hi = 1.0, 1.1
    for _ in range(40):
        if time.time() > time_budget:
            break
        mid = (lo + hi) / 2
        r_trial = r * mid
        if _is_valid(x, y, r_trial, w, h):
            lo = mid
        else:
            hi = mid
    r = r * lo

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


def _binary_search_radius(x, y, r, w, h, max_scale=1.5):
    """Binary search for maximum uniform radius scaling that remains valid."""
    lo = 1.0
    hi = max_scale
    for _ in range(30):
        mid = (lo + hi) / 2
        r_trial = r * mid
        if _is_valid(x, y, r_trial, w, h):
            lo = mid
        else:
            hi = mid
    r_opt = r * lo
    if _is_valid(x, y, r_opt, w, h):
        return r_opt
    return r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_value = 0.0
    best_result = None

    # Phase 1: Delaunay-based construction for various aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55, 1.8, 0.5]

    for w0 in aspect_ratios:
        if time.time() > deadline - 40:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        result = _delaunay_construction(w0, h0, max_iter=100)
        if result is not None:
            x, y, r = result
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w0, h0)

        best_grid = None
        best_grid_r = 0
        for rows in range(3, 8):
            cols = int(np.ceil(N / rows))
            cell_w = w0 / cols
            cell_h = h0 / rows
            r_grid = min(cell_w, cell_h) * 0.48
            if r_grid > best_grid_r:
                best_grid_r = r_grid
                best_grid = (rows, cols)

        if best_grid is not None:
            rows, cols = best_grid
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)

            for i in range(N):
                row = i // cols
                col = i % cols
                x[i] = (col + 0.5) * (w0 / cols)
                y[i] = (row + 0.5) * (h0 / rows)
                r[i] = min(w0 / cols, h0 / rows) * 0.45

            x, y = _force_relax(x, y, r, w0, h0, max_iter=300)

            if not _is_valid(x, y, r, w0, h0):
                r *= 0.95
                if not _is_valid(x, y, r, w0, h0):
                    continue

            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w0, h0)

    # Phase 2: Multi-method optimization
    if best_result is not None and time.time() < deadline - 30:
        x_best, y_best, r_best, w_best, h_best = best_result
        
        # SLSQP
        opt_result = _slsqp_optimize(x_best, y_best, r_best, w_best, h_best, deadline - 10)
        if opt_result is not None:
            x_opt, y_opt, r_opt, w_opt, h_opt = opt_result
            val = float(np.sum(r_opt))
            if val > best_value:
                best_value = val
                best_result = (x_opt.copy(), y_opt.copy(), r_opt.copy(), w_opt, h_opt)

        # Nelder-Mead on best result
        if time.time() < deadline - 15:
            x_best, y_best, r_best, w_best, h_best = best_result
            nm_result = _nelder_mead_optimize(x_best, y_best, r_best, w_best, h_best, deadline - 5)
            if nm_result is not None:
                x_nm, y_nm, r_nm, w_nm, h_nm = nm_result
                val = float(np.sum(r_nm))
                if val > best_value:
                    best_value = val
                    best_result = (x_nm.copy(), y_nm.copy(), r_nm.copy(), w_nm, h_nm)

    # Phase 3: Local search with greedy expansion
    if best_result is not None and time.time() < deadline - 10:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 5
        if remaining > 5:
            x_ls, y_ls, r_ls = _local_search_positions(x_best, y_best, r_best, w_best, h_best, remaining)
            if _is_valid(x_ls, y_ls, r_ls, w_best, h_best):
                val = float(np.sum(r_ls))
                if val > best_value:
                    best_value = val
                    best_result = (x_ls, y_ls, r_ls, w_best, h_best)

    # Phase 4: Polish
    if best_result is not None and time.time() < deadline - 8:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 3
        polished = _polish_positions(x_best, y_best, r_best, w_best, h_best, remaining)
        if polished is not None:
            x_p, y_p, r_p = polished
            val = float(np.sum(r_p))
            if val > best_value:
                best_value = val
                best_result = (x_p, y_p, r_p, w_best, h_best)

    # Phase 5: Greedy expansion
    if best_result is not None and time.time() < deadline - 5:
        x_best, y_best, r_best, w_best, h_best = best_result
        x_g, y_g, r_g = _greedy_expand(x_best, y_best, r_best, w_best, h_best, deadline - 2)
        if _is_valid(x_g, y_g, r_g, w_best, h_best):
            val = float(np.sum(r_g))
            if val > best_value:
                best_value = val
                best_result = (x_g, y_g, r_g, w_best, h_best)

    # Phase 6: Binary search radius scaling
    if best_result is not None and time.time() < deadline - 2:
        x_best, y_best, r_best, w_best, h_best = best_result
        r_scaled = _binary_search_radius(x_best, y_best, r_best, w_best, h_best, max_scale=1.5)
        if _is_valid(x_best, y_best, r_scaled, w_best, h_best):
            val = float(np.sum(r_scaled))
            if val > best_value:
                best_value = val
                best_result = (x_best, y_best, r_scaled, w_best, h_best)
        
        # Additional: try per-circle binary search for non-uniform scaling
        if time.time() < deadline - 1:
            x_b, y_b, r_b, w_b, h_b = best_result
            for i in range(N):
                if time.time() > deadline - 1:
                    break
                # Find max radius for circle i given others fixed
                r_wall = min(x_b[i], w_b - x_b[i], y_b[i], h_b - y_b[i])
                lo_i, hi_i = r_b[i], r_wall
                for _ in range(25):
                    mid_i = (lo_i + hi_i) / 2
                    r_trial = r_b.copy()
                    r_trial[i] = mid_i
                    if _is_valid(x_b, y_b, r_trial, w_b, h_b):
                        lo_i = mid_i
                    else:
                        hi_i = mid_i
                if lo_i > r_b[i] + 1e-9:
                    r_b[i] = lo_i
                    if _is_valid(x_b, y_b, r_b, w_b, h_b):
                        val = float(np.sum(r_b))
                        if val > best_value:
                            best_value = val
                            best_result = (x_b, y_b, r_b, w_b, h_b)

    # Phase 7: Final individual expansion
    if best_result is not None and time.time() < deadline - 1:
        x_best, y_best, r_best, w_best, h_best = best_result
        expanded = _expand_individual(x_best, y_best, r_best, w_best, h_best, deadline)
        if expanded is not None:
            x_e, y_e, r_e, _ = expanded
            val = float(np.sum(r_e))
            if val > best_value:
                best_value = val
                best_result = (x_e, y_e, r_e, w_best, h_best)

    if best_result is None:
        # Fallback: grid packing
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result

    # Final safety shrink
    for _ in range(20):
        r_trial = r * (1 - 1e-8)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    # Ensure perimeter constraint
    if 2 * (w + h) > 4.0 + 1e-9:
        scale = 4.0 / (2 * (w + h))
        x = x * scale
        y = y * scale
        r = r * scale
        w = w * scale
        h = h * scale

    return np.column_stack((x, y, r))
