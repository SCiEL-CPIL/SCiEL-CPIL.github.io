"""Circle packing in perimeter-4 rectangle: Delaunay-based construction + optimization polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, ConvexHull

N = 21
_TIME_DEADLINE = 140.0  # safety margin before 150s kill


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
    """Force-directed relaxation to resolve overlaps and push circles to boundaries (vectorized)."""
    n = N
    dt = dt_init
    iu = np.triu_indices(n, k=1)
    for it in range(max_iter):
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

        # Pairwise repulsion (vectorized)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
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
        # Vectorized accumulation using np.add.at
        np.add.at(fx, iu[0], ux * f_pairs)
        np.add.at(fy, iu[0], uy * f_pairs)
        np.add.at(fx, iu[1], -ux * f_pairs)
        np.add.at(fy, iu[1], -uy * f_pairs)

        # Adaptive step
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
    margin = 0.05
    boundary_pts = np.array([
        [0, 0], [w, 0], [w, h], [0, h],
        [w/2, 0], [w, h/2], [w/2, h], [0, h/2],
        [w/4, 0], [3*w/4, 0], [w, h/4], [w, 3*h/4],
        [3*w/4, h], [w/4, h], [0, 3*h/4], [0, h/4],
    ])
    all_pts = np.vstack([np.column_stack([x, y]), boundary_pts])
    n_total = len(all_pts)
    
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
        
        margin = 0.05
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


def _objective(z, penalty_weight=5000.0):
    """Objective for SLSQP with penalties."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.1 or h < 0.1 or np.any(r <= 0):
        return 1e8

    obj = -np.sum(r)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
    overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)
    obj += penalty_weight * (np.sum(left_viol**2) + np.sum(right_viol**2) +
                             np.sum(bottom_viol**2) + np.sum(top_viol**2))

    if w < 0.35:
        obj += 500.0 * (0.35 - w) ** 2
    elif w > 1.65:
        obj += 500.0 * (w - 1.65) ** 2

    return obj


def _slsqp_optimize(x, y, r, w, h, time_budget):
    """SLSQP optimization with smooth objective."""
    n = N
    z = np.zeros(3 * n + 1)
    z[0::3][:n] = x
    z[1::3][:n] = y
    z[2::3][:n] = r
    z[-1] = w

    bounds = [(0.01, 2.0)] * (2 * n) + [(0.01, 1.0)] * n + [(0.25, 1.75)]
    remaining = time_budget - time.time()
    if remaining < 5:
        return None
    
    maxiter = int(min(600, max(150, remaining * 40)))
    
    try:
        res = minimize(_objective, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
        z_opt = res.x
        x_opt = z_opt[0::3][:n]
        y_opt = z_opt[1::3][:n]
        r_opt = z_opt[2::3][:n]
        w_opt = z_opt[-1]
        h_opt = 2.0 - w_opt

        if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
            return x_opt, y_opt, r_opt, w_opt, h_opt
        else:
            for shrink in [0.995, 0.99, 0.98, 0.97, 0.95]:
                r_repair = r_opt * shrink
                if _is_valid(x_opt, y_opt, r_repair, w_opt, h_opt):
                    return x_opt, y_opt, r_repair, w_opt, h_opt
    except Exception:
        pass
    return None


def _polish_positions(x, y, r, w, h, time_budget):
    """Polish positions via vectorized force-directed method, then grow radii."""
    n = N
    x = x.copy()
    y = y.copy()
    iu = np.triu_indices(n, k=1)

    max_iter = int(min(2000, time_budget * 100))
    dt = 0.001
    for it in range(max_iter):
        if time.time() > time_budget:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 40
        fx += right_viol * 40
        fy -= bottom_viol * 40
        fy += top_viol * 40

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            f_pairs = overlap * 15.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs
            np.add.at(fx, iu[0], ux * f_pairs)
            np.add.at(fy, iu[0], uy * f_pairs)
            np.add.at(fx, iu[1], -ux * f_pairs)
            np.add.at(fy, iu[1], -uy * f_pairs)

        x = x + dt * fx
        y = y + dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0005
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None


def _epigraph_radius_max(x, y, r, w, h, time_budget):
    """Per-circle epigraph-style radius maximization."""
    n = N
    r = r.copy()
    for i in range(n):
        if time.time() > time_budget:
            break
        r_wall = min(x[i], w - x[i], y[i], h - y[i])
        for j in range(n):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_wall = min(r_wall, d - r[j])
        r_max = max(0.0, r_wall)
        if r_max > r[i] + 1e-8:
            r[i] = r_max

    if _is_valid(x, y, r, w, h):
        return r
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


def _smooth_softmin_polish(x, y, r, w, h, time_budget):
    """High-precision polish using a smooth soft-min surrogate for the minimum gap constraint."""
    n = N
    start = time.time()
    
    def _smooth_obj(params):
        x_p = params[0:n]
        y_p = params[n:2*n]
        r_p = params[2*n:3*n]
        
        wall_clear = np.concatenate([
            x_p - r_p,
            w - x_p - r_p,
            y_p - r_p,
            h - y_p - r_p
        ])
        
        dx = x_p[:, None] - x_p[None, :]
        dy = y_p[:, None] - y_p[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r_p[:, None] + r_p[None, :]
        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        pair_clear = d_pairs - r_sum[iu]
        
        all_clear = np.concatenate([wall_clear, pair_clear])
        tau = 0.005
        neg_clear = -all_clear / tau
        neg_clear_clipped = np.clip(neg_clear, -50, 50)
        m = np.max(neg_clear_clipped)
        soft_min_val = -(m + np.log(np.sum(np.exp(neg_clear_clipped - m)))) * tau
        
        obj = -np.sum(r_p) + 200.0 * np.maximum(-soft_min_val, 0.0) ** 2
        return obj
    
    x0 = np.concatenate([x, y, r])
    bounds = [(0.01, w - 0.01) for _ in range(n)] + \
             [(0.01, h - 0.01) for _ in range(n)] + \
             [(0.01, min(w, h) * 0.5) for _ in range(n)]
    
    maxiter = int(min(500, max(100, (time_budget - (time.time() - start)) * 30)))
    
    try:
        res = minimize(_smooth_obj, x0, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100})
        x_new = res.x[0:n]
        y_new = res.x[n:2*n]
        r_new = res.x[2*n:3*n]
        
        if _is_valid(x_new, y_new, r_new, w, h):
            return x_new, y_new, r_new
        else:
            for shrink in [0.995, 0.99, 0.98, 0.97]:
                r_repair = r_new * shrink
                if _is_valid(x_new, y_new, r_repair, w, h):
                    return x_new, y_new, r_repair
    except Exception:
        pass
    return None


def _gradient_expansion_polish(x, y, r, w, h, time_budget):
    """Simultaneous position and radius optimization using smooth clearance gradients."""
    n = N
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    iu = np.triu_indices(n, k=1)
    
    lr = 0.002
    for it in range(2000):
        if time.time() > time_budget:
            break
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        pair_clear = d_pairs - r_sum_pairs
        
        wc_left = x - r
        wc_right = w - x - r
        wc_bottom = y - r
        wc_top = h - y - r
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        overlap = np.maximum(-pair_clear, 0.0)
        if np.any(overlap > 1e-12):
            ux = dx[iu] / d_pairs
            uy = dy[iu] / d_pairs
            
            np.add.at(gx, iu[0], 2 * overlap * ux)
            np.add.at(gy, iu[0], 2 * overlap * uy)
            np.add.at(gx, iu[1], -2 * overlap * ux)
            np.add.at(gy, iu[1], -2 * overlap * uy)
            np.add.at(gr, iu[0], 2 * overlap)
            np.add.at(gr, iu[1], 2 * overlap)
        
        wc_left_v = np.maximum(-wc_left, 0.0)
        wc_right_v = np.maximum(-wc_right, 0.0)
        wc_bottom_v = np.maximum(-wc_bottom, 0.0)
        wc_top_v = np.maximum(-wc_top, 0.0)
        
        gx += 2 * wc_left_v
        gx -= 2 * wc_right_v
        gy += 2 * wc_bottom_v
        gy -= 2 * wc_top_v
        gr += 2 * (wc_left_v + wc_right_v + wc_bottom_v + wc_top_v)
        
        penalty_scale = 50.0
        net_gx = gx * penalty_scale
        net_gy = gy * penalty_scale
        net_gr = np.ones(n) - gr * penalty_scale
        
        max_grad = max(np.max(np.abs(net_gx)), np.max(np.abs(net_gy)), np.max(np.abs(net_gr)), 1e-8)
        if max_grad > 10:
            lr = max(0.0005, lr * 0.8)
        elif max_grad < 1:
            lr = min(0.005, lr * 1.05)
        
        x += lr * net_gx
        y += lr * net_gy
        r += lr * net_gr
        
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        r = np.maximum(r, 0.005)
        
        if np.min(pair_clear) > 1e-6 and np.min(wc_left) > 1e-6 and np.min(wc_right) > 1e-6:
            if np.min(wc_bottom) > 1e-6 and np.min(wc_top) > 1e-6:
                r_trial = r * 1.001
                if _is_valid(x, y, r_trial, w, h):
                    r = r_trial
    
    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None


def _newton_contact_solve(x, y, r, w, h, time_budget):
    """Newton-style contact solve: identify active constraints (circles touching
    walls or each other) and solve the linear system to maximize sum of radii
    while maintaining all contacts. This is effective for tight packings where
    most circles are in contact."""
    n = N
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    
    # Identify active constraints: circles touching walls or each other
    # A constraint is "active" if the clearance is very small
    tol_active = 1e-4
    
    # Build constraint list
    # Each constraint: type, circle index, and for pairs: both indices
    # Wall constraints: left, right, bottom, top
    # Pair constraints: (i, j)
    
    active_constraints = []
    
    # Check wall contacts
    for i in range(n):
        if abs(x[i] - r[i]) < tol_active:  # left wall
            active_constraints.append(('wall_left', i))
        if abs(x[i] + r[i] - w) < tol_active:  # right wall
            active_constraints.append(('wall_right', i))
        if abs(y[i] - r[i]) < tol_active:  # bottom wall
            active_constraints.append(('wall_bottom', i))
        if abs(y[i] + r[i] - h) < tol_active:  # top wall
            active_constraints.append(('wall_top', i))
    
    # Check pair contacts
    iu = np.triu_indices(n, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    d = np.sqrt(dx*dx + dy*dy)
    r_sum = r[iu[0]] + r[iu[1]]
    for k in range(len(iu[0])):
        if abs(d[k] - r_sum[k]) < tol_active:
            active_constraints.append(('pair', iu[0][k], iu[1][k]))
    
    if len(active_constraints) < n:
        # Not enough active constraints for a well-posed system
        return x, y, r
    
    # Build the linear system: we want to find dr such that all active constraints
    # remain satisfied, and we maximize sum(dr).
    # For a tight packing, the contacts form a rigidity structure.
    # Use a simpler approach: iteratively adjust radii to maintain contacts.
    
    # Alternative: use scipy to solve a QP-like problem
    # For now, do a simple iterative adjustment
    
    improved = False
    for iteration in range(50):
        if time.time() > time_budget:
            break
        
        # For each active pair constraint, try to increase both radii slightly
        # while moving centers apart
        for constraint in active_constraints:
            if constraint[0] == 'pair':
                i, j = constraint[1], constraint[2]
                dx_ij = x[i] - x[j]
                dy_ij = y[i] - y[j]
                d_ij = np.sqrt(dx_ij*dx_ij + dy_ij*dy_ij)
                if d_ij < 1e-12:
                    continue
                ux = dx_ij / d_ij
                uy = dy_ij / d_ij
                
                # Try to increase both radii by epsilon
                eps = 0.0005
                dr_i = eps
                dr_j = eps
                
                # Move centers apart to accommodate
                move_i = dr_i / 2.0
                move_j = dr_j / 2.0
                
                new_x_i = x[i] + move_i * ux
                new_y_i = y[i] + move_i * uy
                new_x_j = x[j] - move_j * ux
                new_y_j = y[j] - move_j * uy
                
                # Check if still within bounds
                if (new_x_i >= r[i] + dr_i - tol_active and 
                    new_x_i + r[i] + dr_i <= w + tol_active and
                    new_y_i >= r[i] + dr_i - tol_active and
                    new_y_i + r[i] + dr_i <= h + tol_active and
                    new_x_j >= r[j] + dr_j - tol_active and
                    new_x_j + r[j] + dr_j <= w + tol_active and
                    new_y_j >= r[j] + dr_j - tol_active and
                    new_y_j + r[j] + dr_j <= h + tol_active):
                    
                    # Check no other overlaps
                    valid = True
                    x_test = x.copy()
                    y_test = y.copy()
                    r_test = r.copy()
                    x_test[i] = new_x_i
                    y_test[i] = new_y_i
                    x_test[j] = new_x_j
                    y_test[j] = new_y_j
                    r_test[i] = r[i] + dr_i
                    r_test[j] = r[j] + dr_j
                    
                    if _is_valid(x_test, y_test, r_test, w, h):
                        x[i] = new_x_i
                        y[i] = new_y_i
                        x[j] = new_x_j
                        y[j] = new_y_j
                        r[i] = r_test[i]
                        r[j] = r_test[j]
                        improved = True
    
    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None


def _adaptive_smooth_polish(x, y, r, w, h, time_budget):
    """Adaptive smooth polish with decreasing temperature for the soft-min.
    Starts with a smooth landscape to find good positions, then tightens
    the approximation to the true minimum for precise radius optimization."""
    n = N
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    
    # Phase 1: Smooth optimization with moderate temperature
    tau_start = 0.02
    tau_end = 0.002
    
    def _obj_with_tau(params, tau):
        x_p = params[0:n]
        y_p = params[n:2*n]
        r_p = params[2*n:3*n]
        
        wall_clear = np.concatenate([
            x_p - r_p,
            w - x_p - r_p,
            y_p - r_p,
            h - y_p - r_p
        ])
        
        dx = x_p[:, None] - x_p[None, :]
        dy = y_p[:, None] - y_p[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r_p[:, None] + r_p[None, :]
        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        pair_clear = d_pairs - r_sum[iu]
        
        all_clear = np.concatenate([wall_clear, pair_clear])
        neg_clear = -all_clear / tau
        neg_clear_clipped = np.clip(neg_clear, -50, 50)
        m = np.max(neg_clear_clipped)
        soft_min_val = -(m + np.log(np.sum(np.exp(neg_clear_clipped - m)))) * tau
        
        # Penalty weight increases as we tighten
        penalty_w = 100.0 + 500.0 * (tau_start - tau) / (tau_start - tau_end)
        obj = -np.sum(r_p) + penalty_w * np.maximum(-soft_min_val, 0.0) ** 2
        return obj
    
    params0 = np.concatenate([x, y, r])
    bounds = [(0.01, w - 0.01) for _ in range(n)] + \
             [(0.01, h - 0.01) for _ in range(n)] + \
             [(0.01, min(w, h) * 0.5) for _ in range(n)]
    
    # Multi-stage with decreasing tau
    taus = [tau_start, tau_start * 0.5, tau_start * 0.25, tau_end]
    current_params = params0.copy()
    
    for tau in taus:
        if time.time() > time_budget - 2:
            break
        
        def _obj(params):
            return _obj_with_tau(params, tau)
        
        maxiter = int(min(200, max(50, (time_budget - time.time()) * 10)))
        try:
            res = minimize(_obj, current_params, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
            if res.fun < _obj(current_params):
                current_params = res.x
        except Exception:
            pass
    
    x_new = current_params[0:n]
    y_new = current_params[n:2*n]
    r_new = current_params[2*n:3*n]
    
    if _is_valid(x_new, y_new, r_new, w, h):
        return x_new, y_new, r_new
    else:
        for shrink in [0.995, 0.99, 0.98, 0.97, 0.95]:
            r_repair = r_new * shrink
            if _is_valid(x_new, y_new, r_repair, w, h):
                return x_new, y_new, r_repair
    return None


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_DEADLINE

    best_value = 0.0
    best_result = None

    # Phase 1: Delaunay-based construction for various aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55, 1.8, 0.5]

    for w0 in aspect_ratios:
        if time.time() > deadline - 35:
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

    # Phase 2: SLSQP joint optimization
    if best_result is not None and time.time() < deadline - 40:
        x_best, y_best, r_best, w_best, h_best = best_result
        opt_result = _slsqp_optimize(x_best, y_best, r_best, w_best, h_best, deadline - 20)
        if opt_result is not None:
            x_opt, y_opt, r_opt, w_opt, h_opt = opt_result
            val = float(np.sum(r_opt))
            if val > best_value:
                best_value = val
                best_result = (x_opt.copy(), y_opt.copy(), r_opt.copy(), w_opt, h_opt)
        
        if best_result is not None and time.time() < deadline - 20:
            x_best, y_best, r_best, w_best, h_best = best_result
            opt_result2 = _slsqp_optimize(x_best, y_best, r_best, w_best, h_best, deadline - 10)
            if opt_result2 is not None:
                x_opt2, y_opt2, r_opt2, w_opt2, h_opt2 = opt_result2
                val2 = float(np.sum(r_opt2))
                if val2 > best_value:
                    best_value = val2
                    best_result = (x_opt2.copy(), y_opt2.copy(), r_opt2.copy(), w_opt2, h_opt2)

    # Phase 3: Vectorized force polish + uniform radius growth
    if best_result is not None and time.time() < deadline - 15:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 8
        polished = _polish_positions(x_best, y_best, r_best, w_best, h_best, remaining)
        if polished is not None:
            x_p, y_p, r_p = polished
            val = float(np.sum(r_p))
            if val > best_value:
                best_value = val
                best_result = (x_p, y_p, r_p, w_best, h_best)

    # Phase 4: Epigraph per-circle radius maximization
    if best_result is not None and time.time() < deadline - 10:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 7
        for _ in range(3):
            if time.time() > remaining:
                break
            r_ep = _epigraph_radius_max(x_best, y_best, r_best, w_best, h_best, remaining)
            if r_ep is not None:
                val = float(np.sum(r_ep))
                if val > best_value:
                    best_value = val
                    r_best = r_ep
                    best_result = (x_best.copy(), y_best.copy(), r_ep.copy(), w_best, h_best)
                if time.time() < remaining - 2:
                    polished = _polish_positions(x_best, y_best, r_best, w_best, h_best, remaining - 2)
                    if polished is not None:
                        x_p, y_p, r_p = polished
                        val2 = float(np.sum(r_p))
                        if val2 > best_value:
                            best_value = val2
                            x_best, y_best, r_best = x_p, y_p, r_p
                            best_result = (x_p.copy(), y_p.copy(), r_p.copy(), w_best, h_best)
            else:
                break

    # Phase 5: Gradient-based expansion polish
    if best_result is not None and time.time() < deadline - 10:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 5
        grad_result = _gradient_expansion_polish(x_best, y_best, r_best, w_best, h_best, remaining)
        if grad_result is not None:
            x_g, y_g, r_g = grad_result
            val = float(np.sum(r_g))
            if val > best_value:
                best_value = val
                best_result = (x_g, y_g, r_g, w_best, h_best)

    # Phase 6: Newton contact solve for tight packings
    if best_result is not None and time.time() < deadline - 8:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 4
        newton_result = _newton_contact_solve(x_best, y_best, r_best, w_best, h_best, remaining)
        if newton_result is not None:
            x_n, y_n, r_n = newton_result
            val = float(np.sum(r_n))
            if val > best_value:
                best_value = val
                best_result = (x_n, y_n, r_n, w_best, h_best)

    # Phase 7: Adaptive smooth polish with decreasing temperature
    if best_result is not None and time.time() < deadline - 5:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 2
        adaptive_result = _adaptive_smooth_polish(x_best, y_best, r_best, w_best, h_best, remaining)
        if adaptive_result is not None:
            x_a, y_a, r_a = adaptive_result
            val = float(np.sum(r_a))
            if val > best_value:
                best_value = val
                best_result = (x_a, y_a, r_a, w_best, h_best)

    # Phase 8: Final individual expansion
    if best_result is not None and time.time() < deadline - 2:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 1
        expanded = _expand_individual(x_best, y_best, r_best, w_best, h_best, remaining)
        if expanded is not None:
            x_e, y_e, r_e, _ = expanded
            val = float(np.sum(r_e))
            if val > best_value:
                best_value = val
                best_result = (x_e, y_e, r_e, w_best, h_best)

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result

    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
