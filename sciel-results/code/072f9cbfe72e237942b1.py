"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Combined approach:
 - Smooth surrogate objective (softplus for non-overlap, squared for boundary)
 - L-BFGS-B optimization with multiple aspect ratios
 - Simulated annealing refinement
 - Force-directed relaxation phase
 - Final high-precision polish pass
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _softplus(x, beta=10.0):
    """Numerically stable softplus."""
    if x > 1.0 / beta:
        return x
    return np.log1p(np.exp(beta * x)) / beta


def _compute_overlap_penalty(xs, ys, rs, beta=10.0):
    """Smooth overlap penalty using softplus surrogate."""
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = rs[i] + rs[j]
            val = min_dist - dist
            penalty += _softplus(val, beta)
    return penalty


def _boundary_penalty(xs, ys, rs, w, h):
    """Squared boundary violation penalty."""
    pen = 0.0
    for i in range(N):
        if xs[i] < rs[i]:
            pen += (rs[i] - xs[i]) ** 2
        if xs[i] > w - rs[i]:
            pen += (xs[i] - (w - rs[i])) ** 2
        if ys[i] < rs[i]:
            pen += (rs[i] - ys[i]) ** 2
        if ys[i] > h - rs[i]:
            pen += (ys[i] - (h - rs[i])) ** 2
    return pen


def _objective(params, w, h, beta=10.0):
    """
    Objective: minimize penalty - sum(r).
    params = [x0, y0, r0, x1, y1, r1, ..., x20, y20, r20]
    """
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    if np.any(rs <= 0):
        return 1e10 + np.sum(np.minimum(rs, 0) ** 2)
    
    overlap_pen = _compute_overlap_penalty(xs, ys, rs, beta)
    boundary_pen = _boundary_penalty(xs, ys, rs, w, h)
    
    return overlap_pen + boundary_pen - np.sum(rs)


def _validity_margin(result, tol=1e-4):
    """Return minimum clearance (positive = valid). result is (N,3): x, y, r."""
    xs = result[:, 0]
    ys = result[:, 1]
    rs = result[:, 2]
    if np.any(rs < 1e-6):
        return -1.0
    
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    bw = max_x - min_x
    bh = max_y - min_y
    
    if bw + bh > 2.0 + tol:
        return -(bw + bh - 2.0)
    
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
            m = d - (rs[i] + rs[j])
            if m < -tol:
                return m
    
    return min(bw + bh - 2.0, 0.0)


def _scale_to_perimeter(result):
    """Scale result about its bounding-box center so perimeter <= 4."""
    xs = result[:, 0]
    ys = result[:, 1]
    rs = result[:, 2]
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    bw = max_x - min_x
    bh = max_y - min_y
    perim = 2.0 * (bw + bh)
    if perim > 4.0 + 1e-9 and perim > 0:
        scale = 4.0 / perim
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        for i in range(N):
            result[i, 0] = cx + (result[i, 0] - cx) * scale
            result[i, 1] = cy + (result[i, 1] - cy) * scale
            result[i, 2] = result[i, 2] * scale
    return result


def _check_and_update(params, w, h, best_state):
    """Check validity and update best if improved."""
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    if np.any(rs < 1e-9):
        return
    if np.any(xs - rs < -1e-6) or np.any(xs + rs > w + 1e-6):
        return
    if np.any(ys - rs < -1e-6) or np.any(ys + rs > h + 1e-6):
        return
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            min_dist = rs[i] + rs[j]
            if dist_sq < (min_dist - 1e-7) ** 2:
                return
    
    val = np.sum(rs)
    if val > best_state['value']:
        best_state['value'] = val
        best_state['result'] = np.zeros((N, 3))
        best_state['result'][:, 0] = xs
        best_state['result'][:, 1] = ys
        best_state['result'][:, 2] = rs


def _sa_optimize(w, h, x0, max_iters=2000, seed=42, deadline=None):
    """Simulated annealing for fixed w, h."""
    xs = x0[0::3].copy()
    ys = x0[1::3].copy()
    rs = x0[2::3].copy()
    
    r_max = min(w, h) / 2.0
    T = 0.05
    cooling = 0.999
    current_cost = _compute_overlap_penalty(xs, ys, rs) - np.sum(rs)
    
    rng = np.random.default_rng(seed)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        idx = rng.integers(0, N)
        old_x, old_y, old_r = xs[idx], ys[idx], rs[idx]
        
        scale = T * 0.3
        xs[idx] += rng.normal(0, scale)
        ys[idx] += rng.normal(0, scale)
        rs[idx] += rng.normal(0, scale * 0.5)
        
        xs[idx] = np.clip(xs[idx], rs[idx], w - rs[idx])
        ys[idx] = np.clip(ys[idx], rs[idx], h - rs[idx])
        rs[idx] = np.clip(rs[idx], 1e-4, r_max)
        
        new_cost = _compute_overlap_penalty(xs, ys, rs) - np.sum(rs)
        delta = new_cost - current_cost
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_cost = new_cost
        else:
            xs[idx], ys[idx], rs[idx] = old_x, old_y, old_r
        
        T *= cooling
    
    x_final = np.zeros(3 * N)
    x_final[0::3] = xs
    x_final[1::3] = ys
    x_final[2::3] = rs
    return x_final


def _grid_initial(w, h):
    """Create grid arrangement of 21 circles."""
    a = int(np.ceil(np.sqrt(N * w / h)))
    b = int(np.ceil(N / a))
    while a * b < N:
        b += 1
    
    x0 = np.zeros(3 * N)
    for i in range(N):
        col = i % a
        row = i // a
        x0[3 * i] = (col + 0.5) * w / a
        x0[3 * i + 1] = (row + 0.5) * h / b
        x0[3 * i + 2] = min(w / a, h / b) * 0.45
    
    return x0


def _hex_initial(w, h):
    """Create hexagonal lattice initial arrangement."""
    x0 = np.zeros(3 * N)
    
    # Estimate radius for hexagonal packing
    r_est = np.sqrt(0.9069 * w * h / (N * np.pi))
    
    positions = []
    r_current = r_est
    
    for attempt in range(20):
        positions = []
        y = r_current
        row_idx = 0
        while y <= h - r_current + 1e-9:
            if row_idx % 2 == 0:
                x_start = r_current
            else:
                x_start = r_current + r_current
            
            x = x_start
            while x <= w - r_current + 1e-9:
                positions.append((x, y, r_current))
                x += 2.0 * r_current
            y += np.sqrt(3) * r_current
            row_idx += 1
        
        if len(positions) >= N:
            break
        else:
            r_current *= 0.95
    
    if len(positions) < N:
        # Fallback to square grid
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        positions = []
        for i in range(N):
            col = i % a
            row = i // a
            positions.append(((col + 0.5) * w / a, (row + 0.5) * h / b, min(w / a, h / b) * 0.45))
    
    for i in range(N):
        if i < len(positions):
            x0[3 * i] = positions[i][0]
            x0[3 * i + 1] = positions[i][1]
            x0[3 * i + 2] = positions[i][2]
        else:
            x0[3 * i] = w / 2
            x0[3 * i + 1] = h / 2
            x0[3 * i + 2] = r_est * 0.5
    
    return x0


def _force_directed_relaxation(w, h, x0, max_iters=500, deadline=None):
    """Force-directed relaxation to improve contact graph."""
    xs = x0[0::3].copy()
    ys = x0[1::3].copy()
    rs = x0[2::3].copy()
    
    step_size = 0.01
    decay = 0.999
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rs[i] + rs[j]
                
                if dist < 1e-10:
                    dx = 0.01
                    dy = 0.01
                    dist = np.sqrt(dx * dx + dy * dy)
                
                # Repulsive force if overlapping
                if dist < min_dist:
                    force = (min_dist - dist) * 10.0
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
                    # Slightly increase radii if space allows
                    fr[i] += 0.001
                    fr[j] += 0.001
        
        # Boundary forces
        for i in range(N):
            if xs[i] < rs[i]:
                fx[i] += (rs[i] - xs[i]) * 50.0
            if xs[i] > w - rs[i]:
                fx[i] -= (xs[i] - (w - rs[i])) * 50.0
            if ys[i] < rs[i]:
                fy[i] += (rs[i] - ys[i]) * 50.0
            if ys[i] > h - rs[i]:
                fy[i] -= (ys[i] - (h - rs[i])) * 50.0
        
        # Update positions
        r_max = min(w, h) / 2.0
        for i in range(N):
            xs[i] += step_size * fx[i]
            ys[i] += step_size * fy[i]
            rs[i] += step_size * fr[i] * 0.1
            xs[i] = np.clip(xs[i], rs[i], w - rs[i])
            ys[i] = np.clip(ys[i], rs[i], h - rs[i])
            rs[i] = np.clip(rs[i], 1e-4, r_max)
        
        step_size *= decay
    
    x_final = np.zeros(3 * N)
    x_final[0::3] = xs
    x_final[1::3] = ys
    x_final[2::3] = rs
    return x_final


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_state = {'value': -np.inf, 'result': None}
    
    # Try multiple aspect ratios
    widths = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5]
    
    best_x_for_warm = None
    
    for w in widths:
        if time.time() > deadline - 20.0:
            break
        h = 2.0 - w
        
        # Initialize with hexagonal pattern
        x0 = _hex_initial(w, h)
        
        # L-BFGS-B optimization with smooth objective
        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-4, r_max))
        
        try:
            # First pass with moderate beta
            res = minimize(lambda p: _objective(p, w, h, beta=5.0), x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
            if time.time() > deadline - 10.0:
                break
            _check_and_update(res.x, w, h, best_state)
            
            # Second pass with higher beta for more precise optimization
            if time.time() < deadline - 15.0:
                res2 = minimize(lambda p: _objective(p, w, h, beta=20.0), res.x, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10})
                _check_and_update(res2.x, w, h, best_state)
                if best_x_for_warm is None or np.sum(res2.x[2::3]) > np.sum(best_x_for_warm[2::3]):
                    best_x_for_warm = res2.x.copy()
        except Exception:
            pass
        
        # Force-directed relaxation
        if time.time() < deadline - 30.0:
            x_fd_start = best_x_for_warm if best_x_for_warm is not None else x0
            x_fd = _force_directed_relaxation(w, h, x_fd_start, max_iters=500, deadline=deadline - 10.0)
            _check_and_update(x_fd, w, h, best_state)
            
            # L-BFGS-B after force-directed
            if time.time() < deadline - 20.0:
                try:
                    res_fd = minimize(lambda p: _objective(p, w, h, beta=30.0), x_fd, method='L-BFGS-B', bounds=bounds,
                                     options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10})
                    _check_and_update(res_fd.x, w, h, best_state)
                    if best_x_for_warm is None or np.sum(res_fd.x[2::3]) > np.sum(best_x_for_warm[2::3]):
                        best_x_for_warm = res_fd.x.copy()
                except Exception:
                    pass
        
        # Simulated annealing refinement
        if time.time() < deadline - 25.0:
            x_sa_start = best_x_for_warm if best_x_for_warm is not None else x0
            x_sa = _sa_optimize(w, h, x_sa_start, max_iters=1500, seed=42, deadline=deadline - 10.0)
            _check_and_update(x_sa, w, h, best_state)
            
            # Second SA pass with different seed
            if time.time() < deadline - 15.0:
                rng_sa = np.random.default_rng(123)
                x_sa2 = x_sa.copy()
                for i in range(N):
                    x_sa2[3 * i] += rng_sa.normal(0, 0.01)
                    x_sa2[3 * i + 1] += rng_sa.normal(0, 0.01)
                    x_sa2[3 * i + 2] += rng_sa.normal(0, 0.005)
                x_sa2 = _sa_optimize(w, h, x_sa2, max_iters=1000, seed=123, deadline=deadline - 10.0)
                _check_and_update(x_sa2, w, h, best_state)

    # Final high-precision polish pass
    if time.time() < deadline - 5.0 and best_state['result'] is not None:
        # Extract best parameters
        xs = best_state['result'][:, 0]
        ys = best_state['result'][:, 1]
        rs = best_state['result'][:, 2]
        
        # Determine w, h from bounding box
        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        bw = max_x - min_x
        bh = max_y - min_y
        w_polish = bw
        h_polish = bh
        
        if w_polish > 0 and h_polish > 0:
            x_polish = np.zeros(3 * N)
            x_polish[0::3] = xs - min_x
            x_polish[1::3] = ys - min_y
            x_polish[2::3] = rs
            
            bounds_polish = []
            r_max_polish = min(w_polish, h_polish) / 2.0
            for i in range(N):
                bounds_polish.append((0, w_polish))
                bounds_polish.append((0, h_polish))
                bounds_polish.append((1e-4, r_max_polish))
            
            try:
                # Force-directed polish
                x_fd_polish = _force_directed_relaxation(w_polish, h_polish, x_polish, max_iters=300, deadline=deadline - 2.0)
                _check_and_update(x_fd_polish, w_polish, h_polish, best_state)
                
                # L-BFGS-B polish
                res_polish = minimize(lambda p: _objective(p, w_polish, h_polish, beta=50.0), x_fd_polish,
                                      method='L-BFGS-B', bounds=bounds_polish,
                                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
                _check_and_update(res_polish.x, w_polish, h_polish, best_state)
            except Exception:
                pass
    
    # Fallback if no valid result found
    if best_state['result'] is None or best_state['value'] <= 0:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.1 - 1e-4
        best_result = np.zeros((N, 3))
        for i in range(a):
            for j in range(b):
                idx = i * b + j
                best_result[idx, 0] = (i + 0.5) * w / a
                best_result[idx, 1] = (j + 0.5) * h / b
                best_result[idx, 2] = r
        _scale_to_perimeter(best_result)
        return best_result
    
    # Scale to ensure perimeter <= 4
    result = best_state['result'].copy()
    _scale_to_perimeter(result)
    return result
