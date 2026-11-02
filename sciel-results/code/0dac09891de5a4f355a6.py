"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Combined approach:
 - Smooth surrogate objective (softplus for non-overlap, squared for boundary)
 - Analytical gradients for L-BFGS-B
 - Delaunay-based LP radii solver for exact feasibility given centers
 - Joint aspect ratio optimization integrated into the search
 - Simulated annealing refinement
 - Final high-precision polish pass with joint aspect
"""
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay

N = 21


def _softplus_vec(x, beta=10.0):
    """Numerically stable softplus, vectorized."""
    z = beta * x
    return np.where(z > 0, (z + np.log1p(np.exp(-z))) / beta, np.log1p(np.exp(z)) / beta)


def _sigmoid_vec(x, beta=10.0):
    """Derivative of softplus: sigmoid(beta * x), vectorized and stable."""
    z = beta * x
    return np.where(z > 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


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
    
    # Overlap penalty (vectorized)
    ii, jj = np.triu_indices(N, k=1)
    dx = xs[ii] - xs[jj]
    dy = ys[ii] - ys[jj]
    dist = np.sqrt(dx * dx + dy * dy)
    dist = np.maximum(dist, 1e-12)
    min_dist = rs[ii] + rs[jj]
    val = min_dist - dist
    overlap_pen = np.sum(_softplus_vec(val, beta))
    
    # Boundary penalty (squared, vectorized)
    boundary_pen = 0.0
    boundary_pen += np.sum(np.maximum(rs - xs, 0.0) ** 2)
    boundary_pen += np.sum(np.maximum(xs - (w - rs), 0.0) ** 2)
    boundary_pen += np.sum(np.maximum(rs - ys, 0.0) ** 2)
    boundary_pen += np.sum(np.maximum(ys - (h - rs), 0.0) ** 2)
    
    return overlap_pen + boundary_pen - np.sum(rs)


def _objective_grad(params, w, h, beta=10.0):
    """
    Analytical gradient of _objective (vectorized).
    """
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    grad = np.zeros(3 * N)
    
    if np.any(rs <= 0):
        grad[2::3] = 2.0 * np.minimum(rs, 0)
        return grad

    # 1. Gradient of -sum(r)
    grad[2::3] -= 1.0
    
    # 2. Gradient of Overlap Penalty (vectorized)
    ii, jj = np.triu_indices(N, k=1)
    dx = xs[ii] - xs[jj]
    dy = ys[ii] - ys[jj]
    dist = np.sqrt(dx * dx + dy * dy)
    dist_safe = np.maximum(dist, 1e-12)
    min_dist = rs[ii] + rs[jj]
    val = min_dist - dist_safe
    sp_grad = _sigmoid_vec(val, beta)
    
    inv = sp_grad / dist_safe
    np.add.at(grad, 3 * ii, -inv * dx)
    np.add.at(grad, 3 * ii + 1, -inv * dy)
    np.add.at(grad, 3 * ii + 2, sp_grad)
    np.add.at(grad, 3 * jj, inv * dx)
    np.add.at(grad, 3 * jj + 1, inv * dy)
    np.add.at(grad, 3 * jj + 2, sp_grad)

    # 3. Gradient of Boundary Penalty (vectorized)
    idx = np.arange(N)
    mask = xs < rs
    grad[3 * idx[mask]] += 2.0 * (xs[mask] - rs[mask])
    grad[3 * idx[mask] + 2] += 2.0 * (rs[mask] - xs[mask])
    mask = xs > w - rs
    grad[3 * idx[mask]] += 2.0 * (xs[mask] - w + rs[mask])
    grad[3 * idx[mask] + 2] += -2.0 * (xs[mask] - w + rs[mask])
    mask = ys < rs
    grad[3 * idx[mask] + 1] += 2.0 * (ys[mask] - rs[mask])
    grad[3 * idx[mask] + 2] += 2.0 * (rs[mask] - ys[mask])
    mask = ys > h - rs
    grad[3 * idx[mask] + 1] += 2.0 * (ys[mask] - h + rs[mask])
    grad[3 * idx[mask] + 2] += -2.0 * (ys[mask] - h + rs[mask])

    return grad


def _delaunay_edges(xs, ys):
    """Return list of (i, j) edges from Delaunay triangulation of centers."""
    points = np.column_stack((xs, ys))
    try:
        tri = Delaunay(points)
    except Exception:
        return None
    edges = set()
    for simplex in tri.simplices:
        edges.add(tuple(sorted((simplex[0], simplex[1]))))
        edges.add(tuple(sorted((simplex[1], simplex[2]))))
        edges.add(tuple(sorted((simplex[0], simplex[2]))))
    return list(edges)


def _solve_lp_radii(xs, ys, w, h):
    """
    Solve LP for per-circle radii given fixed centers and rectangle.
    Maximize sum(r) s.t. r_i + r_j <= d_ij (Delaunay edges)
                          r_i <= min(x_i, w-x_i, y_i, h-y_i)
    """
    edges = _delaunay_edges(xs, ys)
    if edges is None:
        return None
    
    # Distance for each edge
    d = np.zeros(len(edges))
    for k, (i, j) in enumerate(edges):
        d[k] = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
    
    # Boundary limits
    r_max = np.minimum(np.minimum(xs, w - xs), np.minimum(ys, h - ys))
    r_max = np.maximum(r_max, 1e-8)
    
    # LP: maximize sum(r)
    c = -np.ones(N)
    
    # Inequality constraints: A_ub @ r <= b_ub
    A_rows = []
    b_vals = []
    for k, (i, j) in enumerate(edges):
        row = np.zeros(N)
        row[i] = 1.0
        row[j] = 1.0
        A_rows.append(row)
        b_vals.append(d[k])
    
    # Boundary: r_i <= r_max_i
    for i in range(N):
        row = np.zeros(N)
        row[i] = 1.0
        A_rows.append(row)
        b_vals.append(r_max[i])
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    
    bounds = [(1e-8, None)] * N
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            return None
    except Exception:
        return None


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
    current_cost = _objective(x0, w, h, beta=10.0)
    
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
        
        x_temp = np.zeros(3*N)
        x_temp[0::3] = xs
        x_temp[1::3] = ys
        x_temp[2::3] = rs
        
        new_cost = _objective(x_temp, w, h, beta=10.0)
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
    # Choose grid dimensions that best fit 21 circles
    # Try 5x5 (25 cells, use 21), 7x3 (21 cells), 4x6 (24 cells)
    best_a, best_b = 5, 5
    for a in range(3, 9):
        for b in range(3, 9):
            if a * b >= N:
                # Prefer aspect ratio close to w/h
                cell_aspect = (w / a) / (h / b)
                target_aspect = w / h
                score = abs(cell_aspect - target_aspect)
                if a * b == N:
                    score -= 0.1  # bonus for exact fit
                if best_a is None or score < best_a:
                    best_a = score
                    best_b = (a, b)
    
    if best_b is None:
        a, b = 5, 5
    else:
        a, b = best_b
    
    x0 = np.zeros(3 * N)
    for i in range(N):
        col = i % a
        row = i // a
        x0[3 * i] = (col + 0.5) * w / a
        x0[3 * i + 1] = (row + 0.5) * h / b
        x0[3 * i + 2] = min(w / a, h / b) * 0.45
    
    return x0


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


def _joint_aspect_optimize(xs, ys, rs, beta=10.0, max_iter=300, tol=1e-10, deadline=None):
    """
    Jointly optimize circle positions/radii AND rectangle aspect ratio.
    This is the key improvement: the aspect ratio is co-optimized with positions.
    """
    w = 1.0
    h = 1.0
    xs = xs.copy()
    ys = ys.copy()
    rs = rs.copy()
    r_max = min(w, h) / 2.0

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        bw = max_x - min_x
        bh = max_y - min_y
        if bw + bh < 1e-6:
            break
        s = 4.0 / (2.0 * (bw + bh))
        w_new = bw * s
        h_new = bh * s
        if abs(w_new - w) < tol and abs(h_new - h) < tol:
            w, h = w_new, h_new
            break
        w, h = w_new, h_new
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        xs = (xs - cx) * s + 0.5 * w
        ys = (ys - cy) * s + 0.5 * h
        rs = rs * s
        r_max = min(w, h) / 2.0
        params = np.zeros(3 * N)
        params[0::3] = xs
        params[1::3] = ys
        params[2::3] = rs
        bounds = []
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-4, r_max))
        try:
            res = minimize(lambda p: _objective(p, w, h, beta=beta),
                           lambda p: _objective_grad(p, w, h, beta=beta),
                           params, method='L-BFGS-B', bounds=bounds, jac=True,
                           options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-12})
            xs = res.x[0::3]
            ys = res.x[1::3]
            rs = res.x[2::3]
        except Exception:
            pass

    return w, h, xs, ys, rs


def _joint_aspect_with_lp(xs, ys, rs, beta=50.0, max_iter=200, deadline=None):
    """
    Joint aspect optimization with LP radii refinement at each iteration.
    This combines the aspect ratio adjustment with exact LP radius solving.
    """
    w = 1.0
    h = 1.0
    xs = xs.copy()
    ys = ys.copy()
    rs = rs.copy()

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        bw = max_x - min_x
        bh = max_y - min_y
        if bw + bh < 1e-6:
            break
        s = 4.0 / (2.0 * (bw + bh))
        w_new = bw * s
        h_new = bh * s
        if abs(w_new - w) < 1e-10 and abs(h_new - h) < 1e-10:
            w, h = w_new, h_new
            break
        w, h = w_new, h_new
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        xs = (xs - cx) * s + 0.5 * w
        ys = (ys - cy) * s + 0.5 * h
        rs = rs * s
        
        # Solve LP for radii with current centers
        r_lp = _solve_lp_radii(xs, ys, w, h)
        if r_lp is not None:
            rs = r_lp.copy()
        
        # Optimize positions with fixed radii (from LP) and current aspect
        params = np.zeros(3 * N)
        params[0::3] = xs
        params[1::3] = ys
        params[2::3] = rs
        r_max = min(w, h) / 2.0
        bounds = []
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-4, r_max))
        try:
            res = minimize(lambda p: _objective(p, w, h, beta=beta),
                           lambda p: _objective_grad(p, w, h, beta=beta),
                           params, method='L-BFGS-B', bounds=bounds, jac=True,
                           options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-12})
            xs = res.x[0::3]
            ys = res.x[1::3]
            rs = res.x[2::3]
        except Exception:
            pass

    return w, h, xs, ys, rs


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    best_state = {'value': -np.inf, 'result': None}
    
    # Strategy: Try several initial aspect ratios, then use joint aspect optimization
    # to refine each, keeping track of the best result.
    
    # Initial aspect ratios to try (w values, with h = 2 - w)
    widths = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    
    best_x_for_warm = None
    best_w = 1.0
    best_h = 1.0
    
    for w_init in widths:
        if time.time() > deadline - 20.0:
            break
        h_init = 2.0 - w_init
        
        # Initialize with grid
        x0 = _grid_initial(w_init, h_init)
        
        # L-BFGS-B optimization with smooth objective and analytical gradient
        bounds = []
        r_max = min(w_init, h_init) / 2.0
        for i in range(N):
            bounds.append((0, w_init))
            bounds.append((0, h_init))
            bounds.append((1e-4, r_max))
        
        try:
            # First pass with moderate beta
            res = minimize(lambda p: _objective(p, w_init, h_init, beta=5.0),
                           lambda p: _objective_grad(p, w_init, h_init, beta=5.0),
                           x0, method='L-BFGS-B', bounds=bounds, jac=True,
                           options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-10})
            if time.time() > deadline - 15.0:
                break
            _check_and_update(res.x, w_init, h_init, best_state)
            
            # Second pass with higher beta for more precise optimization
            if time.time() < deadline - 12.0:
                res2 = minimize(lambda p: _objective(p, w_init, h_init, beta=20.0),
                                lambda p: _objective_grad(p, w_init, h_init, beta=20.0),
                                res.x, method='L-BFGS-B', bounds=bounds, jac=True,
                                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
                _check_and_update(res2.x, w_init, h_init, best_state)
                if best_x_for_warm is None or np.sum(res2.x[2::3]) > np.sum(best_x_for_warm[2::3]):
                    best_x_for_warm = res2.x.copy()
                    best_w = w_init
                    best_h = h_init
        except Exception:
            pass
        
        # Joint aspect optimization to refine aspect ratio
        if time.time() < deadline - 15.0 and best_x_for_warm is not None:
            try:
                xs_j = best_x_for_warm[0::3].copy()
                ys_j = best_x_for_warm[1::3].copy()
                rs_j = best_x_for_warm[2::3].copy()
                
                w_j, h_j, xs_j, ys_j, rs_j = _joint_aspect_optimize(
                    xs_j, ys_j, rs_j, beta=20.0, max_iter=100, deadline=deadline - 10.0)
                
                res_j = np.zeros((N, 3))
                res_j[:, 0] = xs_j
                res_j[:, 1] = ys_j
                res_j[:, 2] = rs_j
                _check_and_update(res_j, w_j, h_j, best_state)
                
                # Update warm start if improved
                if np.sum(rs_j) > np.sum(best_x_for_warm[2::3]):
                    x_warm = np.zeros(3 * N)
                    x_warm[0::3] = xs_j
                    x_warm[1::3] = ys_j
                    x_warm[2::3] = rs_j
                    best_x_for_warm = x_warm
                    best_w = w_j
                    best_h = h_j
            except Exception:
                pass
        
        # Delaunay LP radii refinement
        if time.time() < deadline - 12.0 and best_x_for_warm is not None:
            try:
                xs_lp = best_x_for_warm[0::3].copy()
                ys_lp = best_x_for_warm[1::3].copy()
                xs_lp = np.clip(xs_lp, 1e-4, best_w - 1e-4)
                ys_lp = np.clip(ys_lp, 1e-4, best_h - 1e-4)
                r_lp = _solve_lp_radii(xs_lp, ys_lp, best_w, best_h)
                if r_lp is not None:
                    x_lp = np.zeros(3 * N)
                    x_lp[0::3] = xs_lp
                    x_lp[1::3] = ys_lp
                    x_lp[2::3] = r_lp
                    _check_and_update(x_lp, best_w, best_h, best_state)
                    if np.sum(r_lp) > np.sum(best_x_for_warm[2::3]):
                        best_x_for_warm = x_lp.copy()
            except Exception:
                pass
        
        # Simulated annealing refinement
        if time.time() < deadline - 10.0:
            x_sa_start = best_x_for_warm if best_x_for_warm is not None else x0
            x_sa = _sa_optimize(best_w, best_h, x_sa_start, max_iters=1500, seed=42, deadline=deadline - 8.0)
            _check_and_update(x_sa, best_w, best_h, best_state)
            
            # Second SA pass with different seed
            if time.time() < deadline - 6.0:
                rng_sa = np.random.default_rng(123)
                x_sa2 = x_sa.copy()
                for i in range(N):
                    x_sa2[3 * i] += rng_sa.normal(0, 0.01)
                    x_sa2[3 * i + 1] += rng_sa.normal(0, 0.01)
                    x_sa2[3 * i + 2] += rng_sa.normal(0, 0.005)
                x_sa2 = _sa_optimize(best_w, best_h, x_sa2, max_iters=1000, seed=123, deadline=deadline - 5.0)
                _check_and_update(x_sa2, best_w, best_h, best_state)

    # Final high-precision polish pass with joint aspect optimization
    if time.time() < deadline - 5.0 and best_state['result'] is not None:
        xs = best_state['result'][:, 0]
        ys = best_state['result'][:, 1]
        rs = best_state['result'][:, 2]
        
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
            
            # High-precision joint aspect optimization with LP
            if time.time() < deadline - 3.0:
                try:
                    w_f, h_f, xs_f, ys_f, rs_f = _joint_aspect_with_lp(
                        xs_polish[0::3], xs_polish[1::3], xs_polish[2::3],
                        beta=50.0, max_iter=150, deadline=deadline - 2.0)
                    
                    res_f = np.zeros((N, 3))
                    res_f[:, 0] = xs_f
                    res_f[:, 1] = ys_f
                    res_f[:, 2] = rs_f
                    
                    # Normalize to origin
                    min_x_f = np.min(xs_f - rs_f)
                    min_y_f = np.min(ys_f - rs_f)
                    res_f[:, 0] -= min_x_f
                    res_f[:, 1] -= min_y_f
                    
                    # Validate
                    ok = True
                    if np.any(rs_f < 1e-9) or np.any(res_f[:, 0] - rs_f < -1e-6) or np.any(res_f[:, 0] + rs_f > w_f + 1e-6):
                        ok = False
                    if np.any(res_f[:, 1] - rs_f < -1e-6) or np.any(res_f[:, 1] + rs_f > h_f + 1e-6):
                        ok = False
                    if ok:
                        for i in range(N):
                            for j in range(i + 1, N):
                                d = np.sqrt((res_f[i, 0] - res_f[j, 0]) ** 2 + (res_f[i, 1] - res_f[j, 1]) ** 2)
                                if d < rs_f[i] + rs_f[j] - 1e-7:
                                    ok = False
                                    break
                    if ok:
                        val_f = np.sum(rs_f)
                        if val_f > best_state['value']:
                            best_state['value'] = val_f
                            best_state['result'] = res_f.copy()
                except Exception:
                    pass
            
            # Standard high-precision polish
            if time.time() < deadline - 2.0:
                bounds_polish = []
                r_max_polish = min(w_polish, h_polish) / 2.0
                for i in range(N):
                    bounds_polish.append((0, w_polish))
                    bounds_polish.append((0, h_polish))
                    bounds_polish.append((1e-4, r_max_polish))
                
                try:
                    res_polish = minimize(lambda p: _objective(p, w_polish, h_polish, beta=50.0),
                                          lambda p: _objective_grad(p, w_polish, h_polish, beta=50.0),
                                          x_polish, method='L-BFGS-B', bounds=bounds_polish, jac=True,
                                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14})
                    _check_and_update(res_polish.x, w_polish, h_polish, best_state)
                except Exception:
                    pass
                
                # Delaunay LP on polished result
                if time.time() < deadline - 1.0:
                    try:
                        xs_f = res_polish.x[0::3].copy() if res_polish.success else x_polish[0::3].copy()
                        ys_f = res_polish.x[1::3].copy() if res_polish.success else x_polish[1::3].copy()
                        xs_f = np.clip(xs_f, 1e-4, w_polish - 1e-4)
                        ys_f = np.clip(ys_f, 1e-4, h_polish - 1e-4)
                        r_f = _solve_lp_radii(xs_f, ys_f, w_polish, h_polish)
                        if r_f is not None:
                            x_f = np.zeros(3 * N)
                            x_f[0::3] = xs_f
                            x_f[1::3] = ys_f
                            x_f[2::3] = r_f
                            _check_and_update(x_f, w_polish, h_polish, best_state)
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
