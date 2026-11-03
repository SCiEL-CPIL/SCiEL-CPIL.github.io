"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Strategy:
  - Multi-start from hexagonal / grid / random / farthest-point initializations
  - Delaunay-based LP for per-circle radii (exact feasibility on local constraints)
  - LP-dual (shadow price) driven center optimization: use duals from linprog to
    compute forces on centers, then re-solve LP. This is the structural core.
  - Final SLSQP polish to refine the best LP solution.
  - All within a 145 s safety deadline
"""
import numpy as np
import time
import random
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay

N = 21

def _softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(x)))

def _sigmoid(x):
    return np.where(x > 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

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
    Maximize sum(r) s.t. r_i + r_j <= d_ij (Delaunay edges), r_i <= boundary, r_i >= 0.
    Returns (radii, duals) where duals is a list of (kind, i, j, value).
    """
    limits = np.minimum(np.minimum(xs, w - xs), np.minimum(ys, h - ys))
    if np.any(limits < 0):
        return np.zeros(N), []

    A_list = []
    b_list = []
    edges = _delaunay_edges(xs, ys)
    pair_idx = []
    if edges is not None:
        for i, j in edges:
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_list.append(row)
            b_list.append(dist)
            pair_idx.append((i, j))

    A_bound = np.eye(N)
    if A_list:
        A_ub = np.vstack([A_list, A_bound])
        b_ub = np.concatenate([b_list, limits])
    else:
        A_ub = A_bound
        b_ub = limits

    c = -np.ones(N)
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if res.success:
            radii = np.maximum(0.0, res.x)
            duals = []
            # Extract marginals (dual variables)
            # res.ineqlin.marginals corresponds to A_ub rows
            if hasattr(res, 'ineqlin') and res.ineqlin is not None and hasattr(res.ineqlin, 'marginals'):
                marg = res.ineqlin.marginals
                for k, (i, j) in enumerate(pair_idx):
                    if k < len(marg):
                        duals.append(('pair', i, j, abs(marg[k])))
                offset = len(pair_idx)
                for i in range(N):
                    if offset + i < len(marg):
                        duals.append(('bound', i, -1, abs(marg[offset + i])))
            return radii, duals
    except Exception:
        pass
    return np.maximum(0.0, limits * 0.9), []

def _make_hex(w, h, rows):
    """Hexagonal layout with given per-row counts, scaled to w,h."""
    pts = []
    for k in range(rows):
        n_in_row = N // rows + (1 if k < N % rows else 0)
        y = k * np.sqrt(3.0)
        off = 1.0 if k % 2 == 1 else 0.0
        for i in range(n_in_row):
            pts.append((i * 2.0 + off, y))
    if len(pts) < N:
        # Pad if needed
        while len(pts) < N:
            pts.append((len(pts) * 2.0, 0))
    pts = np.array(pts[:N])
    xs = pts[:, 0]
    ys = pts[:, 1]
    cx = 0.5 * (xs.max() + xs.min())
    cy = 0.5 * (ys.max() + ys.min())
    xs = xs - cx
    ys = ys - cy
    wt = xs.max() - xs.min()
    ht = ys.max() - ys.min()
    if wt == 0: wt = 1
    if ht == 0: ht = 1
    s = min(w / wt, h / ht) * 0.9 # 90% scale to allow movement
    xs_f = xs * s + 0.5 * w
    ys_f = ys * s + 0.5 * h
    return xs_f, ys_f

def _make_grid(w, h, rows, cols):
    pts = []
    for j in range(rows):
        for i in range(cols):
            if len(pts) >= N:
                break
            pts.append((i, j))
    pts = np.array(pts[:N])
    xs = pts[:, 0]
    ys = pts[:, 1]
    cx = 0.5 * (xs.max() + xs.min())
    cy = 0.5 * (ys.max() + ys.min())
    xs = xs - cx
    ys = ys - cy
    wt = xs.max() - xs.min()
    ht = ys.max() - ys.min()
    if wt == 0: wt = 1
    if ht == 0: ht = 1
    s = min(w / wt, h / ht) * 0.9
    xs_f = xs * s + 0.5 * w
    ys_f = ys * s + 0.5 * h
    return xs_f, ys_f

def _make_farthest_point(w, h, seed):
    """Farthest-point sampling initialization for better spread."""
    rng = np.random.RandomState(seed)
    xs = np.zeros(N)
    ys = np.zeros(N)
    xs[0] = rng.uniform(0.2, 0.8) * w
    ys[0] = rng.uniform(0.2, 0.8) * h
    for k in range(1, N):
        # Sample many candidates, pick the one farthest from existing points
        cand_x = rng.uniform(0.05, 0.95, 200) * w
        cand_y = rng.uniform(0.05, 0.95, 200) * h
        min_dists = np.full(200, np.inf)
        for j in range(k):
            d = np.sqrt((cand_x - xs[j]) ** 2 + (cand_y - ys[j]) ** 2)
            min_dists = np.minimum(min_dists, d)
        # Also consider distance to boundaries
        bd = np.minimum(np.minimum(cand_x, w - cand_x), np.minimum(cand_y, h - cand_y))
        min_dists = np.minimum(min_dists, bd)
        best = np.argmax(min_dists)
        xs[k] = cand_x[best]
        ys[k] = cand_y[best]
    return xs, ys

def _dual_driven_optimize(xs, ys, w, h, max_iters=200, deadline=None):
    """
    LP-dual driven center optimization.
    """
    xs = xs.copy().astype(float)
    ys = ys.copy().astype(float)
    best_val = -np.inf
    best_xs, best_ys, best_rs = xs.copy(), ys.copy(), np.zeros(N)

    step = 0.05
    stall = 0
    best_stall_val = -np.inf

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 3.0:
            break

        radii, duals = _solve_lp_radii(xs, ys, w, h)
        val = float(np.sum(radii))

        if val > best_val:
            best_val = val
            best_xs = xs.copy()
            best_ys = ys.copy()
            best_rs = radii.copy()
            stall = 0
        else:
            stall += 1

        if stall > 50:
            # Shake
            xs += np.random.normal(0, 0.01, N)
            ys += np.random.normal(0, 0.01, N)
            xs = np.clip(xs, 1e-4, w - 1e-4)
            ys = np.clip(ys, 1e-4, h - 1e-4)
            step *= 0.8
            stall = 0
            continue

        # Compute forces from duals
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        for kind, i, j, dv in duals:
            if kind == 'pair':
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 1e-8:
                    fx[i] += dv * dx / dist
                    fy[i] += dv * dy / dist
                    fx[j] -= dv * dx / dist
                    fy[j] -= dv * dy / dist
            elif kind == 'bound':
                # dv is shadow price for boundary constraint r_i <= min(x_i, w-x_i, y_i, h-y_i)
                # We need to know WHICH boundary is active.
                # Check distances
                d_left = xs[i]
                d_right = w - xs[i]
                d_down = ys[i]
                d_up = h - ys[i]
                
                min_d = min(d_left, d_right, d_down, d_up)
                # If the active constraint is the one with min distance, push away from it
                # But we need to know which one the LP considered active.
                # Since limits = min(...), if d_left == min_d, it's active.
                if abs(d_left - min_d) < 1e-6:
                    fx[i] += dv
                if abs(d_right - min_d) < 1e-6:
                    fx[i] -= dv
                if abs(d_down - min_d) < 1e-6:
                    fy[i] += dv
                if abs(d_up - min_d) < 1e-6:
                    fy[i] -= dv

        # Normalize forces
        f_norm = np.sqrt(np.sum(fx**2) + np.sum(fy**2))
        if f_norm > 1e-8:
            fx = fx / f_norm
            fy = fy / f_norm

        # Update centers
        xs += step * fx
        ys += step * fy
        
        # Project back to bounds (centers must be inside)
        xs = np.clip(xs, 1e-4, w - 1e-4)
        ys = np.clip(ys, 1e-4, h - 1e-4)

    return best_xs, best_ys, best_rs, best_val

def _objective_variable_radius(params, w, h, W=100.0):
    """
    Objective for variable-radius packing: minimize -sum(ri) + penalties.
    params = [x1, y1, r1, x2, y2, r2, ...] (3*N)
    """
    n = N
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    obj = -np.sum(rs)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0
    
    # Boundary penalties
    for i in range(n):
        xi, yi, ri = xs[i], ys[i], rs[i]
        if ri < 1e-8:
            ri = 1e-8
        v = ri - xi
        if v > -20:
            s = _sigmoid(v)
            obj += W * _softplus(v)
            grad[3*i] -= W * s
            grad[3*i+2] += W * s
        v = xi + ri - w
        if v > -20:
            s = _sigmoid(v)
            obj += W * _softplus(v)
            grad[3*i] += W * s
            grad[3*i+2] += W * s
        v = ri - yi
        if v > -20:
            s = _sigmoid(v)
            obj += W * _softplus(v)
            grad[3*i+1] -= W * s
            grad[3*i+2] += W * s
        v = yi + ri - h
        if v > -20:
            s = _sigmoid(v)
            obj += W * _softplus(v)
            grad[3*i+1] += W * s
            grad[3*i+2] += W * s
    
    # Pairwise overlap
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            min_d = rs[i] + rs[j]
            if dist_sq < min_d * min_d:
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                v = min_d - dist
                if v > -20:
                    s = _sigmoid(v)
                    obj += W * _softplus(v)
                    coeff = W * s / dist
                    grad[3*i] -= coeff * dx
                    grad[3*i+1] -= coeff * dy
                    grad[3*i+2] += W * s
                    grad[3*j] += coeff * dx
                    grad[3*j+1] += coeff * dy
                    grad[3*j+2] += W * s
    
    return obj, grad

def _check_valid_var(xs, ys, rs, w, h, tol=1e-7):
    """Check validity of variable-radius configuration."""
    for i in range(N):
        if rs[i] < 1e-6:
            return False
        if xs[i] - rs[i] < -tol or xs[i] + rs[i] > w + tol:
            return False
        if ys[i] - rs[i] < -tol or ys[i] + rs[i] > h + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            if d < rs[i] + rs[j] - tol:
                return False
    return True

def _bbox_perim(xs, ys, rs):
    """Compute bounding box perimeter."""
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    return 2.0 * ((max_x - min_x) + (max_y - min_y))

def _scale_to_perim(xs, ys, rs, target_perim=4.0):
    """Scale configuration to target perimeter."""
    perim = _bbox_perim(xs, ys, rs)
    if perim <= target_perim + 1e-9:
        return xs, ys, rs, perim
    scale = target_perim / perim
    cx = np.mean(xs)
    cy = np.mean(ys)
    xs_new = cx + (xs - cx) * scale
    ys_new = cy + (ys - cy) * scale
    rs_new = rs * scale
    return xs_new, ys_new, rs_new, target_perim

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin

    random.seed(42)
    np.random.seed(42)
    
    best_value = -np.inf
    best_result = None

    # Try multiple aspect ratios
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2]

    for ar in aspect_ratios:
        if time.time() > deadline - 15:
            break
        
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 - w
        
        # Generate initializations
        inits = []
        
        # Grid
        for rows, cols in [(3, 7), (4, 6), (5, 5)]:
            if rows * cols >= N:
                xs, ys = _make_grid(w, h, rows, cols)
                inits.append((xs.copy(), ys.copy()))
        
        # Hex
        for rows in [3, 4, 5]:
            xs, ys = _make_hex(w, h, rows)
            inits.append((xs.copy(), ys.copy()))
        
        # Farthest Point
        for seed in [1, 2, 3]:
            xs, ys = _make_farthest_point(w, h, seed)
            inits.append((xs.copy(), ys.copy()))

        for xs_init, ys_init in inits:
            if time.time() > deadline - 10:
                break
            
            # 1. LP-Dual Optimization
            xs_opt, ys_opt, rs_opt, val_lp = _dual_driven_optimize(
                xs_init, ys_init, w, h, max_iters=150, deadline=deadline
            )
            
            # Check validity of LP result
            if _check_valid_var(xs_opt, ys_opt, rs_opt, w, h):
                if val_lp > best_value:
                    best_value = val_lp
                    best_result = np.column_stack([xs_opt, ys_opt, rs_opt])
            
            # 2. SLSQP Polish from LP result
            if time.time() < deadline - 5:
                try:
                    params = np.concatenate([xs_opt, ys_opt, rs_opt])
                    bounds = [(0, w)] * N + [(0, h)] * N + [(1e-4, 0.5)] * N
                    
                    res = minimize(
                        lambda p: _objective_variable_radius(p, w, h),
                        params,
                        jac=True,
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': 200, 'ftol': 1e-10}
                    )
                    
                    p_opt = res.x
                    xs_s = p_opt[0::3]
                    ys_s = p_opt[1::3]
                    rs_s = p_opt[2::3]
                    
                    if _check_valid_var(xs_s, ys_s, rs_s, w, h):
                        val_s = np.sum(rs_s)
                        if val_s > best_value:
                            best_value = val_s
                            best_result = np.column_stack([xs_s, ys_s, rs_s])
                except Exception:
                    pass

    # Final refinement: try to improve best_result with local search
    if best_result is not None and time.time() < deadline - 5:
        for _ in range(2):
            if time.time() > deadline - 3:
                break
            perturbed = best_result + np.random.randn(*best_result.shape) * 0.002
            # Clip to reasonable bounds
            w_cur = 2.0
            h_cur = 2.0 # We don't know exact w,h here, but we can re-estimate or just clip to 0..2
            perturbed[:, 0] = np.clip(perturbed[:, 0], 0, 2.0)
            perturbed[:, 1] = np.clip(perturbed[:, 1], 0, 2.0)
            perturbed[:, 2] = np.clip(perturbed[:, 2], 1e-4, 0.5)
            
            # Quick validity check and value
            # We need w,h for validity. Since best_result came from a specific w,h, 
            # we should ideally store w,h. For now, we'll assume the bounding box of the result
            # is close to the target, or we skip this if we can't verify strictly.
            # Let's just try to optimize it again with SLSQP if we have time.
            try:
                # Estimate w,h from bounding box of best_result
                min_x = np.min(best_result[:, 0] - best_result[:, 2])
                max_x = np.max(best_result[:, 0] + best_result[:, 2])
                min_y = np.min(best_result[:, 1] - best_result[:, 2])
                max_y = np.max(best_result[:, 1] + best_result[:, 2])
                w_est = max_x - min_x
                h_est = max_y - min_y
                
                if w_est > 0 and h_est > 0:
                    params = perturbed.flatten()
                    bounds = [(0, w_est)] * N + [(0, h_est)] * N + [(1e-4, 0.5)] * N
                    res = minimize(
                        lambda p: _objective_variable_radius(p, w_est, h_est),
                        params,
                        jac=True,
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': 50, 'ftol': 1e-10}
                    )
                    p_opt = res.x
                    xs_s = p_opt[0::3]
                    ys_s = p_opt[1::3]
                    rs_s = p_opt[2::3]
                    
                    if _check_valid_var(xs_s, ys_s, rs_s, w_est, h_est):
                        val_s = np.sum(rs_s)
                        if val_s > best_value:
                            best_value = val_s
                            best_result = np.column_stack([xs_s, ys_s, rs_s])
            except Exception:
                pass

    # Return result
    if best_result is None:
        # Fallback: simple grid
        xs, ys = _make_grid(1.0, 1.0, 5, 5)
        r = 0.08
        best_result = np.column_stack([xs, ys, np.full(N, r)])
    
    # Scale to ensure perimeter <= 4
    xs_f = best_result[:, 0]
    ys_f = best_result[:, 1]
    rs_f = best_result[:, 2]
    
    perim = _bbox_perim(xs_f, ys_f, rs_f)
    if perim > 4.0 + 1e-9:
        scale = 4.0 / perim
        cx = np.mean(xs_f)
        cy = np.mean(ys_f)
        xs_f = cx + (xs_f - cx) * scale
        ys_f = cy + (ys_f - cy) * scale
        rs_f = rs_f * scale
    
    return np.column_stack([xs_f, ys_f, rs_f])
