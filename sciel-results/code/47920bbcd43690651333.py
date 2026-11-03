"""Circle packing in unit square for N=26 circles."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def _validate_and_score(circles):
    """Check validity and compute score. circles: (N,3) array [x,y,r]."""
    if circles.shape != (N, 3):
        return None
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    if np.any(r <= 0):
        return None
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9) or np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return None
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx * dx + dy * dy)
    if np.any(dist < r[i_idx] + r[j_idx] - 1e-9):
        return None
    return float(np.sum(r))


def _softplus(v, temp=0.01):
    v_clipped = np.clip(v / temp, -50, 50)
    return temp * np.log1p(np.exp(v_clipped))


def _softplus_grad(v, temp=0.01):
    v_clipped = np.clip(v / temp, -50, 50)
    return 1.0 / (1.0 + np.exp(-v_clipped))


def _hex_initial(n=N, seed=0):
    """Generate a hexagonal initial configuration with non-overlapping circles."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    
    # Hexagonal packing: ~5 cols x 6 rows = 30, use 26
    cols = 5
    rows = 6
    dx = 1.0 / (cols + 0.5)
    dy = 1.0 / (rows + 0.5)
    r_base = min(dx, dy) * 0.48
    
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            x_offset = 0.5 * dx if row % 2 == 1 else 0
            cx = (col + 0.5 + x_offset) * dx
            cy = (row + 0.5) * dy
            cx += 0.005 * rng.randn()
            cy += 0.005 * rng.randn()
            cx = np.clip(cx, r_base, 1 - r_base)
            cy = np.clip(cy, r_base, 1 - r_base)
            circles[idx, 0] = cx
            circles[idx, 1] = cy
            circles[idx, 2] = r_base
            idx += 1
    
    return circles


def _corner_hex_initial(n=N, seed=0):
    """Hybrid: larger circles in corners, hex grid in center."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    
    # 4 corner circles
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for i, (cx, cy) in enumerate(corners):
        if i >= n:
            break
        circles[i, 0] = cx + 0.01 * rng.randn()
        circles[i, 1] = cy + 0.01 * rng.randn()
        circles[i, 2] = 0.13
    
    # Fill center with hex grid
    remaining = n - 4
    if remaining > 0:
        r_fill = 0.075
        dx = 2 * r_fill * 1.08
        dy = dx * np.sqrt(3) / 2
        x_start = 0.32
        y_start = 0.32
        idx = 4
        for row in range(5):
            for col in range(5):
                if idx >= n:
                    break
                cx = x_start + col * dx + (dx / 2 if row % 2 else 0)
                cy = y_start + row * dy
                cx = np.clip(cx, r_fill, 1 - r_fill)
                cy = np.clip(cy, r_fill, 1 - r_fill)
                circles[idx, 0] = cx + 0.003 * rng.randn()
                circles[idx, 1] = cy + 0.003 * rng.randn()
                circles[idx, 2] = r_fill
                idx += 1
    
    return circles


def _resolve_overlaps(circles, max_iter=500):
    """Iteratively resolve overlaps by pushing circles apart."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for it in range(max_iter):
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = r[i_idx] + r[j_idx] - dist
        
        if not np.any(overlap > 1e-9):
            break
        
        for k in range(len(i_idx)):
            if overlap[k] > 1e-9:
                i = i_idx[k]
                j = j_idx[k]
                if dist[k] < 1e-12:
                    dx_k = 1e-6
                    dy_k = 0
                else:
                    dx_k = dx[k] / dist[k]
                    dy_k = dy[k] / dist[k]
                
                push = overlap[k] * 0.5
                x[i] -= push * dx_k
                y[i] -= push * dy_k
                x[j] += push * dx_k
                y[j] += push * dy_k
                
                shrink = overlap[k] * 0.1
                r[i] = max(r[i] - shrink, 0.001)
                r[j] = max(r[j] - shrink, 0.001)
        
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))
    
    return np.column_stack([x, y, r])


def _get_delaunay_pairs(cx, cy, i_idx, j_idx):
    """Get Delaunay edge pairs."""
    points = np.column_stack([cx, cy])
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        if not edges:
            return i_idx, j_idx
        edge_list = sorted(edges)
        return np.array([e[0] for e in edge_list]), np.array([e[1] for e in edge_list])
    except Exception:
        return i_idx, j_idx


def _physics_relax(circles, time_limit=12, seed=0):
    """Physics-based force relaxation to resolve overlaps and find good positions."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    dt = 0.003
    decay = 0.99997
    no_improve = 0
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0
    
    for it in range(25000):
        if time.time() - start_time > time_limit:
            break
        
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        overlap = r[i_idx] + r[j_idx] - dist
        
        mask = overlap > -0.01
        if np.any(mask):
            force_mag = np.where(overlap[mask] > 0, 60.0 * overlap[mask], 5.0 * overlap[mask])
            fx[i_idx[mask]] -= force_mag * dx[mask] / dist[mask]
            fy[i_idx[mask]] -= force_mag * dy[mask] / dist[mask]
            fx[j_idx[mask]] += force_mag * dx[mask] / dist[mask]
            fy[j_idx[mask]] += force_mag * dy[mask] / dist[mask]
        
        fx += np.where(x < r, 250.0, 0.0) - np.where(1 - x < r, 250.0, 0.0)
        fy += np.where(y < r, 250.0, 0.0) - np.where(1 - y < r, 250.0, 0.0)
        
        if it % 150 == 75:
            fx += 0.008 * np.random.randn(n)
            fy += 0.008 * np.random.randn(n)
        
        x += dt * fx
        y += dt * fy
        
        x = np.clip(x, 0.001, 0.999)
        y = np.clip(y, 0.001, 0.999)
        
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))
        
        dt *= decay
        
        if it % 25 == 0:
            score = _validate_and_score(np.column_stack([x, y, r]))
            if score is not None and score > best_score + 1e-10:
                best_score = score
                best_circles = np.column_stack([x, y, r]).copy()
                no_improve = 0
            else:
                no_improve += 1
            if no_improve > 600:
                break
    
    return best_circles, best_score


def _delaunay_penalty_optimize(circles, time_limit=25, seed=0):
    """Optimize using Delaunay-based penalty objective with L-BFGS-B and analytic gradients.
    
    Key structural idea: use squared softplus penalty (like inspiration 1) with
    analytic gradient via np.add.at for efficient vectorized accumulation.
    Only penalizes Delaunay contact pairs for speed.
    """
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    params = np.concatenate([x, y, r])
    
    temp = 0.015
    penalty_weight = 150.0
    
    cached_di, cached_dj = _get_delaunay_pairs(x, y, i_idx, j_idx)
    
    def objective(p):
        if time.time() - start_time > time_limit:
            return 1e10
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        
        # Boundary violations (squared softplus)
        lx = rr - xx
        ux = xx + rr - 1.0
        ly = rr - yy
        uy = yy + rr - 1.0
        
        bv = (np.sum(_softplus(lx, temp)**2) + np.sum(_softplus(ux, temp)**2) +
              np.sum(_softplus(ly, temp)**2) + np.sum(_softplus(uy, temp)**2))
        
        # Pair violations: only Delaunay pairs
        di, dj = cached_di, cached_dj
        dx = xx[di] - xx[dj]
        dy = yy[di] - yy[dj]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        ov = _softplus(rr[di] + rr[dj] - dist, temp)
        pv = np.sum(ov**2)
        
        return -np.sum(rr) + penalty_weight * (bv + pv)
    
    def gradient(p):
        if time.time() - start_time > time_limit:
            return np.zeros_like(p)
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        # d(-sum r)/dr
        gr -= 1.0
        
        # Boundary gradients (squared softplus)
        lx = rr - xx
        ux = xx + rr - 1.0
        ly = rr - yy
        uy = yy + rr - 1.0
        
        sp_lx = _softplus(lx, temp)
        sg_lx = _softplus_grad(lx, temp)
        gx += -2.0 * penalty_weight * sp_lx * sg_lx
        gr += 2.0 * penalty_weight * sp_lx * sg_lx
        
        sp_ux = _softplus(ux, temp)
        sg_ux = _softplus_grad(ux, temp)
        gx += 2.0 * penalty_weight * sp_ux * sg_ux
        gr += 2.0 * penalty_weight * sp_ux * sg_ux
        
        sp_ly = _softplus(ly, temp)
        sg_ly = _softplus_grad(ly, temp)
        gy += -2.0 * penalty_weight * sp_ly * sg_ly
        gr += 2.0 * penalty_weight * sp_ly * sg_ly
        
        sp_uy = _softplus(uy, temp)
        sg_uy = _softplus_grad(uy, temp)
        gy += 2.0 * penalty_weight * sp_uy * sg_uy
        gr += 2.0 * penalty_weight * sp_uy * sg_uy
        
        # Pair gradients: only Delaunay pairs
        di, dj = cached_di, cached_dj
        dx = xx[di] - xx[dj]
        dy = yy[di] - yy[dj]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        overlap = rr[di] + rr[dj] - dist
        
        sp_ov = _softplus(overlap, temp)
        sg_ov = _softplus_grad(overlap, temp)
        factor = 2.0 * penalty_weight * sp_ov * sg_ov
        inv_dist = 1.0 / dist
        
        gx[di] -= factor * dx * inv_dist
        gx[dj] += factor * dx * inv_dist
        gy[di] -= factor * dy * inv_dist
        gy[dj] += factor * dy * inv_dist
        gr[di] += factor
        gr[dj] += factor
        
        return np.concatenate([gx, gy, gr])
    
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.001, 0.5)] * n
    
    best_params = params.copy()
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    if best_score is None:
        best_score = 0.0
    
    # Multi-restart with graph updates
    n_restarts = 3
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 4:
            break
        
        # Update Delaunay cache
        xx = params[:n]
        yy = params[n:2*n]
        cached_di, cached_dj = _get_delaunay_pairs(xx, yy, i_idx, j_idx)
        
        try:
            result = minimize(objective, params, method='L-BFGS-B', jac=gradient,
                              bounds=bounds, options={'maxiter': 600, 'maxfun': 2500,
                                                       'ftol': 1e-15, 'gtol': 1e-13,
                                                       'maxls': 50})
            params = result.x
        except Exception:
            pass
        
        # Validate and track best
        new_x = np.clip(params[:n], 0, 1)
        new_y = np.clip(params[n:2*n], 0, 1)
        new_r = np.maximum(params[2*n:3*n], 0.001)
        max_r = np.minimum(np.minimum(new_x, 1 - new_x), np.minimum(new_y, 1 - new_y))
        new_r = np.minimum(new_r, np.maximum(max_r, 0.001))
        
        new_circles = np.column_stack([new_x, new_y, new_r])
        new_score = _validate_and_score(new_circles)
        if new_score is not None and new_score > best_score:
            best_score = new_score
            best_params = params.copy()
        
        # Small perturbation for next restart
        if restart < n_restarts - 1:
            params = best_params.copy()
            rng_r = np.random.RandomState(restart * 137 + seed)
            k = rng_r.randint(3 * n)
            params[k] += 0.004 * rng_r.randn()
    
    new_x = np.clip(best_params[:n], 0, 1)
    new_y = np.clip(best_params[n:2*n], 0, 1)
    new_r = np.maximum(best_params[2*n:3*n], 0.001)
    max_r = np.minimum(np.minimum(new_x, 1 - new_x), np.minimum(new_y, 1 - new_y))
    new_r = np.minimum(new_r, np.maximum(max_r, 0.001))
    
    new_circles = np.column_stack([new_x, new_y, new_r])
    new_score = _validate_and_score(new_circles)
    input_score = _validate_and_score(circles)
    if input_score is None:
        input_score = 0.0
    if new_score is None:
        new_score = 0.0
    
    if new_score >= input_score:
        return new_circles, new_score
    else:
        return circles.copy(), input_score


def _full_penalty_optimize(circles, time_limit=20, seed=0):
    """Full pairwise penalty optimization with analytic gradients (no Delaunay restriction).
    
    This is the structural change: instead of restricting to Delaunay pairs,
    use ALL pairs with the squared-softplus penalty and np.add.at for
    efficient gradient accumulation. This avoids the Delaunay graph
    staleness problem and gives a more accurate gradient.
    """
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    params = np.concatenate([x, y, r])
    
    temp = 0.01
    penalty_weight = 100.0
    
    def objective(p):
        if time.time() - start_time > time_limit:
            return 1e10
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        
        lx = rr - xx
        ux = xx + rr - 1.0
        ly = rr - yy
        uy = yy + rr - 1.0
        
        bv = (np.sum(_softplus(lx, temp)**2) + np.sum(_softplus(ux, temp)**2) +
              np.sum(_softplus(ly, temp)**2) + np.sum(_softplus(uy, temp)**2))
        
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        ov = _softplus(rr[i_idx] + rr[j_idx] - dist, temp)
        pv = np.sum(ov**2)
        
        return -np.sum(rr) + penalty_weight * (bv + pv)
    
    def gradient(p):
        if time.time() - start_time > time_limit:
            return np.zeros_like(p)
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        gr -= 1.0
        
        lx = rr - xx
        ux = xx + rr - 1.0
        ly = rr - yy
        uy = yy + rr - 1.0
        
        sp_lx = _softplus(lx, temp)
        sg_lx = _softplus_grad(lx, temp)
        gx += -2.0 * penalty_weight * sp_lx * sg_lx
        gr += 2.0 * penalty_weight * sp_lx * sg_lx
        
        sp_ux = _softplus(ux, temp)
        sg_ux = _softplus_grad(ux, temp)
        gx += 2.0 * penalty_weight * sp_ux * sg_ux
        gr += 2.0 * penalty_weight * sp_ux * sg_ux
        
        sp_ly = _softplus(ly, temp)
        sg_ly = _softplus_grad(ly, temp)
        gy += -2.0 * penalty_weight * sp_ly * sg_ly
        gr += 2.0 * penalty_weight * sp_ly * sg_ly
        
        sp_uy = _softplus(uy, temp)
        sg_uy = _softplus_grad(uy, temp)
        gy += 2.0 * penalty_weight * sp_uy * sg_uy
        gr += 2.0 * penalty_weight * sp_uy * sg_uy
        
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        overlap = rr[i_idx] + rr[j_idx] - dist
        
        sp_ov = _softplus(overlap, temp)
        sg_ov = _softplus_grad(overlap, temp)
        factor = 2.0 * penalty_weight * sp_ov * sg_ov
        inv_dist = 1.0 / dist
        
        np.add.at(gx, i_idx, -factor * dx * inv_dist)
        np.add.at(gx, j_idx, factor * dx * inv_dist)
        np.add.at(gy, i_idx, -factor * dy * inv_dist)
        np.add.at(gy, j_idx, factor * dy * inv_dist)
        np.add.at(gr, i_idx, factor)
        np.add.at(gr, j_idx, factor)
        
        return np.concatenate([gx, gy, gr])
    
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.001, 0.5)] * n
    
    best_params = params.copy()
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    if best_score is None:
        best_score = 0.0
    
    n_restarts = 2
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 4:
            break
        
        try:
            result = minimize(objective, params, method='L-BFGS-B', jac=gradient,
                              bounds=bounds, options={'maxiter': 800, 'maxfun': 3000,
                                                       'ftol': 1e-15, 'gtol': 1e-13,
                                                       'maxls': 50})
            params = result.x
        except Exception:
            pass
        
        new_x = np.clip(params[:n], 0, 1)
        new_y = np.clip(params[n:2*n], 0, 1)
        new_r = np.maximum(params[2*n:3*n], 0.001)
        max_r = np.minimum(np.minimum(new_x, 1 - new_x), np.minimum(new_y, 1 - new_y))
        new_r = np.minimum(new_r, np.maximum(max_r, 0.001))
        
        new_circles = np.column_stack([new_x, new_y, new_r])
        new_score = _validate_and_score(new_circles)
        if new_score is not None and new_score > best_score:
            best_score = new_score
            best_params = params.copy()
        
        if restart < n_restarts - 1:
            params = best_params.copy()
            rng_r = np.random.RandomState(restart * 211 + seed)
            k = rng_r.randint(3 * n)
            params[k] += 0.003 * rng_r.randn()
    
    new_x = np.clip(best_params[:n], 0, 1)
    new_y = np.clip(best_params[n:2*n], 0, 1)
    new_r = np.maximum(best_params[2*n:3*n], 0.001)
    max_r = np.minimum(np.minimum(new_x, 1 - new_x), np.minimum(new_y, 1 - new_y))
    new_r = np.minimum(new_r, np.maximum(max_r, 0.001))
    
    new_circles = np.column_stack([new_x, new_y, new_r])
    new_score = _validate_and_score(new_circles)
    input_score = _validate_and_score(circles)
    if input_score is None:
        input_score = 0.0
    if new_score is None:
        new_score = 0.0
    
    if new_score >= input_score:
        return new_circles, new_score
    else:
        return circles.copy(), input_score


def _simulated_annealing(circles, time_limit=25, seed=0):
    """Simulated annealing with penalty-based scoring."""
    rng = np.random.RandomState(seed)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def _penalty_score(cx, cy, cr, pw=100.0):
        if np.any(cr <= 0):
            return -1e6
        bv = np.maximum(0, cr - cx) + np.maximum(0, cr - (1 - cx)) + \
              np.maximum(0, cr - cy) + np.maximum(0, cr - (1 - cy))
        dx = cx[i_idx] - cx[j_idx]
        dy = cy[i_idx] - cy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        ov = np.maximum(0, cr[i_idx] + cr[j_idx] - dist)
        return float(np.sum(cr) - pw * (np.sum(bv) + np.sum(ov)))
    
    def _strict_score(cx, cy, cr):
        if np.any(cr <= 0):
            return None
        if np.any(cx - cr < -1e-9) or np.any(cx + cr > 1 + 1e-9) or \
           np.any(cy - cr < -1e-9) or np.any(cy + cr > 1 + 1e-9):
            return None
        dx = cx[i_idx] - cx[j_idx]
        dy = cy[i_idx] - cy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        if np.any(dist < cr[i_idx] + cr[j_idx] - 1e-9):
            return None
        return float(np.sum(cr))
    
    best_strict_score = _strict_score(x, y, r)
    best_circles = np.column_stack([x, y, r]).copy()
    if best_strict_score is None:
        best_strict_score = 0.0
    
    current_ps = _penalty_score(x, y, r)
    
    t_init = 0.05
    t_min = 0.0003
    max_iter = 50000
    cooling = (t_min / t_init) ** (1.0 / max_iter)
    t = t_init
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        i = rng.randint(n)
        move_type = rng.rand()
        
        new_x = x.copy()
        new_y = y.copy()
        new_r = r.copy()
        
        if move_type < 0.45:
            delta = 0.012 * rng.randn(2)
            new_x[i] += delta[0]
            new_y[i] += delta[1]
        elif move_type < 0.75:
            new_r[i] *= (1 + 0.06 * rng.randn())
            new_r[i] = max(0.005, min(new_r[i], 0.5))
        elif move_type < 0.88:
            j = rng.randint(n)
            if j != i:
                new_r[i], new_r[j] = new_r[j], new_r[i]
        else:
            j = rng.randint(n)
            if j != i:
                delta = 0.004 * rng.randn()
                new_r[i] += delta
                new_r[j] -= delta
                new_r[i] = max(0.005, new_r[i])
                new_r[j] = max(0.005, new_r[j])
        
        new_ps = _penalty_score(new_x, new_y, new_r)
        delta = new_ps - current_ps
        if delta > 0 or rng.rand() < np.exp(delta / max(t, 1e-10)):
            x, y, r = new_x, new_y, new_r
            current_ps = new_ps
            
            if it % 50 == 0:
                s = _strict_score(x, y, r)
                if s is not None and s > best_strict_score:
                    best_strict_score = s
                    best_circles = np.column_stack([x, y, r]).copy()
        
        t *= cooling
    
    s = _strict_score(x, y, r)
    if s is not None and s > best_strict_score:
        best_strict_score = s
        best_circles = np.column_stack([x, y, r]).copy()
    
    return best_circles, best_strict_score


def _high_precision_polish(circles, time_limit=15):
    """Final high-precision polish using very tight tolerances."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    params = np.concatenate([x, y, r])
    
    temp = 0.004
    penalty_weight = 250.0
    
    def objective(p):
        if time.time() - start_time > time_limit:
            return 1e10
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        
        lx = rr - xx
        ux = xx + rr - 1.0
        ly = rr - yy
        uy = yy + rr - 1.0
        
        bv = (np.sum(_softplus(lx, temp)**2) + np.sum(_softplus(ux, temp)**2) +
              np.sum(_softplus(ly, temp)**2) + np.sum(_softplus(uy, temp)**2))
        
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        ov = _softplus(rr[i_idx] + rr[j_idx] - dist, temp)
        pv = np.sum(ov**2)
        
        return -np.sum(rr) + penalty_weight * (bv + pv)
    
    def gradient(p):
        if time.time() - start_time > time_limit:
            return np.zeros_like(p)
        xx = p[:n]
        yy = p[n:2*n]
        rr = p[2*n:3*n]
        
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        gr -= 1.0
        
        lx = rr - xx
        ux = xx + rr - 1.0
        ly = rr - yy
        uy = yy + rr - 1.0
        
        sp_lx = _softplus(lx, temp)
        sg_lx = _softplus_grad(lx, temp)
        gx += -2.0 * penalty_weight * sp_lx * sg_lx
        gr += 2.0 * penalty_weight * sp_lx * sg_lx
        
        sp_ux = _softplus(ux, temp)
        sg_ux = _softplus_grad(ux, temp)
        gx += 2.0 * penalty_weight * sp_ux * sg_ux
        gr += 2.0 * penalty_weight * sp_ux * sg_ux
        
        sp_ly = _softplus(ly, temp)
        sg_ly = _softplus_grad(ly, temp)
        gy += -2.0 * penalty_weight * sp_ly * sg_ly
        gr += 2.0 * penalty_weight * sp_ly * sg_ly
        
        sp_uy = _softplus(uy, temp)
        sg_uy = _softplus_grad(uy, temp)
        gy += 2.0 * penalty_weight * sp_uy * sg_uy
        gr += 2.0 * penalty_weight * sp_uy * sg_uy
        
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        overlap = rr[i_idx] + rr[j_idx] - dist
        
        sp_ov = _softplus(overlap, temp)
        sg_ov = _softplus_grad(overlap, temp)
        factor = 2.0 * penalty_weight * sp_ov * sg_ov
        inv_dist = 1.0 / dist
        
        np.add.at(gx, i_idx, -factor * dx * inv_dist)
        np.add.at(gx, j_idx, factor * dx * inv_dist)
        np.add.at(gy, i_idx, -factor * dy * inv_dist)
        np.add.at(gy, j_idx, factor * dy * inv_dist)
        np.add.at(gr, i_idx, factor)
        np.add.at(gr, j_idx, factor)
        
        return np.concatenate([gx, gy, gr])
    
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.001, 0.5)] * n
    
    try:
        result = minimize(objective, params, method='L-BFGS-B', jac=gradient,
                          bounds=bounds, options={'maxiter': 1500, 'maxfun': 8000,
                                                   'ftol': 1e-16, 'gtol': 1e-15,
                                                   'maxls': 100})
    except Exception:
        return circles.copy(), _validate_and_score(circles) or 0.0
    
    if time.time() - start_time > time_limit:
        return circles.copy(), _validate_and_score(circles) or 0.0
    
    new_params = result.x
    new_x = np.clip(new_params[:n], 0, 1)
    new_y = np.clip(new_params[n:2*n], 0, 1)
    new_r = np.maximum(new_params[2*n:3*n], 0.001)
    max_r = np.minimum(np.minimum(new_x, 1 - new_x), np.minimum(new_y, 1 - new_y))
    new_r = np.minimum(new_r, np.maximum(max_r, 0.001))
    
    new_circles = np.column_stack([new_x, new_y, new_r])
    new_score = _validate_and_score(new_circles)
    input_score = _validate_and_score(circles)
    if input_score is None:
        input_score = 0.0
    if new_score is None:
        new_score = 0.0
    
    if new_score >= input_score:
        return new_circles, new_score
    else:
        return circles.copy(), input_score


def construct():
    """Construct optimal circle packing in unit square for N=26 circles."""
    start_time = time.time()
    time_budget = 140  # Keep 10s safety margin before 150s kill
    
    best_circles = None
    best_score = 0.0
    
    # Multiple diverse initializations
    initializations = [
        ('hex1', _hex_initial(N, seed=1)),
        ('hex2', _hex_initial(N, seed=7)),
        ('corner1', _corner_hex_initial(N, seed=3)),
        ('corner2', _corner_hex_initial(N, seed=11)),
    ]
    
    for name, init_circles in initializations:
        elapsed = time.time() - start_time
        if elapsed > time_budget - 40:
            break
        
        # Physics relaxation to resolve overlaps
        remaining = time_budget - (time.time() - start_time) - 35
        if remaining > 8:
            phys_circles, phys_score = _physics_relax(init_circles, time_limit=min(remaining * 0.25, 10), seed=hash(name) % 10000)
            if phys_score > best_score:
                best_score = phys_score
                best_circles = phys_circles
            resolved = phys_circles
        else:
            resolved = _resolve_overlaps(init_circles, max_iter=100)
        
        # Delaunay-based penalty optimization (fast)
        remaining = time_budget - (time.time() - start_time) - 30
        if remaining > 15:
            del_circles, del_score = _delaunay_penalty_optimize(
                resolved, time_limit=min(remaining * 0.4, 22), seed=hash(name) % 10000)
            if del_score > best_score:
                best_score = del_score
                best_circles = del_circles
        
        # Full pairwise penalty optimization (more accurate gradients)
        remaining = time_budget - (time.time() - start_time) - 25
        if remaining > 12:
            full_circles, full_score = _full_penalty_optimize(
                best_circles if best_circles is not None else resolved,
                time_limit=min(remaining * 0.45, 18), seed=hash(name) % 10000)
            if full_score > best_score:
                best_score = full_score
                best_circles = full_circles
        
        # Simulated annealing for escape from local optima
        remaining = time_budget - (time.time() - start_time) - 20
        if remaining > 15:
            sa_circles, sa_score = _simulated_annealing(
                best_circles if best_circles is not None else resolved,
                time_limit=min(remaining * 0.5, 22), seed=hash(name) % 10000)
            if sa_score > best_score:
                best_score = sa_score
                best_circles = sa_circles
    
    # Final polish phase
    if best_circles is not None:
        # Full pairwise optimization polish
        remaining = time_budget - (time.time() - start_time) - 25
        if remaining > 15:
            full_polish_circles, full_polish_score = _full_penalty_optimize(
                best_circles, time_limit=min(remaining * 0.5, 20), seed=9999)
            if full_polish_score > best_score:
                best_score = full_polish_score
                best_circles = full_polish_circles
        
        # Delaunay-based polish
        remaining = time_budget - (time.time() - start_time) - 18
        if remaining > 12:
            del_polish_circles, del_polish_score = _delaunay_penalty_optimize(
                best_circles, time_limit=min(remaining * 0.5, 16), seed=7777)
            if del_polish_score > best_score:
                best_score = del_polish_score
                best_circles = del_polish_circles
        
        # Simulated annealing polish
        remaining = time_budget - (time.time() - start_time) - 12
        if remaining > 12:
            sa_polish_circles, sa_polish_score = _simulated_annealing(
                best_circles, time_limit=min(remaining * 0.5, 15), seed=5555)
            if sa_polish_score > best_score:
                best_score = sa_polish_score
                best_circles = sa_polish_circles
        
        # High-precision final polish
        remaining = time_budget - (time.time() - start_time) - 5
        if remaining > 10:
            hp_circles, hp_score = _high_precision_polish(
                best_circles, time_limit=min(remaining * 0.8, 14))
            if hp_score > best_score:
                best_score = hp_score
                best_circles = hp_circles
    
    # Final validation
    if best_circles is None:
        return _hex_initial(N)
    
    final_score = _validate_and_score(best_circles)
    if final_score is None:
        return _hex_initial(N)
    
    return best_circles

# EVOLVE-BLOCK-END
