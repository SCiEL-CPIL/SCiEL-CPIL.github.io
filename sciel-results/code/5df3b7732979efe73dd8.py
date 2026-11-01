"""Circle packing in perimeter-4 rectangle: Multi-strategy optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _is_valid(cx, cy, r, W, H, tol=1e-7):
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

def _force_relax(x, y, r, w, h, max_iter=2000, time_budget=10.0):
    """Force-directed relaxation to resolve overlaps."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 1e-5
    rng = np.random.default_rng(42)
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if w < 0.2 or h < 0.2:
            break
            
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall repulsion
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
        
        max_overlap = np.max(overlap) if len(overlap) > 0 else 0.0
        
        if max_overlap > 1e-8:
            f_pairs = overlap * 50.0
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
        wall_viol = np.max(np.abs(left_viol)) + np.max(np.abs(right_viol)) + \
                    np.max(np.abs(bottom_viol)) + np.max(np.abs(top_viol))
        if max_overlap < 1e-8 and wall_viol < 1e-8:
            break
            
        # Update positions
        x += dt * fx
        y += dt * fy
        
        # Jitter periodically
        if it % 100 == 0 and it > 0:
            x += rng.normal(0, 0.0005, n)
            y += rng.normal(0, 0.0005, n)
            
        # Clamp positions
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        
        # Adaptive step size
        force_mag = np.max(np.sqrt(fx*fx + fy*fy))
        if force_mag > 2.0:
            dt = max(min_dt, dt * 0.5)
        else:
            dt = min(0.02, dt * 1.05)
            
    return x, y, r

def _expand_radii(x, y, r, w, h, max_iter=500):
    """Iteratively expand radii as much as possible."""
    r = r.copy()
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            r_max = min(x[i], w - x[i], y[i], h - y[i])
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(r_max, 0.0)
            
            if r_max > r[i] + 1e-9:
                r[i] = r_max
                improved = True
        if not improved:
            break
    return r

def _slsqp_optimize(x, y, r, w, h, time_budget=15.0):
    """SLSQP optimization to maximize sum of radii."""
    n = N
    z = np.zeros(3*n + 1)
    z[0::3][:n] = x
    z[1::3][:n] = y
    z[2::3][:n] = r
    z[-1] = w
    
    def objective(zv):
        rv = zv[2::3][:n]
        return -np.sum(rv)
        
    constraints = []
    bounds = []
    
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))
    
    # Boundary constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3*i+1] - p[3*i+2]})
        
    # Overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2 - (p[3*i+2]+p[3*j+2])**2
            })
            
    try:
        res = minimize(objective, z, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 300, 'ftol': 1e-12, 'maxls': 50})
        if res.success:
            x_new = res.x[0::3][:n]
            y_new = res.x[1::3][:n]
            r_new = res.x[2::3][:n]
            w_new = res.x[-1]
            h_new = 2.0 - w_new
            if _is_valid(x_new, y_new, r_new, w_new, h_new):
                return x_new, y_new, r_new, w_new, h_new
    except Exception:
        pass
    return x, y, r, w, h

def _grid_init(w, h, rows, cols, rng, radius_frac=0.95):
    """Initialize circles in a grid layout."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    total_cells = rows * cols
    if total_cells < N:
        while rows * cols < N:
            cols += 1
        total_cells = rows * cols
        
    for i in range(N):
        row = i // cols
        col = i % cols
        
        x_min = col * (w / cols)
        x_max = (col + 1) * (w / cols)
        y_min = row * (h / rows)
        y_max = (row + 1) * (h / rows)
        
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        
        cell_w = w / cols
        cell_h = h / rows
        max_r = min(cell_w, cell_h) / 2.0
        r0 = max_r * radius_frac
        
        jitter_x = rng.uniform(-0.005, 0.005)
        jitter_y = rng.uniform(-0.005, 0.005)
        
        cx = np.clip(cx + jitter_x, x_min + r0, x_max - r0)
        cy = np.clip(cy + jitter_y, y_min + r0, y_max - r0)
        
        x[i] = cx
        y[i] = cy
        r[i] = r0
        
    return x, y, r

def _hex_init(w, h, rng):
    """Initialize circles in a hexagonal pattern."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    # Hexagonal packing: rows offset
    # Estimate grid dimensions
    side = int(np.ceil(np.sqrt(N * 1.2)))
    rows = side
    cols = side
    
    for i in range(N):
        row = i // cols
        col = i % cols
        
        if row % 2 == 0:
            cx = (col + 0.5) * (w / cols)
        else:
            cx = (col + 1.0) * (w / cols)
            if col == cols - 1:
                cx = (col - 0.5) * (w / cols)
                
        cy = (row + 0.5) * (h / rows)
        
        # Estimate max radius based on local density
        spacing_x = w / cols
        spacing_y = h / rows
        max_r = min(spacing_x, spacing_y) / 2.0 * 0.9
        
        # Clamp to stay in bounds
        cx = np.clip(cx, max_r, w - max_r)
        cy = np.clip(cy, max_r, h - max_r)
        
        x[i] = cx
        y[i] = cy
        r[i] = max_r
        
    return x, y, r

def _random_init(w, h, rng):
    """Random initialization with small radii."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    min_r = 0.02
    max_r = 0.15
    
    for i in range(N):
        x[i] = rng.uniform(min_r, w - min_r)
        y[i] = rng.uniform(min_r, h - min_r)
        r[i] = rng.uniform(min_r, max_r)
        
    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_value = 0.0
    best_result = None
    
    rng = np.random.default_rng(42)
    
    # Different layouts to try
    layouts = [
        (5, 5),
        (4, 6),
        (6, 4),
        (3, 7),
        (7, 3),
        (5, 4),
        (4, 5),
        (2, 11),
        (11, 2),
        (3, 8),
        (8, 3),
        (4, 7),
        (7, 4),
    ]
    
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.4, 0.7, 1.6, 0.65, 1.3, 0.85]
    
    # Strategy 1: Grid-based initialization
    for w0 in aspect_ratios:
        if time.time() > deadline - 40:
            break
            
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
            
        for rows, cols in layouts:
            if time.time() > deadline - 30:
                break
                
            try:
                x, y, r = _grid_init(w0, h0, rows, cols, rng)
                
                # Force relax
                x, y, r = _force_relax(x, y, r, w0, h0, max_iter=1500, time_budget=5.0)
                
                # Expand radii
                r = _expand_radii(x, y, r, w0, h0, max_iter=300)
                
                # SLSQP refine
                x, y, r, w, h = _slsqp_optimize(x, y, r, w0, h0, time_budget=10.0)
                
                if _is_valid(x, y, r, w, h):
                    val = np.sum(r)
                    if val > best_value:
                        best_value = val
                        best_result = (x.copy(), y.copy(), r.copy(), w, h)
                        
            except Exception:
                continue
                
    # Strategy 2: Hexagonal initialization
    if time.time() < deadline - 50:
        for w0 in [1.0, 0.9, 1.1, 0.85, 1.15]:
            if time.time() > deadline - 40:
                break
                
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
                
            try:
                x, y, r = _hex_init(w0, h0, rng)
                
                x, y, r = _force_relax(x, y, r, w0, h0, max_iter=1500, time_budget=5.0)
                r = _expand_radii(x, y, r, w0, h0, max_iter=300)
                x, y, r, w, h = _slsqp_optimize(x, y, r, w0, h0, time_budget=10.0)
                
                if _is_valid(x, y, r, w, h):
                    val = np.sum(r)
                    if val > best_value:
                        best_value = val
                        best_result = (x.copy(), y.copy(), r.copy(), w, h)
                        
            except Exception:
                continue
                
    # Strategy 3: Random initialization with growth
    if time.time() < deadline - 40:
        for w0 in [1.0, 0.95, 1.05]:
            if time.time() > deadline - 30:
                break
                
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
                
            for attempt in range(3):
                if time.time() > deadline - 25:
                    break
                    
                try:
                    x, y, r = _random_init(w0, h0, rng)
                    
                    # Grow radii iteratively
                    for grow_iter in range(20):
                        x, y, r = _force_relax(x, y, r, w0, h0, max_iter=500, time_budget=3.0)
                        r = _expand_radii(x, y, r, w0, h0, max_iter=200)
                        if time.time() > deadline - 20:
                            break
                            
                    x, y, r, w, h = _slsqp_optimize(x, y, r, w0, h0, time_budget=8.0)
                    
                    if _is_valid(x, y, r, w, h):
                        val = np.sum(r)
                        if val > best_value:
                            best_value = val
                            best_result = (x.copy(), y.copy(), r.copy(), w, h)
                            
                except Exception:
                    continue

    # Fallback: simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side * 0.9
        best_result = (x, y, r, w, h)

    x, y, r, w, h = best_result
    return np.column_stack((x, y, r))
