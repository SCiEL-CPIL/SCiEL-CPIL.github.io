import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

N = 21

# EVOLVE-BLOCK-START

def _voronoi_inradii(centers, W, H):
    """
    Compute inradii for each circle center using Voronoi-like distance to boundaries and neighbors.
    For each center, the max radius is the min distance to:
    - rectangle boundaries
    - other circle centers (half the distance)
    """
    n = len(centers)
    r = np.zeros(n)
    for i in range(n):
        xi, yi = centers[i]
        # Distance to boundaries
        d_bound = min(xi, W - xi, yi, H - yi)
        # Distance to other centers
        d_min = d_bound
        for j in range(n):
            if i == j:
                continue
            dx = xi - centers[j][0]
            dy = yi - centers[j][1]
            d = math.sqrt(dx*dx + dy*dy)
            if d < d_min:
                d_min = d
        r[i] = 0.5 * d_min
    return r

def _compute_value_from_centers(centers, W):
    """
    Given centers and W (H=2-W), compute the max valid sum of radii.
    Returns (sum_r, radii) or (0.0, None) if infeasible.
    """
    H = 2.0 - W
    if W <= 0 or H <= 0:
        return 0.0, None
    
    n = len(centers)
    # Compute max possible radius for each center
    r_max = np.zeros(n)
    for i in range(n):
        xi, yi = centers[i]
        d_bound = min(xi, W - xi, yi, H - yi)
        d_min = d_bound
        for j in range(n):
            if i == j:
                continue
            dx = xi - centers[j][0]
            dy = yi - centers[j][1]
            d = math.sqrt(dx*dx + dy*dy)
            if d < d_min:
                d_min = d
        r_max[i] = 0.5 * d_min
    
    # The actual radius is limited by pairwise constraints.
    # We need r[i] + r[j] <= dist(i,j) for all pairs.
    # Simple approach: set r[i] = r_max[i], then iteratively reduce.
    r = r_max.copy()
    
    # Iterative reduction to satisfy all pairwise constraints
    for _ in range(50):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                d = math.sqrt(dx*dx + dy*dy)
                if r[i] + r[j] > d + 1e-12:
                    # Need to reduce
                    excess = r[i] + r[j] - d
                    # Reduce proportionally
                    if r[i] > 1e-8 and r[j] > 1e-8:
                        scale_i = r[i] / (r[i] + r[j])
                        scale_j = r[j] / (r[i] + r[j])
                        r[i] -= excess * scale_i
                        r[j] -= excess * scale_j
                        changed = True
                    elif r[i] > 1e-8:
                        r[i] -= excess
                        changed = True
                    else:
                        r[j] -= excess
                        changed = True
        if not changed:
            break
    
    # Final validation
    for i in range(n):
        if r[i] < 1e-10:
            return 0.0, None
        xi, yi = centers[i]
        if xi < r[i] - 1e-9 or xi > W - r[i] + 1e-9:
            return 0.0, None
        if yi < r[i] - 1e-9 or yi > H - r[i] + 1e-9:
            return 0.0, None
    
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            d = math.sqrt(dx*dx + dy*dy)
            if r[i] + r[j] > d + 1e-9:
                return 0.0, None
    
    return np.sum(r), r

def _objective_smooth(centers_flat, W, H, temperature=0.1):
    """
    Smooth objective: -sum(r) where r is computed via soft min of distances.
    """
    centers = centers_flat.reshape(-1, 2)
    n = len(centers)
    sum_r = 0.0
    for i in range(n):
        xi, yi = centers[i]
        d_bound = min(xi, W - xi, yi, H - yi)
        d_min = d_bound
        for j in range(n):
            if i == j:
                continue
            dx = xi - centers[j][0]
            dy = yi - centers[j][1]
            d = math.sqrt(dx*dx + dy*dy)
            if d < d_min:
                d_min = d
        r = 0.5 * d_min
        sum_r += r
    return -sum_r

def _objective_and_grads(centers_flat, W, H):
    """
    Objective and gradient for center optimization.
    Objective: -sum_i r_i where r_i = 0.5 * min(dist_to_boundaries, min_j dist(i,j))
    """
    centers = centers_flat.reshape(-1, 2)
    n = len(centers)
    obj = 0.0
    grad = np.zeros_like(centers_flat)
    
    for i in range(n):
        xi, yi = centers[i]
        # Compute distances to boundaries
        d_left = xi
        d_right = W - xi
        d_bottom = yi
        d_top = H - yi
        
        # Compute distances to other centers
        d_min = min(d_left, d_right, d_bottom, d_top)
        min_type = 'bound'
        min_idx = -1
        
        for j in range(n):
            if i == j:
                continue
            dx = xi - centers[j][0]
            dy = yi - centers[j][1]
            d = math.sqrt(dx*dx + dy*dy)
            if d < d_min:
                d_min = d
                min_type = 'center'
                min_idx = j
        
        r_i = 0.5 * d_min
        obj -= r_i
        
        # Gradient with respect to center i
        if min_type == 'bound':
            if d_min == d_left:
                grad[2*i] += 0.5  # d/dxi of -0.5*xi = -0.5, but we want grad of objective
                # objective = -0.5 * d_left = -0.5 * xi
                # d(obj)/dxi = -0.5
                # So grad[2*i] = -0.5
                grad[2*i] = -0.5
                grad[2*i+1] = 0.0
            elif d_min == d_right:
                # obj = -0.5 * (W - xi), d/dxi = 0.5
                grad[2*i] = 0.5
                grad[2*i+1] = 0.0
            elif d_min == d_bottom:
                # obj = -0.5 * yi, d/dyi = -0.5
                grad[2*i] = 0.0
                grad[2*i+1] = -0.5
            else:  # d_top
                # obj = -0.5 * (H - yi), d/dyi = 0.5
                grad[2*i] = 0.0
                grad[2*i+1] = 0.5
        else:
            # min_type == 'center', min_idx = j
            j = min_idx
            dx = xi - centers[j][0]
            dy = yi - centers[j][1]
            d = d_min
            if d > 1e-12:
                # obj = -0.5 * d, where d = sqrt((xi-xj)^2 + (yi-yj)^2)
                # d(obj)/dxi = -0.5 * (xi-xj)/d
                # d(obj)/dyi = -0.5 * (yi-yj)/d
                # Also affects center j:
                # d(obj)/dxj = -0.5 * (xj-xi)/d = 0.5 * (xi-xj)/d
                # d(obj)/dyj = 0.5 * (yi-yj)/d
                
                gx = -0.5 * dx / d
                gy = -0.5 * dy / d
                grad[2*i] += gx
                grad[2*i+1] += gy
                grad[2*j] -= gx
                grad[2*j+1] -= gy
            else:
                # Degenerate case
                grad[2*i] += 0.1
                grad[2*i+1] += 0.1
    
    return obj, grad

def _grid_init_centers(W, H, n_cols, n_rows):
    """
    Initialize centers in a grid pattern.
    """
    centers = []
    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if idx >= N:
                break
            cx = (col + 0.5) * (W / n_cols)
            cy = (row + 0.5) * (H / n_rows)
            centers.append([cx, cy])
            idx += 1
        if idx >= N:
            break
    
    return np.array(centers)

def _run_center_optimization(centers0, W, deadline):
    """
    Run optimization on centers only.
    """
    H = 2.0 - W
    centers_flat0 = centers0.flatten()
    
    # Bounds: centers must be within [r_min, W-r_min] x [r_min, H-r_min]
    # But we don't know r yet. Use conservative bounds.
    r_min_est = 0.01
    bounds = []
    for i in range(N):
        bounds.append((r_min_est, W - r_min_est))
        bounds.append((r_min_est, H - r_min_est))
    
    options = {
        'maxiter': 500,
        'ftol': 1e-12,
        'gtol': 1e-12,
        'disp': False
    }
    
    try:
        res = minimize(
            lambda v: _objective_and_grads(v, W, H)[0],
            lambda v: _objective_and_grads(v, W, H)[1],
            centers_flat0,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options=options
        )
        
        centers_opt = res.x.reshape(-1, 2)
        
        # Polish: iterative local improvement
        centers_polished = _polish_centers(centers_opt, W, H)
        
        val, radii = _compute_value_from_centers(centers_polished, W)
        
        if val > 0:
            return centers_polished, radii, W, val
        else:
            # Try original
            val2, radii2 = _compute_value_from_centers(centers_opt, W)
            if val2 > 0:
                return centers_opt, radii2, W, val2
            else:
                val3, radii3 = _compute_value_from_centers(centers0, W)
                return centers0, radii3, W, val3
            
    except Exception as e:
        val, radii = _compute_value_from_centers(centers0, W)
        return centers0, radii, W, val

def _polish_centers(centers, W, H, max_iter=100):
    """
    Iterative polish: move each center to maximize its radius given others fixed.
    """
    centers = centers.copy()
    n = len(centers)
    
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            # For center i, find best position that maximizes min distance to boundaries and other centers
            # Simple local search: try small perturbations
            best_pos = centers[i].copy()
            best_d = 0.0
            
            # Compute current min distance
            d_bound = min(centers[i][0], W - centers[i][0], centers[i][1], H - centers[i][1])
            d_min = d_bound
            for j in range(n):
                if i == j:
                    continue
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                d = math.sqrt(dx*dx + dy*dy)
                if d < d_min:
                    d_min = d
            best_d = d_min
            
            # Try perturbations
            step_sizes = [0.01, 0.005, 0.001]
            for step in step_sizes:
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        nx = centers[i][0] + dx
                        ny = centers[i][1] + dy
                        # Check bounds
                        if nx < 0.001 or nx > W - 0.001:
                            continue
                        if ny < 0.001 or ny > H - 0.001:
                            continue
                        # Compute min distance
                        d_bound = min(nx, W - nx, ny, H - ny)
                        d_min = d_bound
                        for j in range(n):
                            if i == j:
                                continue
                            dxj = nx - centers[j][0]
                            dyj = ny - centers[j][1]
                            d = math.sqrt(dxj*dxj + dyj*dyj)
                            if d < d_min:
                                d_min = d
                        if d_min > best_d + 1e-10:
                            best_d = d_min
                            best_pos = [nx, ny]
            
            if not np.allclose(best_pos, centers[i]):
                centers[i] = best_pos
                improved = True
        
        if not improved:
            break
    
    return centers

def construct():
    deadline = time.time() + 115.0  # Safety margin
    
    best_val = 0.0
    best_centers = None
    best_radii = None
    best_W = 1.0
    
    # Try different aspect ratios
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    
    # Grid configurations (cols, rows)
    grid_configs = [(3, 7), (4, 6), (5, 5), (6, 4), (7, 3)]
    
    # First pass: find a good aspect ratio and grid
    for W0 in aspect_ratios:
        if time.time() > deadline:
            break
        H0 = 2.0 - W0
        
        for n_cols, n_rows in grid_configs:
            if time.time() > deadline:
                break
            
            centers0 = _grid_init_centers(W0, H0, n_cols, n_rows)
            
            # Add small noise
            noise = np.random.uniform(-0.01, 0.01, centers0.shape)
            centers0 += noise
            # Clamp to valid range
            centers0[:, 0] = np.clip(centers0[:, 0], 0.01, W0 - 0.01)
            centers0[:, 1] = np.clip(centers0[:, 1], 0.01, H0 - 0.01)
            
            centers_opt, radii_opt, W_opt, val = _run_center_optimization(centers0, W0, deadline)
            
            if val > best_val:
                best_val = val
                best_centers = centers_opt.copy()
                best_radii = radii_opt.copy()
                best_W = W_opt
    
    # Second pass: refine the best aspect ratio with more restarts
    if time.time() < deadline - 15 and best_centers is not None:
        W_best = best_W
        H_best = 2.0 - W_best
        
        # Determine best grid config for this W
        if W_best < 0.9:
            grid_configs_best = [(3, 7), (4, 6)]
        elif W_best > 1.1:
            grid_configs_best = [(7, 3), (6, 4)]
        else:
            grid_configs_best = [(5, 5), (4, 6), (6, 4)]
        
        n_restarts = 8
        for _ in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            n_cols, n_rows = grid_configs_best[np.random.randint(len(grid_configs_best))]
            centers0 = _grid_init_centers(W_best, H_best, n_cols, n_rows)
            
            # Larger noise for diversity
            noise = np.random.uniform(-0.03, 0.03, centers0.shape)
            centers0 += noise
            centers0[:, 0] = np.clip(centers0[:, 0], 0.01, W_best - 0.01)
            centers0[:, 1] = np.clip(centers0[:, 1], 0.01, H_best - 0.01)
            
            centers_opt, radii_opt, W_opt, val = _run_center_optimization(centers0, W_best, deadline)
            
            if val > best_val:
                best_val = val
                best_centers = centers_opt.copy()
                best_radii = radii_opt.copy()
                best_W = W_opt
    
    # Fallback: simple grid
    if best_val == 0.0 or best_centers is None:
        W = 1.0
        H = 1.0
        side = 5
        best_W = W
        best_centers = np.zeros((N, 2))
        best_radii = np.zeros(N)
        for i in range(N):
            row = i // 5
            col = i % 5
            best_centers[i] = [(col + 0.5) / side, (row + 0.5) / side]
            best_radii[i] = 0.5 / side - 1e-4
    
    # Final validation and construction
    val_final, radii_final = _compute_value_from_centers(best_centers, best_W)
    if val_final <= 0 or radii_final is None:
        # Use fallback radii
        W = 1.0
        H = 1.0
        side = 5
        best_W = W
        best_centers = np.zeros((N, 2))
        best_radii = np.zeros(N)
        for i in range(N):
            row = i // 5
            col = i % 5
            best_centers[i] = [(col + 0.5) / side, (row + 0.5) / side]
            best_radii[i] = 0.5 / side - 1e-4
        radii_final = best_radii
    else:
        best_radii = radii_final
    
    # Construct output
    result = np.column_stack([best_centers[:, 0], best_centers[:, 1], best_radii])
    
    return result

# EVOLVE-BLOCK-END
