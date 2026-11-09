import numpy as np
import time
from scipy.optimize import minimize

N = 200

def compute_radius(points):
    # points: (N, 2)
    # Calculate min pairwise distance
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist_sq = np.min(dist_sq)
    
    # Calculate min wall distance
    x = points[:, 0]
    y = points[:, 1]
    dist_x = 0.5 - np.abs(x)
    dist_y = 0.5 - np.abs(y)
    min_wall_dist = np.min(np.minimum(dist_x, dist_y))
    
    r = np.min([np.sqrt(min_pair_dist_sq)/2, min_wall_dist])
    return r

def _energy_and_grad(flat_pts, p=4, eps=1e-12):
    """
    Smooth energy: sum over pairs of 1/(d_ij^2 + eps)^p + wall penalties.
    Minimize this to maximize minimum distance.
    """
    pts = flat_pts.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Smooth pairwise energy
    # E_pairs = sum_{i<j} 1/(d_ij^2 + eps)^p
    # Only upper triangle to avoid double counting
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    denom = dist_sq + eps
    # Avoid issues with inf on diagonal (already masked)
    denom_safe = np.where(mask, denom, 1.0)
    term = np.where(mask, 1.0 / (denom_safe ** p), 0.0)
    
    E_pairs = np.sum(term)
    
    # Gradient of E_pairs w.r.t. pts
    # d/dp_i [1/(d_ij^2 + eps)^p] = -2p * (p_i - p_j) / (d_ij^2 + eps)^(p+1)
    # For each pair (i,j), contribution to grad_i = -2p*(p_i-p_j)/(d_ij^2+eps)^(p+1)
    #                contribution to grad_j =  2p*(p_i-p_j)/(d_ij^2+eps)^(p+1)
    
    denom_grad = np.where(mask, (dist_sq + eps) ** (p + 1), 1.0)
    weight = np.where(mask, -2.0 * p / denom_grad, 0.0)
    
    # grad_pairs[i] = sum_j weight[i,j] * (p_i - p_j)
    # = sum_j weight[i,j] * p_i - sum_j weight[i,j] * p_j
    # = p_i * sum_j weight[i,j] - sum_j weight[i,j] * p_j
    
    sum_w = np.sum(weight, axis=1)  # (N,)
    grad_pairs = sum_w[:, np.newaxis] * pts - np.sum(weight[:, :, np.newaxis] * pts[np.newaxis, :, :], axis=1)
    
    # Wall energy: penalize points too close to walls
    # For each point, distance to nearest wall
    # Use smooth penalty: sum of 1/(d_wall + eps)^p for each wall
    x = pts[:, 0]
    y = pts[:, 1]
    
    d_x_l = x + 0.5
    d_x_r = 0.5 - x
    d_y_b = y + 0.5
    d_y_t = 0.5 - y
    
    # Clip to avoid negative (shouldn't happen if points are in bounds, but be safe)
    d_x_l = np.maximum(d_x_l, eps)
    d_x_r = np.maximum(d_x_r, eps)
    d_y_b = np.maximum(d_y_b, eps)
    d_y_t = np.maximum(d_y_t, eps)
    
    E_walls = np.sum(1.0 / (d_x_l ** p) + 1.0 / (d_x_r ** p) + 1.0 / (d_y_b ** p) + 1.0 / (d_y_t ** p))
    
    # Gradient of wall energy
    # d/dx [1/(x+0.5)^p] = -p/(x+0.5)^(p+1)
    # d/dx [1/(0.5-x)^p] = p/(0.5-x)^(p+1)
    grad_x_walls = -p / (d_x_l ** (p+1)) + p / (d_x_r ** (p+1))
    grad_y_walls = -p / (d_y_b ** (p+1)) + p / (d_y_t ** (p+1))
    
    grad_walls = np.column_stack((grad_x_walls, grad_y_walls))
    
    total_E = E_pairs + E_walls
    total_grad = grad_pairs + grad_walls
    
    return total_E, total_grad.flatten()

def _hex_init(n_rows):
    """Generate hexagonal grid initialization."""
    c_base = N // n_rows
    rem = N % n_rows
    counts = [c_base] * n_rows
    for i in range(rem):
        counts[i] += 1
        
    pts = []
    y = 0
    for i, c in enumerate(counts):
        xs = np.arange(c) - (c - 1) / 2.0
        if i % 2 == 1:
            xs += 0.5
        for x in xs:
            pts.append([x, y])
        y += np.sqrt(3) / 2.0
        
    pts = np.array(pts)
    max_x = np.max(np.abs(pts[:, 0]))
    max_y = np.max(np.abs(pts[:, 1]))
    scale = 0.45 / max(max_x, max_y)
    pts *= scale
    return pts

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
    best_r = 0
    best_points = None
    
    # Generate diverse initializations
    inits = []
    
    # Hexagonal grids with different row counts
    for n_rows in range(12, 22):
        inits.append(_hex_init(n_rows))
    
    # Random initializations
    rng = np.random.RandomState(42)
    for _ in range(10):
        pts = rng.rand(N, 2) - 0.5
        inits.append(pts)
    
    # Grid initialization
    m = 14  # 14x14 = 196, close to 200
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= N:
        inits.append(grid[:N].copy())
    else:
        # Add more points
        extra = N - len(grid)
        extra_pts = rng.rand(extra, 2) - 0.5
        inits.append(np.vstack([grid, extra_pts]))
    
    p = 4  # Power for inverse distance penalty
    
    for init in inits:
        if time.time() > deadline:
            break
            
        pts = init.copy()
        # Ensure points are strictly inside
        pts = np.clip(pts, -0.499, 0.499)
        
        flat_pts = pts.flatten()
        
        # Use L-BFGS-B to minimize the smooth energy
        try:
            result = minimize(
                _energy_and_grad,
                flat_pts,
                args=(p,),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
            )
            if result.success or result.fun < 1e6:
                new_pts = result.x.reshape(N, 2)
                new_pts = np.clip(new_pts, -0.5, 0.5)
                r = compute_radius(new_pts)
                if r > best_r:
                    best_r = r
                    best_points = new_pts.copy()
        except Exception:
            pass
        
        # Also try with higher power p for finer local optimization
        if time.time() > deadline:
            break
            
        p2 = 8
        try:
            result = minimize(
                _energy_and_grad,
                flat_pts,
                args=(p2,),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 500}
            )
            if result.success or result.fun < 1e6:
                new_pts = result.x.reshape(N, 2)
                new_pts = np.clip(new_pts, -0.5, 0.5)
                r = compute_radius(new_pts)
                if r > best_r:
                    best_r = r
                    best_points = new_pts.copy()
        except Exception:
            pass
    
    # Final polish: if we have a good solution, do a very fine local optimization
    if best_points is not None and time.time() < deadline - 10:
        p_final = 16
        flat_best = best_points.flatten()
        try:
            result = minimize(
                _energy_and_grad,
                flat_best,
                args=(p_final,),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 300}
            )
            if result.success or result.fun < 1e6:
                new_pts = result.x.reshape(N, 2)
                new_pts = np.clip(new_pts, -0.5, 0.5)
                r = compute_radius(new_pts)
                if r > best_r:
                    best_r = r
                    best_points = new_pts.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
        
    return best_points
