import numpy as np
import time
from scipy.optimize import minimize

N = 200
DEADLINE = 135.0  # Safety margin for 150s limit

def compute_radius(points):
    """Compute the maximum common radius for given circle centers."""
    if points is None:
        return 0.0
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (N, 2) or not np.all(np.isfinite(points)):
        return 0.0
    
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
    return float(r)

def hex_init(n_rows):
    """Generate a hexagonal lattice initialization with n_rows rows."""
    c_base = N // n_rows
    rem = N % n_rows
    counts = [c_base] * n_rows
    for i in range(rem):
        counts[i] += 1
    
    pts = []
    y = 0.0
    for i, c in enumerate(counts):
        xs = np.arange(c) - (c - 1) / 2.0
        if i % 2 == 1:
            xs += 0.5
        for x in xs:
            pts.append([x, y])
        y += np.sqrt(3) / 2.0
    
    pts = np.array(pts, dtype=np.float64)
    # Center and scale to fit in [-0.5, 0.5]
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    max_extent = max(np.max(np.abs(pts[:, 0])), np.max(np.abs(pts[:, 1])))
    if max_extent > 0:
        scale = 0.45 / max_extent
        pts *= scale
    return pts

def grid_init(m):
    """Generate a grid initialization."""
    pts = []
    for i in range(m):
        for j in range(m):
            if len(pts) < N:
                pts.append([i/m - 0.5 + 0.5/m, j/m - 0.5 + 0.5/m])
    pts = np.array(pts[:N], dtype=np.float64)
    return pts

def energy_value_grad(pts, alpha, beta):
    """
    Compute the energy value and gradient for the inverse-power-sum objective.
    
    Energy E = sum_{i<j} d_ij^{-alpha} + sum_i wall_dist_i^{-beta}
    
    We minimize E to maximize the minimum distance.
    """
    pts = np.asarray(pts, dtype=np.float64)
    N = pts.shape[0]
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, 1e-10)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, k=1)
    d_pairs = dist[iu]
    
    # Avoid division by zero
    safe_d = np.where(d_pairs > 1e-10, d_pairs, 1e-10)
    
    # Energy from pairs: sum d^{-alpha}
    e_pairs = np.sum(safe_d ** (-alpha))
    
    # Wall distances (4 per point)
    x = pts[:, 0]
    y = pts[:, 1]
    
    w_x_l = x + 0.5
    w_x_r = 0.5 - x
    w_y_b = y + 0.5
    w_y_t = 0.5 - y
    
    # Ensure positive
    w_x_l = np.maximum(w_x_l, 1e-10)
    w_x_r = np.maximum(w_x_r, 1e-10)
    w_y_b = np.maximum(w_y_b, 1e-10)
    w_y_t = np.maximum(w_y_t, 1e-10)
    
    e_walls = np.sum(w_x_l ** (-beta) + w_x_r ** (-beta) + w_y_b ** (-beta) + w_y_t ** (-beta))
    
    value = e_pairs + e_walls
    
    # Gradient
    grad = np.zeros_like(pts)
    
    # Pairwise gradient
    # d/dp_i (d_ij^{-alpha}) = -alpha * d_ij^{-alpha-1} * (p_i - p_j) / d_ij
    #                        = -alpha * d_ij^{-alpha-2} * (p_i - p_j)
    
    # We need to sum over all j != i
    # Using the full matrix approach
    # force_weight_ij = -alpha * d_ij^{-alpha-2}
    
    safe_dist_full = np.where(dist > 1e-10, dist, 1e-10)
    force_weights = -alpha * safe_dist_full ** (-alpha - 2)
    np.fill_diagonal(force_weights, 0.0)
    
    # grad_i = sum_j force_weights[i,j] * (p_i - p_j)
    #        = p_i * sum_j force_weights[i,j] - sum_j force_weights[i,j] * p_j
    
    row_sums = np.sum(force_weights, axis=1)
    weighted_pts = force_weights.dot(pts)
    
    grad_pairs = pts * row_sums[:, np.newaxis] - weighted_pts
    
    # Wall gradient
    # d/dx (w^{-beta}) = -beta * w^{-beta-1} * dw/dx
    # w_x_l = x + 0.5, dw/dx = 1
    # w_x_r = 0.5 - x, dw/dx = -1
    # w_y_b = y + 0.5, dw/dy = 1
    # w_y_t = 0.5 - y, dw/dy = -1
    
    grad_walls_x = -beta * (w_x_l ** (-beta - 1) * 1 + w_x_r ** (-beta - 1) * (-1))
    grad_walls_y = -beta * (w_y_b ** (-beta - 1) * 1 + w_y_t ** (-beta - 1) * (-1))
    
    grad[:, 0] += grad_pairs[:, 0] + grad_walls_x
    grad[:, 1] += grad_pairs[:, 1] + grad_walls_y
    
    return value, grad

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_r = 0
    best_points = None
    
    # Initializations
    inits = []
    
    # 1. Random
    for _ in range(15):
        pts = np.random.rand(N, 2) - 0.5
        inits.append(pts)
        
    # 2. Hexagonal
    for n_rows in range(12, 19):
        pts = hex_init(n_rows)
        inits.append(pts)
        
    # 3. Grid
    m = 14
    grid_pts = grid_init(m)
    grid_pts += np.random.randn(N, 2) * 0.01
    inits.append(grid_pts)
    
    # Optimization loop
    for init in inits:
        if time.time() > deadline - 5:
            break
            
        pts = init.copy()
        
        # Define objective for L-BFGS-B
        def obj(x):
            x = x.reshape(N, 2)
            val, grad = energy_value_grad(x, alpha=4.0, beta=2.0)
            return val, grad.flatten()
        
        # L-BFGS-B
        try:
            res = minimize(obj, pts.flatten(), method='L-BFGS-B', jac=True,
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success:
                pts_opt = res.x.reshape(N, 2)
                r = compute_radius(pts_opt)
                if r > best_r:
                    best_r = r
                    best_points = pts_opt.copy()
        except Exception:
            pass
            
        # Check time
        if time.time() > deadline - 10:
            break
    
    # Final polish with higher alpha if time remains
    if best_points is not None and time.time() < deadline - 5:
        def obj_polish(x):
            x = x.reshape(N, 2)
            val, grad = energy_value_grad(x, alpha=8.0, beta=4.0)
            return val, grad.flatten()
            
        try:
            res = minimize(obj_polish, best_points.flatten(), method='L-BFGS-B', jac=True,
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10})
            if res.success:
                pts_polished = res.x.reshape(N, 2)
                r_polished = compute_radius(pts_polished)
                if r_polished > best_r:
                    best_r = r_polished
                    best_points = pts_polished.copy()
        except Exception:
            pass
            
    if best_points is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
        
    return best_points
