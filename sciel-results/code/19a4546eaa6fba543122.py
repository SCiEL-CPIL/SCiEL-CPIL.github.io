import numpy as np
import time

N = 200

def compute_radius(points):
    """Compute the actual packing radius for a given set of points."""
    if points is None:
        return 0.0
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (N, 2) or not np.all(np.isfinite(points)):
        return 0.0
    
    # Pairwise distances
    # Use broadcasting for efficiency, but be careful with memory (200x200 is fine)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist_sq = np.min(dist_sq)
    
    # Wall distances
    x = points[:, 0]
    y = points[:, 1]
    dist_x = 0.5 - np.abs(x)
    dist_y = 0.5 - np.abs(y)
    min_wall_dist = np.min(np.minimum(dist_x, dist_y))
    
    r = np.min([np.sqrt(min_pair_dist_sq)/2, min_wall_dist])
    return float(r)

def hex_grid_init():
    """Generate a hexagonal lattice initialization for N=200."""
    # Approximate square root of N
    # 14x14 grid is 196, 15x15 is 225.
    # Hexagonal packing is denser.
    # Let's try a grid of roughly 15x14 or similar and trim.
    # A standard hexagonal grid with spacing 1 has density pi/(2*sqrt(3)) ~ 0.9069.
    # Area of square is 1. Area of circles is N * pi * r^2.
    # r ~ 0.0366 => Area ~ 200 * pi * 0.00134 ~ 0.84.
    # Density ~ 0.84. This is close to hexagonal limit.
    
    # Create a grid of points
    pts = []
    # We want to fill a square of side 1.
    # Let's generate points in a hexagonal pattern and then scale/center them.
    
    # Rows
    n_rows = 14
    # Points per row
    base_count = N // n_rows
    rem = N % n_rows
    
    # We will generate a slightly larger grid and then select the best N points
    # Or just generate exactly N points in a hex pattern.
    
    # Let's generate a hex grid that is roughly square.
    # Spacing s.
    # Width ~ s * n_cols, Height ~ s * sqrt(3)/2 * n_rows.
    # We want Width ~ Height ~ 1.
    # n_cols ~ n_rows * sqrt(3)/2 ~ 14 * 0.866 ~ 12.
    
    # Let's just use a simple grid and jitter it into hex-like shape?
    # No, let's generate a proper hex grid.
    
    s = 1.0 # initial spacing
    n_rows = 14
    n_cols = 13 # 14*13 = 182, need 200.
    # Let's try 15 rows.
    n_rows = 15
    n_cols = 13 # 15*13 = 195. Close.
    
    # Actually, let's just generate a large hex grid and pick the central N points.
    # This ensures the shape is compact.
    
    max_radius = 10
    hex_pts = []
    # Hex grid coordinates
    # q, r -> x, y
    for q in range(-max_radius, max_radius + 1):
        for r in range(-max_radius, max_radius + 1):
            if abs(q + r) <= max_radius:
                x = 1.5 * q
                y = np.sqrt(3) * (r + q / 2.0)
                hex_pts.append([x, y])
                
    hex_pts = np.array(hex_pts)
    
    # Sort by distance from center
    centers = np.linalg.norm(hex_pts, axis=1)
    indices = np.argsort(centers)[:N]
    pts = hex_pts[indices]
    
    # Center the points
    pts -= np.mean(pts, axis=0)
    
    # Scale to fit in [-0.5, 0.5] with some margin
    max_extent = np.max(np.abs(pts))
    if max_extent > 0:
        scale = 0.48 / max_extent
        pts *= scale
        
    return pts

def optimize_packing(initial_pts, deadline):
    """
    Optimize the packing using Lagrangian relaxation and simulated annealing.
    """
    pts = initial_pts.copy()
    best_r = compute_radius(pts)
    best_pts = pts.copy()
    
    # Current target radius
    r_target = best_r * 0.95 # Start slightly lower to allow movement
    # Or start from a known good value? 
    # Let's start from the current best and try to increase it.
    r_current = best_r
    
    # Penalty weights
    lambda_pair = 100.0
    lambda_wall = 100.0
    
    # Annealing temperature
    T = 0.01
    T_min = 0.0001
    T_decay = 0.99
    
    n_steps = 2000
    lr = 0.001
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Calculate gradients and objective
        # Objective: E = sum max(0, 2r - d_ij)^2 + sum max(0, r - d_wall)^2
        # We want to minimize E.
        # Grad of E wrt x_i:
        # For pairs: if d_ij < 2r, grad += -4(2r - d_ij) * (x_i - x_j)/d_ij
        # For walls: if d_wall < r, grad += -2(r - d_wall) * sign(x_i)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        
        # Mask for violating pairs
        # Condition: d_ij < 2 * r_current
        # To avoid division by zero, we only compute gradient where dist > eps
        eps = 1e-8
        
        # Vectorized gradient calculation for pairs
        # term_ij = 4 * (2*r_current - dist_ij) if dist_ij < 2*r_current else 0
        # grad_i = sum_j term_ij * (x_i - x_j) / dist_ij
        
        # Note: (x_i - x_j) / dist_ij is the unit vector from j to i.
        # Let's compute the force.
        
        # We need to be careful with memory and speed.
        # 200x200 matrix operations are fast.
        
        mask_pairs = dist < (2.0 * r_current)
        # Set diagonal to False
        mask_pairs[~np.eye(N, dtype=bool)] = mask_pairs[~np.eye(N, dtype=bool)] # already handled by inf? No, inf < val is False.
        
        # Calculate penalty terms
        # penalty_ij = max(0, 2r - d)
        penalty_pairs = np.where(mask_pairs, 2.0 * r_current - dist, 0.0)
        
        # Gradient contribution from pairs
        # dE/dx_i = sum_j 4 * penalty_ij * (x_i - x_j) / d_ij
        # Note: derivative of (2r - d)^2 is 2(2r-d)(-dd/dx_i) = 2(2r-d)(d_ij/d_ij * (x_i-x_j))
        # So term is 2 * penalty * unit_vec.
        
        # We sum over j for each i.
        # grad_pairs[i] = sum_j 2 * penalty_pairs[i,j] * (pts[i] - pts[j]) / dist[i,j]
        
        # To vectorize:
        # unit_vecs = diff / dist[:, :, np.newaxis]
        # weighted_unit_vecs = (2 * penalty_pairs)[:, :, np.newaxis] * unit_vecs
        # grad_pairs = np.sum(weighted_unit_vecs, axis=1)
        
        # Handle division by zero (dist=0)
        safe_dist = np.where(dist > eps, dist, 1.0)
        unit_vecs = diff / safe_dist[:, :, np.newaxis]
        
        # Zero out non-violating pairs
        weights = 2.0 * penalty_pairs
        weights[~mask_pairs] = 0.0
        
        weighted_unit_vecs = weights[:, :, np.newaxis] * unit_vecs
        grad_pairs = np.sum(weighted_unit_vecs, axis=1)
        
        # Wall constraints
        x = pts[:, 0]
        y = pts[:, 1]
        
        d_x_l = x + 0.5
        d_x_r = 0.5 - x
        d_y_b = y + 0.5
        d_y_t = 0.5 - y
        
        # Distances to walls
        d_wall = np.minimum(np.minimum(d_x_l, d_x_r), np.minimum(d_y_b, d_y_t))
        
        # Penalty for walls
        penalty_wall = np.where(d_wall < r_current, r_current - d_wall, 0.0)
        
        # Gradient for walls
        # dE/dx = 2 * penalty * d(d_wall)/dx
        # If x < 0, d_wall = x + 0.5, deriv = 1
        # If x > 0, d_wall = 0.5 - x, deriv = -1
        # So deriv is sign(x) if violating.
        
        grad_x_walls = np.zeros_like(x)
        grad_y_walls = np.zeros_like(y)
        
        # Identify which wall is the closest and if it's violated
        # This is slightly complex to vectorize perfectly with min, but we can approximate
        # or just calculate gradient for all walls and take the max penalty?
        # Actually, the objective is sum of penalties.
        # E_wall = sum_i max(0, r - d_wall_i)^2
        # If multiple walls are close, we sum the penalties.
        
        # Let's compute gradients for each wall separately and sum them.
        # Wall Left: x = -0.5
        viol_l = d_x_l < r_current
        if np.any(viol_l):
            p_l = r_current - d_x_l[viol_l]
            grad_x_walls[viol_l] += 2.0 * p_l * 1.0 # d(x+0.5)/dx = 1
            
        # Wall Right: x = 0.5
        viol_r = d_x_r < r_current
        if np.any(viol_r):
            p_r = r_current - d_x_r[viol_r]
            grad_x_walls[viol_r] += 2.0 * p_r * (-1.0) # d(0.5-x)/dx = -1
            
        # Wall Bottom: y = -0.5
        viol_b = d_y_b < r_current
        if np.any(viol_b):
            p_b = r_current - d_y_b[viol_b]
            grad_y_walls[viol_b] += 2.0 * p_b * 1.0
            
        # Wall Top: y = 0.5
        viol_t = d_y_t < r_current
        if np.any(viol_t):
            p_t = r_current - d_y_t[viol_t]
            grad_y_walls[viol_t] += 2.0 * p_t * (-1.0)
            
        grad_walls = np.column_stack((grad_x_walls, grad_y_walls))
        
        total_grad = lambda_pair * grad_pairs + lambda_wall * grad_walls
        
        # Compute current error
        E = np.sum(penalty_pairs**2) + np.sum(penalty_wall**2)
        
        # Update points
        # Adaptive step size: reduce if E is small, increase if stuck?
        # Simple annealing: add noise
        noise = np.random.randn(N, 2) * T
        pts_new = pts - lr * total_grad + noise
        
        # Clip to square
        pts_new = np.clip(pts_new, -0.5, 0.5)
        
        # Check new radius
        r_new = compute_radius(pts_new)
        
        # Acceptance criteria
        # If r_new > r_current, accept.
        # If r_new < r_current, accept with probability exp((r_new - r_current) / T)?
        # Or simply accept if E is decreasing?
        # Since we are optimizing for r, let's try to increase r if E is small.
        
        if r_new > best_r:
            best_r = r_new
            best_pts = pts_new.copy()
            
        # Update r_current
        # If E is very small, we can increase r_current
        if E < 0.01:
            r_current = min(r_current + 0.0001, 0.04) # Cap at known best approx
        else:
            # If E is large, maybe decrease r_current slightly to help optimization?
            # Or just keep it and let gradient push points apart.
            pass
        
        pts = pts_new
        
        # Cool down
        T = max(T_min, T * T_decay)
        
        # Periodically restart noise or adjust lr
        if step % 100 == 0:
            # Slightly increase noise if stuck
            if step > 500:
                T = max(T, 0.005)
                
    return best_pts, best_r

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s safety margin
    
    best_r = 0
    best_points = None
    
    # Initialization
    init = hex_grid_init()
    
    # Run optimization
    pts, r = optimize_packing(init, deadline)
    
    if r > best_r:
        best_r = r
        best_points = pts.copy()
        
    # If we have time, try a few more restarts with different initializations
    # or polish the current best.
    
    if best_points is not None and time.time() < deadline - 10:
        # Polish phase: very small steps, no noise, high precision
        # We can run optimize_packing again with T=0 and smaller lr
        # But optimize_packing has internal T. Let's create a specific polish function or reuse with params.
        
        # For now, let's just do a few more random restarts if time permits
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            # Jitter the best points slightly
            jittered = best_points + np.random.randn(N, 2) * 0.01
            jittered = np.clip(jittered, -0.5, 0.5)
            
            pts, r = optimize_packing(jittered, min(deadline, time.time() + 10))
            if r > best_r:
                best_r = r
                best_points = pts.copy()
                
    # Fallback
    if best_points is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
        
    return best_points
