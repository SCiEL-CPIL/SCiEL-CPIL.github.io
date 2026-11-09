import numpy as np
import time
from scipy.spatial import cKDTree

N = 200

def compute_radius(points):
    """Compute the maximum common radius for given circle centers."""
    if points is None:
        return 0.0
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (N, 2) or not np.all(np.isfinite(points)):
        return 0.0
    
    # Calculate min pairwise distance
    tree = cKDTree(points)
    # Use query_pairs to find all pairs within a small distance
    # But we need the minimum, so let's just do a full distance matrix for N=200
    # 200x200 = 40000 elements, very fast
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
        scale = 0.48 / max_extent
        pts *= scale
    return pts

def soft_min_radius(pts, lam=50.0):
    """
    Compute a smooth approximation of the min radius.
    Returns (value, grad) where value is the soft-min radius and grad is the gradient.
    
    We approximate min(r_pairs, r_walls) using a log-sum-exp trick on the reciprocals
    or directly on the constraints.
    
    A better approach: define the objective as the minimum of:
      1. min_{i<j} d_ij / 2
      2. min_i min_k (0.5 - |c_ik|)
    
    We can use a differentiable soft-min. Let's define:
    R_soft = -1/lam * log( sum_{i<j} exp(-lam * d_ij^2/4) + sum_i sum_k exp(-lam * w_ik^2) )
    
    Wait, that's not quite right. The radius r is min(d_ij/2, w_i).
    So 1/r = max(2/d_ij, 1/w_i).
    This is a max, which is easier to smooth with log-sum-exp.
    
    Let's minimize the "violation" or maximize the "safety".
    
    Alternative: Use a penalty method or direct gradient ascent on the soft-min.
    
    Let's stick to the previous approach but optimize it.
    The previous approach used:
    F = log(sum_pairs exp(-lambda d_ij^2)) + log(sum_walls exp(-lambda w_i^2))
    This is a smooth approximation of the max of the "badness" terms.
    Minimizing F is equivalent to maximizing the soft-min distance.
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Pair weights
    w_pairs = np.exp(-lam * dist_sq)
    w_pairs[~np.isfinite(w_pairs)] = 0
    
    sum_w_pairs = np.sum(w_pairs, axis=1)
    
    # Gradient from pairs
    # d/dp_i log(sum_j exp(-lam d_ij^2)) = (1/sum_j) * sum_j [ exp(-lam d_ij^2) * (-2 lam (p_i - p_j)) ]
    term = w_pairs[:, :, np.newaxis] * (-2 * lam) * diff
    grad_pairs = np.sum(term, axis=1)
    
    sum_w_pairs_safe = np.where(sum_w_pairs > 0, sum_w_pairs, 1.0)
    grad_pairs /= sum_w_pairs_safe[:, np.newaxis]
    
    # Wall terms
    x = pts[:, 0]
    y = pts[:, 1]
    
    d_x_l = x + 0.5
    d_x_r = 0.5 - x
    d_y_b = y + 0.5
    d_y_t = 0.5 - y
    
    d_x_l_sq = d_x_l**2
    d_x_r_sq = d_x_r**2
    d_y_b_sq = d_y_b**2
    d_y_t_sq = d_y_t**2
    
    w_x_l = np.exp(-lam * d_x_l_sq)
    w_x_r = np.exp(-lam * d_x_r_sq)
    w_y_b = np.exp(-lam * d_y_b_sq)
    w_y_t = np.exp(-lam * d_y_t_sq)
    
    sum_w_walls = w_x_l + w_x_r + w_y_b + w_y_t
    
    grad_x_walls = (w_x_l * (-2 * lam * d_x_l) + w_x_r * (2 * lam * d_x_r)) / sum_w_walls
    grad_y_walls = (w_y_b * (-2 * lam * d_y_b) + w_y_t * (2 * lam * d_y_t)) / sum_w_walls
    
    grad_walls = np.column_stack((grad_x_walls, grad_y_walls))
    
    total_grad = grad_pairs + grad_walls
    
    # The value F is log-sum-exp of the terms.
    # We don't strictly need F for the update, just the gradient direction.
    # But for monitoring, we can compute it.
    # F = log(sum_pairs) + log(sum_walls) is not quite right because they are separate sums.
    # Actually, the objective is sum of two log-sum-exp terms.
    # F = log(sum_{i<j} exp(-lam d_ij^2)) + log(sum_i sum_k exp(-lam w_ik^2))
    # This is what we are minimizing.
    
    return total_grad

def repulsive_relax(pts, n_steps=50, lr=0.001, force_k=1.0, wall_k=1.0):
    """Simple repulsive force-based relaxation to spread points apart."""
    pts = pts.copy()
    for step in range(n_steps):
        # Compute pairwise repulsive forces
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)
        
        # Repulsive force: k / dist^2 in direction of separation
        force_mask = dist < 0.15
        force_mag = np.zeros_like(dist)
        force_mag[force_mask] = force_k / (dist[force_mask] ** 2)
        
        forces = np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Wall repulsion
        x = pts[:, 0]
        y = pts[:, 1]
        wall_force_x = np.zeros_like(x)
        wall_force_y = np.zeros_like(y)
        
        d_left = x + 0.5
        wall_force_x[d_left < 0.05] += wall_k * (0.05 - d_left[d_left < 0.05])
        d_right = 0.5 - x
        wall_force_x[d_right < 0.05] -= wall_k * (0.05 - d_right[d_right < 0.05])
        d_bottom = y + 0.5
        wall_force_y[d_bottom < 0.05] += wall_k * (0.05 - d_bottom[d_bottom < 0.05])
        d_top = 0.5 - y
        wall_force_y[d_top < 0.05] -= wall_k * (0.05 - d_top[d_top < 0.05])
        
        forces[:, 0] += wall_force_x
        forces[:, 1] += wall_force_y
        
        pts += lr * forces
        pts = np.clip(pts, -0.49, 0.49)
    
    return pts

def polish(pts, deadline, n_steps=500):
    """
    Final polish phase using a more refined gradient ascent.
    Focuses on the most constraining pairs/walls.
    """
    pts = pts.copy()
    lam = 100.0
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        grad = soft_min_radius(pts, lam)
        
        # Adaptive step size
        # Estimate curvature or just use a decaying lr
        lr = 0.005 / (1 + step * 0.01)
        
        pts_new = pts - lr * grad
        pts_new = np.clip(pts_new, -0.49, 0.49)
        
        # Check if we improved
        # For speed, we can skip this and just accept
        # Or do a line search?
        # Let's just accept for now.
        pts = pts_new
        
        # Occasionally add small noise to escape local minima
        if step % 50 == 0:
            pts += np.random.randn(N, 2) * 0.001
            pts = np.clip(pts, -0.49, 0.49)
            
    return pts

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin for safety
    
    best_r = 0
    best_points = None
    
    # Initializations
    inits = []
    
    # 1. Random
    for _ in range(50):
        pts = np.random.rand(N, 2) - 0.5
        inits.append(pts)
        
    # 2. Hexagonal
    for n_rows in range(10, 21):
        pts = hex_init(n_rows)
        inits.append(pts)
        
    # 3. Grid with jitter
    m = 14
    grid_pts = []
    for i in range(m):
        for j in range(m):
            if len(grid_pts) < N:
                grid_pts.append([i/m - 0.5 + 0.5/m, j/m - 0.5 + 0.5/m])
    if len(grid_pts) >= N:
        grid_pts = np.array(grid_pts[:N])
        grid_pts += np.random.randn(N, 2) * 0.01
        inits.append(grid_pts)
    
    # Optimization loop
    lambda_val = 50.0
    
    for init in inits:
        if time.time() > deadline:
            break
            
        pts = init.copy()
        
        # First, do a quick repulsive relaxation to get out of obvious overlaps
        pts = repulsive_relax(pts, n_steps=20, lr=0.001)
        
        # Gradient ascent on soft-min
        for step in range(1000):
            if time.time() > deadline:
                break
                
            grad = soft_min_radius(pts, lambda_val)
            
            # Adaptive learning rate
            lr = 0.001 / (1 + step * 0.001)
            
            pts_new = pts - lr * grad
            pts_new = np.clip(pts_new, -0.49, 0.49)
            
            pts = pts_new
            
            # Every 100 steps, check radius
            if step % 100 == 0:
                r = compute_radius(pts)
                if r > best_r:
                    best_r = r
                    best_points = pts.copy()
                    
                # Random restart/jitter if stuck?
                if step % 200 == 0:
                    pts += np.random.randn(N, 2) * 0.01
                    pts = np.clip(pts, -0.49, 0.49)
                    
    # Final check
    if best_points is not None:
        r = compute_radius(best_points)
        if r > best_r:
            best_r = r
            # best_points already set
    
    # Final polish on best solution
    if best_points is not None and time.time() < deadline - 5:
        polished = polish(best_points, deadline, n_steps=500)
        r_polished = compute_radius(polished)
        if r_polished > best_r:
            best_r = r_polished
            best_points = polished
    
    if best_points is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
        
    return best_points
