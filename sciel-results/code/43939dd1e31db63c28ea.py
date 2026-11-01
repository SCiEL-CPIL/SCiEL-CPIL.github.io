"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time

def _cost_vec(circles, weight=1.0):
    """
    Calculate the cost of a configuration using vectorized operations.
    Cost = Penalty - weight * sum(radii)
    """
    n = len(circles)
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    sum_r = np.sum(rs)
    
    # Boundary penalties
    pen_x_low = np.maximum(0, rs - xs) ** 2
    pen_x_high = np.maximum(0, xs + rs - 1) ** 2
    pen_y_low = np.maximum(0, rs - ys) ** 2
    pen_y_high = np.maximum(0, ys + rs - 1) ** 2
    boundary_pen = np.sum(pen_x_low) + np.sum(pen_x_high) + np.sum(pen_y_low) + np.sum(pen_y_high)
    
    # Pairwise overlaps
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = rs[:, None] + rs[None, :]
    overlaps = np.maximum(0, min_dist - dist)
    overlap_pen = np.sum(overlaps**2) / 2.0
    
    total_pen = boundary_pen + overlap_pen
    return total_pen - weight * sum_r

def _is_valid_vec(circles, eps=1e-6):
    """
    Check validity using vectorized operations.
    """
    n = len(circles)
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    if np.any(rs < -eps): return False
    if np.any(xs - rs < -eps) or np.any(xs + rs > 1 + eps): return False
    if np.any(ys - rs < -eps) or np.any(ys + rs > 1 + eps): return False
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = rs[:, None] + rs[None, :]
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    if np.any(dist[mask] < min_dist[mask] - eps):
        return False
    return True

def construct():
    """
    Pack 32 circles with disjoint interiors inside [0,1]^2, maximizing sum of radii.
    Uses Simulated Annealing.
    """
    np.random.seed(42)
    
    # Initial state: Grid
    side = int(np.ceil(np.sqrt(N)))
    circles = np.zeros((N, 3))
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N: break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.4 / side
            circles[idx] = [x, y, r]
            idx += 1
            
    current_circles = circles.copy()
    current_cost = _cost_vec(current_circles, weight=1.0)
    
    best_circles = current_circles.copy()
    best_sum_r = -np.inf
    
    if _is_valid_vec(current_circles):
        best_sum_r = np.sum(current_circles[:, 2])

    # Simulated Annealing Parameters
    T_start = 0.05
    T_end = 0.0005
    num_iterations = 100000
    time_limit = 110.0
    start_time = time.time()
    
    move_scale_pos = 0.05
    move_scale_r = 0.02
    
    for k in range(num_iterations):
        if time.time() - start_time > time_limit:
            break
            
        alpha = k / num_iterations
        T = T_start * (T_end / T_start) ** alpha
        
        new_circles = current_circles.copy()
        idx = np.random.randint(N)
        
        # Perturb
        new_circles[idx, 0] += np.random.uniform(-move_scale_pos, move_scale_pos)
        new_circles[idx, 1] += np.random.uniform(-move_scale_pos, move_scale_pos)
        new_circles[idx, 2] += np.random.uniform(-move_scale_r, move_scale_r)
        
        # Clamp
        new_circles[idx, 0] = np.clip(new_circles[idx, 0], 0, 1)
        new_circles[idx, 1] = np.clip(new_circles[idx, 1], 0, 1)
        new_circles[idx, 2] = np.clip(new_circles[idx, 2], 0.001, 0.5)
        
        new_cost = _cost_vec(new_circles, weight=1.0)
        
        delta = new_cost - current_cost
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_circles = new_circles
            current_cost = new_cost
            
            if _is_valid_vec(current_circles):
                sum_r = np.sum(current_circles[:, 2])
                if sum_r > best_sum_r:
                    best_sum_r = sum_r
                    best_circles = current_circles.copy()

    if not _is_valid_vec(best_circles):
        best_circles = circles.copy()
        
    return best_circles
# EVOLVE-BLOCK-END
