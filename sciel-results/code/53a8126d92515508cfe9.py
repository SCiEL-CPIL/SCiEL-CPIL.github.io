import numpy as np
import time

N = 200
DEADLINE = 110.0

def _compute_radius(points):
    # points: (N, 2)
    # Returns radius r = min(dist_center, dist_wall) / 2
    
    # Center-center distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_center_dist = np.sqrt(min_dist_sq)
    
    # Wall distances
    dx = 0.5 - np.abs(points[:, 0])
    dy = 0.5 - np.abs(points[:, 1])
    min_wall_dist = np.min(np.concatenate([dx, dy]))
    
    r = min(min_center_dist, min_wall_dist) / 2.0
    return r

def _generate_hex_grid(n):
    # Generate a hexagonal grid of points
    # s is the spacing
    s = np.sqrt(2.0 / (n * np.sqrt(3)))
    cols = int(np.ceil(1.0 / s))
    rows = int(np.ceil(1.0 / (s * np.sqrt(3) / 2)))
    
    points = []
    for r in range(rows):
        for c in range(cols):
            x = -0.5 + (c + 0.5) * s
            y = -0.5 + (r + 0.5) * s * np.sqrt(3) / 2
            if r % 2 != 0:
                x += s / 2.0
            points.append([x, y])
            
    points = np.array(points)
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    if len(points) > n:
        indices = np.random.choice(len(points), n, replace=False)
        points = points[indices]
    elif len(points) < n:
        missing = n - len(points)
        random_pts = np.random.uniform(-0.5, 0.5, (missing, 2))
        points = np.vstack([points, random_pts])
        
    return points

def construct():
    start_time = time.time()
    best_score = 0.0
    best_points = None
    
    # Multiple restarts to escape local minima
    # Each restart uses a different initial configuration or random seed
    
    while time.time() - start_time < DEADLINE - 5.0:
        np.random.seed(int(time.time() * 1000) % 10000)
        points = _generate_hex_grid(N)
        # Add small noise to break symmetry
        points += np.random.normal(0, 0.001, points.shape)
        points = np.clip(points, -0.49, 0.49)
        
        # Simulated Annealing parameters
        # We will ramp up the radius
        r = 0.01 # Start small
        dr = 0.0005 # Step size for radius
        temp = 0.01
        cooling = 0.999
        min_temp = 0.001
        
        # Pre-allocate arrays for speed
        # We will do many iterations
        
        valid_count = 0
        max_valid_streak = 1000 # If valid for this many steps, increase r
        
        for iter in range(20000): # Hard cap per restart
            if time.time() - start_time > DEADLINE - 2.0:
                break
            
            # 1. Calculate Forces
            # Pairwise distances
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            dist = np.sqrt(dist_sq)
            np.fill_diagonal(dist, np.inf)
            
            # Force from other circles
            # Only consider pairs where dist < 2*r
            # Force magnitude: k / dist^2 (repulsive)
            # Direction: diff / dist
            # F_i = sum_j (k / dist_ij^2) * (p_i - p_j) / dist_ij
            #     = sum_j (k / dist_ij^3) * (p_i - p_j)
            
            # Mask for active pairs
            mask = dist < 2.0 * r
            # Avoid division by zero (dist is inf on diagonal)
            mask = mask & (dist > 1e-6)
            
            # Calculate force vector
            # factor = k / dist^3 for active pairs, 0 otherwise
            k = 1.0
            factor = np.zeros_like(dist)
            factor[mask] = k / (dist[mask] ** 3)
            
            # Force on i is sum_j factor_ij * (p_i - p_j)
            # (p_i - p_j) is diff_ij
            # Sum over j (axis=1)
            forces = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
            
            # Wall Forces
            # If x < -0.5 + r, push right
            # If x > 0.5 - r, push left
            # Force magnitude proportional to penetration
            wall_k = 10.0 # Stronger wall force to keep them in
            
            # X walls
            pen_x_left = np.maximum(0, (-0.5 + r) - points[:, 0])
            pen_x_right = np.maximum(0, (points[:, 0] - (0.5 - r)))
            forces[:, 0] += wall_k * (pen_x_left - pen_x_right)
            
            # Y walls
            pen_y_left = np.maximum(0, (-0.5 + r) - points[:, 1])
            pen_y_right = np.maximum(0, (points[:, 1] - (0.5 - r)))
            forces[:, 1] += wall_k * (pen_y_left - pen_y_right)
            
            # 2. Update Positions
            # Learning rate
            lr = 0.01 * temp
            points += lr * forces
            points += np.random.normal(0, temp * 0.01, points.shape)
            
            # Keep points inside square (hard constraint for stability)
            points = np.clip(points, -0.5, 0.5)
            
            # 3. Check Validity and Ramp Radius
            # Calculate min distances
            min_dist = np.min(dist)
            min_wall_dist = np.min(np.minimum(
                np.minimum(points[:, 0] + 0.5, 0.5 - points[:, 0]),
                np.minimum(points[:, 1] + 0.5, 0.5 - points[:, 1])
            ))
            
            current_r = min(min_dist, min_wall_dist) / 2.0
            
            # Update best score
            if current_r > best_score:
                best_score = current_r
                best_points = points.copy()
            
            # Logic for ramping r
            # If current config is valid for r, try to increase r
            # If not, we are stuck, maybe lower r or just keep optimizing
            if current_r >= r:
                valid_count += 1
                if valid_count > max_valid_streak:
                    r += dr
                    valid_count = 0
            else:
                valid_count = 0
                # If we are significantly invalid, don't increase r
                # Optionally, we could decrease r to allow movement, but ramping is usually better
                # Let's stick to current r and let forces resolve it
            
            # Cooling
            temp *= cooling
            if temp < min_temp:
                break
        
        # End of restart loop

    if best_points is None:
        # Fallback
        best_points = _generate_hex_grid(N)
        best_points = np.clip(best_points, -0.49, 0.49)
        
    return best_points
