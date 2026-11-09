import numpy as np
import time

N = 200

def compute_radius(points):
    if len(points) != N:
        return 0.0
    
    dist_to_wall_x = 0.5 - np.abs(points[:, 0])
    dist_to_wall_y = 0.5 - np.abs(points[:, 1])
    dist_to_wall = np.minimum(dist_to_wall_x, dist_to_wall_y)
    
    min_dist_to_wall = np.min(dist_to_wall)
    
    if min_dist_to_wall < 0:
        return 0.0

    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
    np.fill_diagonal(dist, np.inf)
    
    min_dist_between = np.min(dist) / 2.0
    
    r = min(min_dist_between, min_dist_to_wall)
    return r

def hex_init(n):
    best_r = 0
    best_points = None
    best_R = 0
    
    for R in range(10, 21):
        q, rem = divmod(n, R)
        n_max = q + (1 if rem > 0 else 0)
        
        r_width = 1.0 / (2.0 * n_max)
        r_height = 1.0 / (np.sqrt(3.0) * (R - 1) + 2.0)
        r = min(r_width, r_height)
        
        if r > best_r:
            best_r = r
            best_R = R

    R = best_R
    q, rem = divmod(n, R)
    
    lengths = [q] * R
    for i in range(rem):
        lengths[i] = q + 1
        
    w = 2.0
    h = np.sqrt(3.0)
    
    points = []
    for i in range(R):
        n_i = lengths[i]
        xs = np.arange(n_i) * w - (n_i - 1) * w / 2.0
        y_i = i * h
        
        if i % 2 == 1:
            xs = xs + w / 2.0
            
        for x in xs:
            points.append((x, y_i))
            
    points = np.array(points)
    
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    points = points - [cx, cy]
    
    x_max = np.max(np.abs(points[:, 0]))
    y_max = np.max(np.abs(points[:, 1]))
    
    term_x = x_max + 1.0
    term_y = y_max + 1.0
    k = 0.5 / max(term_x, term_y)
    
    points = points * k
    
    return points

def optimize(points, start_time, deadline, beta=50.0, lr=0.005, max_steps=2000):
    current_r = compute_radius(points)
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        
        weights = np.exp(-beta * dist) * beta
        np.fill_diagonal(weights, 0.0)
        
        dist_safe = np.where(dist > 1e-12, dist, 1.0)
        term = weights[:, :, np.newaxis] * diff / dist_safe[:, :, np.newaxis]
        grad_pairs = np.sum(term, axis=1)
        
        # Add wall repulsion
        # Force away from walls
        F_wall = np.zeros_like(points)
        
        # Vectorized wall force
        # d_x_left = points[:, 0] + 0.5
        # d_x_right = 0.5 - points[:, 0]
        # d_y_bottom = points[:, 1] + 0.5
        # d_y_top = 0.5 - points[:, 1]
        
        d_x_left = points[:, 0] + 0.5
        d_x_right = 0.5 - points[:, 0]
        d_y_bottom = points[:, 1] + 0.5
        d_y_top = 0.5 - points[:, 1]
        
        # We want to push away from the closest wall.
        # This is equivalent to maximizing the minimum distance to any wall.
        # Gradient of min(d1, d2) is 1 for the min term.
        
        min_d = np.minimum(np.minimum(d_x_left, d_x_right), np.minimum(d_y_bottom, d_y_top))
        
        # Force magnitude
        # Use a smooth function like 1/(min_d + eps)^2
        eps = 1e-6
        force_mag = 1.0 / (min_d + eps)**2
        
        # Direction
        # If min_d is d_x_left, push right (+x)
        # If min_d is d_x_right, push left (-x)
        # etc.
        
        # We can use np.where
        # But we need to handle ties.
        
        F_wall[:, 0] = np.where(min_d == d_x_left, force_mag, 0.0) - np.where(min_d == d_x_right, force_mag, 0.0)
        F_wall[:, 1] = np.where(min_d == d_y_bottom, force_mag, 0.0) - np.where(min_d == d_y_top, force_mag, 0.0)
        
        grad_total = grad_pairs + F_wall
        
        points = points + lr * grad_total
        
        # Clamp to square
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        
        new_r = compute_radius(points)
        
        if new_r < current_r:
            # Step back? Or just reduce lr?
            # For simplicity, just reduce lr
            lr = max(lr * 0.9, 1e-6)
        else:
            current_r = new_r
            
    return points, current_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_r = 0
    best_points = None
    
    init_points = hex_init(N)
    
    num_restarts = 5
    
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        if i == 0:
            points = init_points.copy()
        else:
            points = init_points + np.random.normal(0, 0.01, (N, 2))
            points = np.clip(points, -0.49, 0.49)
            
        points, r = optimize(points, start_time, deadline, beta=50.0, lr=0.005, max_steps=2000)
        
        if r > best_r:
            best_r = r
            best_points = points.copy()
            
    if best_points is not None and time.time() < deadline - 5.0:
        points, r = optimize(best_points, start_time, deadline, beta=100.0, lr=0.001, max_steps=1000)
        if r > best_r:
            best_r = r
            best_points = points.copy()
            
    if best_points is None:
        best_points = init_points
        
    return best_points
