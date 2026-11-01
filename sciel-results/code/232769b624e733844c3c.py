import numpy as np
import time

N = 300

def _get_hex_points(spacing):
    dx = spacing
    dy = spacing * np.sqrt(3) / 2
    
    # Cover a square that contains the unit circle
    margin = 0.1
    x_range = np.arange(-1.0 - margin, 1.0 + margin, dx)
    y_range = np.arange(-1.0 - margin, 1.0 + margin, dy)
    
    points = []
    for i, y in enumerate(y_range):
        offset = (i % 2) * (dx / 2)
        for x in x_range:
            px = x + offset
            py = y
            if px*px + py*py <= 1.0:
                points.append([px, py])
            
    if not points:
        return np.empty((0, 2))
    return np.array(points)

def _init_packing(seed_offset=0):
    # Try a few spacings to find a good valid initialization
    base_spacing = 0.11
    spacings = [base_spacing * (1.0 + 0.01 * seed_offset), 
                base_spacing * (0.99 + 0.01 * seed_offset),
                base_spacing * (1.01 + 0.01 * seed_offset)]
    
    np.random.seed(seed_offset)
    
    best_points = None
    best_score = -1
    
    for s in spacings:
        points = _get_hex_points(s)
        if len(points) < N:
            continue
        
        # Filter to valid region for radius s/2
        radii = np.hypot(points[:, 0], points[:, 1])
        mask = radii <= 1.0 - s/2
        points = points[mask]
        
        if len(points) < N:
            continue
            
        # Select N points closest to center
        radii = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(radii)
        candidate = points[indices[:N]].copy()
        
        # Calculate score
        diff = candidate[:, np.newaxis, :] - candidate[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        radii_c = np.linalg.norm(candidate, axis=1)
        min_bound = np.min(1.0 - radii_c)
        
        score = min(min_pair / 2.0, min_bound)
        
        if score > best_score:
            best_score = score
            best_points = candidate
            
    if best_points is None:
        # Fallback: Random points inside 0.9 radius
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.random.uniform(0, 0.9, N)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack((x, y))
        
    return best_points

def _optimize(points, time_limit):
    start_time = time.time()
    
    best_radius = -1.0
    best_points = points.copy()
    
    step_size = 0.005
    noise_level = 0.005
    stagnation_counter = 0
    epsilon = 1e-5
    
    while time.time() < time_limit:
        # Calculate distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        
        radii = np.linalg.norm(points, axis=1)
        boundary_dists = 1.0 - radii
        min_boundary_dist = np.min(boundary_dists)
        
        current_radius = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_points = points.copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            if stagnation_counter > 20:
                points += np.random.normal(0, noise_level, points.shape)
                noise_level *= 0.95
                stagnation_counter = 0
                
        # Forces
        forces = np.zeros_like(points)
        
        active_pairs = dists < (2 * current_radius + epsilon)
        
        weights = np.zeros_like(dists)
        # Add epsilon to avoid division by zero
        weights[active_pairs] = 1.0 / (dists[active_pairs] ** 2 + 1e-8)
        
        row_sums = np.sum(weights, axis=1)
        term1 = points * row_sums[:, np.newaxis]
        term2 = weights @ points
        
        forces += (term1 - term2)
        
        active_boundary = boundary_dists < (current_radius + epsilon)
        
        norms = np.linalg.norm(points, axis=1)
        norms[norms < 1e-8] = 1e-8
        unit_vectors = points / norms[:, np.newaxis]
        
        active_mask = active_boundary[:, np.newaxis]
        forces += -unit_vectors * active_mask
        
        points += step_size * forces
        
        step_size *= 0.9995
        if step_size < 1e-7:
            step_size = 1e-7
            
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    best_points = None
    best_score = -1.0
    
    # Run multiple restarts
    n_restarts = 3
    time_per_restart = (deadline - start_time) / n_restarts
    
    for i in range(n_restarts):
        remaining_time = deadline - time.time()
        if remaining_time < 2.0:
            break
            
        # Allocate time for this restart, keep 1s for next/overhead
        t_limit = min(time_per_restart, remaining_time - 1.0)
        if t_limit < 1.0:
            break
            
        points = _init_packing(seed_offset=i)
        optimized_points, score = _optimize(points, time.time() + t_limit)
        
        if score > best_score:
            best_score = score
            best_points = optimized_points
            
    if best_points is None:
        # Should not happen
        best_points = _init_packing(0)
        
    return best_points
