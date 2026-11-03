"""Structural change: Force-directed relaxation with momentum (sum-of-squared-distances objective)."""
N = 200
import numpy as np
import time

_DEADLINE = 115.0
_START_TIME = time.time()

def _check_time():
    return time.time() - _START_TIME < _DEADLINE

def _compute_radius(centers):
    """Return the largest common radius for given centers in [-0.5,0.5]^2."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    boundary = np.min(np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    if boundary < 0:
        return 0.0
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(boundary, min_pair / 2.0)

def _force_directed_step(centers, velocity, lr, momentum, boundary_penalty):
    """
    One step of force-directed relaxation.
    Objective: maximize sum of squared pairwise distances, penalize boundary violations.
    Gradient of sum_{i<j} |c_i - c_j|^2 wrt c_i is:
      sum_{j != i} 2*(c_i - c_j)
    This encourages uniform spreading.
    Boundary penalty: for each point, add gradient pushing away from boundaries if too close.
    """
    # Compute forces from pairwise repulsion
    # F_i = sum_{j != i} 2 * (c_i - c_j) = 2 * N * c_i - 2 * sum_j c_j
    # This is O(N) per point if we use the identity: sum_{j != i}(c_i - c_j) = (N-1)*c_i - (sum_j c_j - c_i) = N*c_i - sum_j c_j
    sum_c = np.sum(centers, axis=0)  # (2,)
    # F_i = 2 * (N * c_i - sum_c)
    forces = 2.0 * (N * centers - sum_c[np.newaxis, :])
    
    # Boundary forces: push points away from edges
    # For x: if c_i_x > 0.5 - margin, push left; if c_i_x < -0.5 + margin, push right
    # Use a smooth penalty: gradient of -0.5 * max(0, |c| - (0.5 - eps))^2
    # Simpler: linear penalty when within 'boundary_width' of edge
    boundary_width = 0.05
    # x boundary
    # If x > 0.5 - boundary_width: force_x -= strength * (x - (0.5 - boundary_width))
    # If x < -0.5 + boundary_width: force_x += strength * ((-0.5 + boundary_width) - x)
    strength = boundary_penalty
    
    # x forces
    over_x_right = np.maximum(0, centers[:, 0] - (0.5 - boundary_width))
    over_x_left = np.maximum(0, (-0.5 + boundary_width) - centers[:, 0])
    forces[:, 0] -= strength * over_x_right
    forces[:, 0] += strength * over_x_left
    
    # y forces
    over_y_right = np.maximum(0, centers[:, 1] - (0.5 - boundary_width))
    over_y_left = np.maximum(0, (-0.5 + boundary_width) - centers[:, 1])
    forces[:, 1] -= strength * over_y_right
    forces[:, 1] += strength * over_y_left
    
    # Update velocity with momentum
    velocity = momentum * velocity + forces
    centers_new = centers + lr * velocity
    
    # Clip to stay inside
    centers_new[:, 0] = np.clip(centers_new[:, 0], -0.5, 0.5)
    centers_new[:, 1] = np.clip(centers_new[:, 1], -0.5, 0.5)
    
    return centers_new, velocity

def _initial_hex_grid(n_points):
    """Generate a hexagonal grid of points."""
    s = np.sqrt(2.0 / (n_points * np.sqrt(3)))
    row_height = s * np.sqrt(3) / 2
    n_rows = int(1.0 / row_height) + 2
    n_cols = int(1.0 / s) + 2
    
    total_width = (n_cols - 1) * s
    total_height = (n_rows - 1) * row_height
    
    start_x = -0.5 + (1.0 - total_width) / 2
    start_y = -0.5 + (1.0 - total_height) / 2
    
    points = []
    for i in range(n_rows):
        for j in range(n_cols):
            x = start_x + j * s
            if i % 2 == 1:
                x += s / 2
            y = start_y + i * row_height
            if -0.49 <= x <= 0.49 and -0.49 <= y <= 0.49:
                points.append([x, y])
                
    points = np.array(points)
    
    if len(points) > n_points:
        center_dist = np.sum(points ** 2, axis=1)
        idx = np.argsort(center_dist)[:n_points]
        points = points[idx]
        
    if len(points) < n_points:
        extra = np.random.uniform(-0.45, 0.45, size=(n_points - len(points), 2))
        points = np.vstack([points, extra])
        
    return points[:n_points]

def construct():
    start_time = time.time()
    
    # 1. Initialize with hex grid + noise
    centers = _initial_hex_grid(N)
    centers += np.random.normal(0, 0.001, size=centers.shape)
    centers[:, 0] = np.clip(centers[:, 0], -0.49, 0.49)
    centers[:, 1] = np.clip(centers[:, 1], -0.49, 0.49)
    
    r_best = _compute_radius(centers)
    best_centers = centers.copy()
    
    velocity = np.zeros_like(centers)
    
    # 2. Force-directed relaxation with adaptive learning rate
    # Start with moderate lr, decrease over time
    lr = 0.001
    momentum = 0.9
    boundary_penalty = 10.0
    
    n_steps = 5000
    
    for step in range(n_steps):
        if not _check_time():
            break
        
        # Adaptive learning rate: decay over time
        lr_current = lr * (1.0 - step / n_steps) + 0.0001
        
        centers, velocity = _force_directed_step(
            centers, velocity, lr_current, momentum, boundary_penalty
        )
        
        # Occasionally check radius
        if step % 100 == 0:
            r_current = _compute_radius(centers)
            if r_current > r_best:
                r_best = r_current
                best_centers = centers.copy()
    
    # 3. Fine-tuning phase: smaller steps, focus on local improvements
    # Reset velocity for a fresh local search
    velocity = np.zeros_like(centers)
    lr_fine = 0.0001
    momentum_fine = 0.5
    boundary_penalty_fine = 50.0
    
    n_fine_steps = 3000
    
    for step in range(n_fine_steps):
        if not _check_time():
            break
        
        lr_current = lr_fine * (1.0 - step / n_fine_steps) + 0.00001
        
        centers, velocity = _force_directed_step(
            centers, velocity, lr_current, momentum_fine, boundary_penalty_fine
        )
        
        if step % 50 == 0:
            r_current = _compute_radius(centers)
            if r_current > r_best:
                r_best = r_current
                best_centers = centers.copy()
    
    # 4. Final polish: very small moves on the best configuration
    centers = best_centers.copy()
    velocity = np.zeros_like(centers)
    lr_polish = 0.00001
    momentum_polish = 0.3
    boundary_penalty_polish = 100.0
    
    n_polish_steps = 2000
    
    for step in range(n_polish_steps):
        if not _check_time():
            break
        
        centers, velocity = _force_directed_step(
            centers, velocity, lr_polish, momentum_polish, boundary_penalty_polish
        )
        
        if step % 20 == 0:
            r_current = _compute_radius(centers)
            if r_current > r_best:
                r_best = r_current
                best_centers = centers.copy()
    
    # Use the best found
    centers = best_centers
    
    # Final validation
    r_final = _compute_radius(centers)
    if r_final <= 0:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        centers = grid[:N]
    
    return centers
