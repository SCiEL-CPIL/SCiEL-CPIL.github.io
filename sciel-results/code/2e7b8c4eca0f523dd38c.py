"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_score(points):
    """Compute the packing score: largest common radius r."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Check boundary constraint
    dist_to_center = np.linalg.norm(points, axis=1)
    boundary_margin = 1.0 - dist_to_center
    if np.any(boundary_margin < -1e-10):
        return 0.0
    
    # Check pairwise distances
    # Use efficient computation
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, np.min(boundary_margin))
    return max(r, 0.0)


def _sunflower_init(n):
    """Generate initial points using the sunflower/Fibonacci disk model."""
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    angles = np.arange(n) * golden_angle
    radii = np.sqrt(np.arange(1, n + 1) / n)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])


def _relaxation_step(points, step_size, r_target):
    """Perform one relaxation step to maximize minimum distance."""
    n = len(points)
    forces = np.zeros_like(points)
    
    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Find nearest neighbors for each point
    min_dists = np.min(dists, axis=1)
    
    # For each point, compute repulsion from nearest neighbors
    for i in range(n):
        # Find all points within 2*r_target (overlapping or close)
        close_mask = dists[i] < 2.0 * r_target
        close_indices = np.where(close_mask)[0]
        
        for j in close_indices:
            d = dists[i, j]
            if d < 1e-12:
                # Add small random perturbation
                forces[i] += np.random.randn(2) * 1e-6
            else:
                # Repulsive force proportional to overlap
                overlap = 2.0 * r_target - d
                if overlap > 0:
                    direction = (points[i] - points[j]) / d
                    forces[i] += overlap * direction
    
    # Boundary constraint: push points inward if too close to boundary
    dist_to_center = np.linalg.norm(points, axis=1)
    boundary_margin = 1.0 - dist_to_center
    for i in range(n):
        if boundary_margin[i] < r_target:
            direction = -points[i] / (dist_to_center[i] + 1e-12)
            push = r_target - boundary_margin[i]
            forces[i] += push * direction
    
    # Apply forces with step size
    points_new = points + step_size * forces
    
    # Ensure points stay within unit circle
    new_dist = np.linalg.norm(points_new, axis=1)
    mask = new_dist > 1.0 - 1e-10
    if np.any(mask):
        points_new[mask] = points_new[mask] * (1.0 - 1e-10) / new_dist[mask]
    
    return points_new


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline with 10s safety margin
    
    # Initialize with sunflower pattern
    points = _sunflower_init(N)
    
    # Initial score
    current_score = _compute_score(points)
    best_points = points.copy()
    best_score = current_score
    
    # Adaptive step size
    step_size = 0.01
    r_target = current_score if current_score > 0 else 0.01
    
    # Relaxation loop
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # Perform relaxation step
        points_new = _relaxation_step(points, step_size, r_target)
        
        # Compute new score
        new_score = _compute_score(points_new)
        
        if new_score > current_score:
            # Accept the move
            points = points_new
            current_score = new_score
            no_improve_count = 0
            
            # Update target radius
            r_target = current_score
            
            # Track best
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
            
            # Increase step size slightly
            step_size = min(step_size * 1.1, 0.05)
        else:
            # Reject the move
            no_improve_count += 1
            
            # Decrease step size
            step_size = max(step_size * 0.9, 0.001)
            
            # If no improvement for a while, try a different strategy
            if no_improve_count >= max_no_improve:
                # Try random perturbation
                perturb_mask = np.random.rand(N) < 0.1
                points[perturb_mask] += np.random.randn(np.sum(perturb_mask), 2) * 0.01
                
                # Ensure points stay in circle
                dist = np.linalg.norm(points, axis=1)
                mask = dist > 1.0 - 1e-10
                if np.any(mask):
                    points[mask] = points[mask] * (1.0 - 1e-10) / dist[mask]
                
                current_score = _compute_score(points)
                no_improve_count = 0
                step_size = 0.01
        
        # Check deadline
        if time.time() > deadline - 5.0:
            break
    
    # Return the best configuration found
    return best_points
# EVOLVE-BLOCK-END
