"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius_fast(pts):
    """
    Compute the packing radius for given points using vectorized operations.
    r = min(min_pair_dist / 2, min_boundary_dist)
    """
    if len(pts) < 2:
        return 0.0
    
    # 1. Minimum distance to boundary
    # dist_to_center = np.linalg.norm(pts, axis=1)
    # min_boundary = 1.0 - np.min(dist_to_center)
    
    # Vectorized pairwise distances
    # pts shape: (N, 2)
    # diff shape: (N, N, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dist, np.inf)
    
    min_pair_dist = np.min(dist)
    
    # Boundary check
    r_centers = np.linalg.norm(pts, axis=1)
    min_boundary = 1.0 - np.min(r_centers)
    
    return min(min_pair_dist / 2.0, min_boundary)

def _init_hex(n_points):
    """Initialize points on a hexagonal lattice."""
    # Estimate spacing. Area of unit circle = pi. Area per circle ~ pi * r^2.
    # N * pi * r^2 <= pi => r <= 1/sqrt(N) ~ 0.057.
    # Spacing d = 2r ~ 0.114.
    # Let's start with a slightly smaller spacing to ensure enough points, then optimize.
    spacing = 0.115
    
    points = []
    # Generate enough points
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3) / 2
            if np.hypot(x, y) <= 0.9: # Keep well inside initially
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < n_points:
        # Fallback if not enough points generated (should not happen with these bounds)
        # Generate more
        for j in range(-15, 16):
            for i in range(-15, 16):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3) / 2
                if np.hypot(x, y) <= 0.95:
                    points = np.vstack([points, [[x, y]]])
        
    # Select n_points closest to center
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)[:n_points]
    return points[indices]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    pts = _init_hex(N)
    current_r = _compute_radius_fast(pts)
    best_pts = pts.copy()
    best_r = current_r
    
    # Simulated Annealing Parameters
    T = 0.005 # Initial temperature
    T_min = 1e-7
    alpha = 0.9999 # Cooling factor
    move_scale = 0.05 # Initial step size
    
    n_steps = 0
    
    while time.time() < deadline:
        # Check if we should reduce move scale
        # If T is low, reduce move scale to fine-tune
        if T < 0.001:
            move_scale = 0.01
        elif T < 0.0001:
            move_scale = 0.005
        else:
            move_scale = 0.05
            
        # Pick a random point
        idx = np.random.randint(N)
        
        # Propose move
        delta = np.random.randn(2) * move_scale
        new_pts = pts.copy()
        new_pts[idx] += delta
        
        # Check boundary constraint for the moved point
        if np.linalg.norm(new_pts[idx]) > 1.0:
            # Reject immediately
            continue
            
        # Compute new radius
        new_r = _compute_radius_fast(new_pts)
        
        # Acceptance criterion
        delta_r = new_r - current_r
        if delta_r > 0 or (T > 0 and np.random.rand() < np.exp(delta_r / T)):
            pts = new_pts
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
            
        # Cool down
        T *= alpha
        n_steps += 1
        
        # Periodic "shake" if stuck? 
        # If no improvement in last 1000 steps, maybe increase T slightly?
        # For now, simple exponential cooling.
        
    # Final local polish: 
    # Try to maximize the minimum distance by nudging the closest pairs
    # This is optional if SA runs long enough, but helps with precision.
    if time.time() < deadline - 5.0:
        for _ in range(1000):
            if time.time() > deadline - 5.0: break
            # Find closest pair
            diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, np.inf)
            min_idx = np.unravel_index(np.argmin(dist), dist.shape)
            i, j = min_idx
            d = dist[i, j]
            
            # Move i slightly away from j
            direction = (best_pts[i] - best_pts[j]) / d
            # Try moving i in direction
            step = 0.001
            new_i = best_pts[i] + direction * step
            if np.linalg.norm(new_i) < 1.0:
                trial = best_pts.copy()
                trial[i] = new_i
                r_trial = _compute_radius_fast(trial)
                if r_trial > best_r:
                    best_r = r_trial
                    best_pts = trial
            # Move j slightly away from i
            new_j = best_pts[j] - direction * step
            if np.linalg.norm(new_j) < 1.0:
                trial = best_pts.copy()
                trial[j] = new_j
                r_trial = _compute_radius_fast(trial)
                if r_trial > best_r:
                    best_r = r_trial
                    best_pts = trial

    return best_pts
# EVOLVE-BLOCK-END
