# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    
    Strategy:
    1. Initialize with icosahedron vertices (12) + 2 poles (high symmetry)
    2. Iterative local search to maximize d_min/d_max ratio
    3. Deadline guard for safety
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 550  # 50s safety margin

    def get_score(pts):
        """Compute (d_min/d_max)^2"""
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-15:
            return 0.0
        return (d_min / d_max) ** 2

    def time_exceeded():
        return time.time() > deadline - 10  # 10s safety margin

    # --- Initialization: Icosahedron + 2 poles ---
    # Icosahedron vertices (normalized to unit sphere)
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    ico_raw = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(ico_raw, axis=1, keepdims=True)
    ico = ico_raw / norms
    
    # Place 2 poles along the z-axis (aligned with 5-fold symmetry axis)
    pole_north = np.array([0.0, 0.0, 1.0])
    pole_south = np.array([0.0, 0.0, -1.0])
    
    # Combine: 12 icosahedron points + 2 poles
    points = np.vstack([ico, pole_north, pole_south])
    
    # Scale to unit sphere (all points already on unit sphere)
    best_points = points.copy()
    best_score = get_score(points)
    
    # --- Local search: coordinate-wise perturbation ---
    # Try to improve by small perturbations
    rng = np.random.RandomState(42)
    step_size = 0.01
    max_iter = 50000
    
    for iteration in range(max_iter):
        if time_exceeded():
            break
        
        # Randomly pick a point and a coordinate to perturb
        pt_idx = rng.randint(n)
        coord_idx = rng.randint(d)
        delta = rng.uniform(-step_size, step_size)
        
        # Save original
        orig_val = points[pt_idx, coord_idx]
        
        # Apply perturbation
        points[pt_idx, coord_idx] = orig_val + delta
        
        # Compute new score
        new_score = get_score(points)
        
        if new_score > best_score:
            best_score = new_score
            best_points = points.copy()
            # Increase step size slightly on improvement
            step_size = min(step_size * 1.05, 0.1)
        else:
            # Revert
            points[pt_idx, coord_idx] = orig_val
            # Decrease step size
            step_size = max(step_size * 0.95, 0.001)
    
    # Final refinement with smaller steps
    step_size = 0.001
    for iteration in range(20000):
        if time_exceeded():
            break
        
        pt_idx = rng.randint(n)
        coord_idx = rng.randint(d)
        delta = rng.uniform(-step_size, step_size)
        
        orig_val = best_points[pt_idx, coord_idx]
        best_points[pt_idx, coord_idx] = orig_val + delta
        
        new_score = get_score(best_points)
        
        if new_score > best_score:
            best_score = new_score
        else:
            best_points[pt_idx, coord_idx] = orig_val
    
    return best_points

# EVOLVE-BLOCK-END
