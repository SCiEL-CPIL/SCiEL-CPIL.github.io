"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_lattice_init(N):
    """Generate initial points on a hexagonal lattice."""
    # Estimate spacing for N points in unit circle
    # Area ~ pi, point area ~ (2r)^2 * 0.866
    # r ~ 1/sqrt(N)
    # Spacing d ~ 2r
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    
    # Generate a grid larger than needed
    k = int(1.0 / spacing) + 3
    rows = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                rows.append((x, y))
                
    points = np.array(rows)
    if len(points) < N:
        # Fallback to random if lattice fails (should not happen)
        return np.random.randn(N, 2) * 0.5
    
    # Select N points closest to center to ensure good coverage
    radii = np.hypot(points[:, 0], points[:, 1])
    indices = np.argsort(radii)[:N]
    return points[indices]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    # 1. Initialization
    points = _hex_lattice_init(N)
    r = 0.01
    alpha = 0.005 # Learning rate
    
    best_r = 0.0
    best_points = points.copy()
    
    # Pre-allocate arrays for speed?
    # We will recompute pairwise distances each step.
    
    while time.time() < deadline:
        # Compute pairwise distances
        # diff[i, j] = points[i] - points[j]
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Mask for i != j
        mask = ~np.eye(N, dtype=bool)
        dist[~mask] = np.inf
        
        # Compute gradient of energy
        # E = sum max(0, 2r - dist)^2
        # dE/dc_i = -2 * sum_j max(0, 2r - dist_ij) * (c_i - c_j) / dist_ij
        
        overlap = np.maximum(0, 2*r - dist)
        
        # Gradient term from pairs
        # factor[i, j] = overlap[i, j] / dist[i, j]
        # Handle dist=0
        factor = np.zeros_like(dist)
        nonzero = dist > 1e-8
        factor[nonzero] = overlap[nonzero] / dist[nonzero]
        
        # grad[i] = -2 * sum_j factor[i, j] * (c_i - c_j)
        # Note: diff[i, j] = c_i - c_j
        grad_pairs = -2.0 * np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary term
        # E_b = sum max(0, |c_i| - (1-r))^2
        # dE_b/dc_i = -2 * max(0, |c_i| - (1-r)) * c_i / |c_i|
        
        radii = np.linalg.norm(points, axis=1)
        boundary_violation = np.maximum(0, radii - (1.0 - r))
        
        # Handle center point
        grad_boundary = np.zeros_like(points)
        nonzero_r = radii > 1e-8
        grad_boundary[nonzero_r] = -2.0 * (boundary_violation[nonzero_r] / radii[nonzero_r])[:, np.newaxis] * points[nonzero_r]
        
        total_grad = grad_pairs + grad_boundary
        
        # Update points
        # Use a small random jitter to escape local minima
        jitter = np.random.randn(N, 2) * 0.001
        points = points + alpha * total_grad + jitter
        
        # Keep points within a safe bound (e.g., radius 1.5) to prevent divergence
        # If a point goes too far, pull it back
        radii_new = np.linalg.norm(points, axis=1)
        too_far = radii_new > 1.5
        if np.any(too_far):
            points[too_far] *= (1.0 / radii_new[too_far])[:, np.newaxis]
            
        # Check energy to decide if we can increase r
        # Calculate current min distance and min boundary distance
        min_dist = np.min(dist[mask])
        min_boundary = 1.0 - np.max(radii)
        
        current_r = min(min_dist / 2.0, min_boundary)
        
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
            
        # Adjust r and alpha
        # If the packing is "tight" (current_r is close to r), we can increase r
        # If current_r is much smaller than r, we have violated constraints, reduce r or alpha
        
        if current_r >= r * 0.95:
            # We are stable, increase r
            r *= 1.001
            alpha *= 0.99 # Slow down as we get tighter
        else:
            # Unstable, reduce r slightly and increase alpha to recover
            r *= 0.99
            alpha *= 1.01
            if alpha > 0.05: alpha = 0.05
            if r < 0.005: r = 0.005
            
        # Periodic restart to escape local minima?
        # If no improvement for a while, we could jitter more.
        # For now, the jitter is constant.
        
    return best_points
# EVOLVE-BLOCK-END
