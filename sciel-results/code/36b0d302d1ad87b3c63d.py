import numpy as np
import time

N = 50

def _hex_lattice_init(n, scale=1.0):
    """
    Generate a hexagonal lattice of points, centered at origin.
    Scaled to fit within a circle of radius `scale`.
    """
    # Generate a large grid
    k = int(np.sqrt(n)) + 2
    points = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * 1.0
            y = j * np.sqrt(3) / 2.0
            points.append((x, y))
    points = np.array(points)
    
    # Keep only points within a circle of radius scale
    radii = np.hypot(points[:, 0], points[:, 1])
    mask = radii <= scale
    points = points[mask]
    
    # Sort by distance from center to keep the most central ones
    if len(points) > n:
        indices = np.argsort(radii[mask])
        points = points[indices[:n]]
    elif len(points) < n:
        # If not enough points, add random points near the boundary
        while len(points) < n:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, scale)
            points = np.vstack([points, [[r * np.cos(angle), r * np.sin(angle)]]])
            
    return points

def _relax(points, r, max_steps=5000, alpha=0.05, tol=1e-6):
    """
    Relax the system to find a valid configuration for radius r.
    Returns (points, success).
    """
    points = points.copy()
    n = len(points)
    
    for step in range(max_steps):
        # Compute pairwise distances
        # Use broadcasting for efficiency
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        # Avoid division by zero
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)
        
        # Check for overlaps
        # Overlap if dist < 2r
        overlap_mask = dist < 2.0 * r
        np.fill_diagonal(overlap_mask, False)
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Repulsive forces between overlapping circles
        # F = k * (2r - dist) * direction
        # direction = diff / dist
        # We only apply force if overlapping
        # To avoid issues with dist=0 (should not happen with diagonal filled), 
        # we use a safe division
        inv_dist = np.zeros_like(dist)
        np.fill_diagonal(inv_dist, 0.0)
        mask_non_zero = dist > 1e-10
        inv_dist[mask_non_zero] = 1.0 / dist[mask_non_zero]
        
        # Force magnitude proportional to overlap
        overlap_amt = np.maximum(0, 2.0 * r - dist)
        overlap_amt[~overlap_mask] = 0.0
        
        # Force vector = overlap_amt * diff * inv_dist
        # Sum forces for each point
        # forces[i] = sum_j overlap_amt[i,j] * diff[i,j] * inv_dist[i,j]
        # diff[i,j] = points[i] - points[j]
        # So force on i is towards j if overlapping? No, repulsive, so away from j.
        # diff[i,j] is vector from j to i. So adding this pushes i away from j. Correct.
        
        # Vectorized sum:
        # term = overlap_amt * inv_dist
        # forces = np.sum(term[:, :, np.newaxis] * diff, axis=1)
        
        term = overlap_amt * inv_dist
        forces += np.sum(term[:, :, np.newaxis] * diff, axis=1)
        
        # Confining forces
        # If |p| > 1-r, push towards center
        radii = np.hypot(points[:, 0], points[:, 1])
        outside_mask = radii > (1.0 - r)
        
        # Force direction is -p / |p|
        # Magnitude proportional to how far outside
        outside_amt = np.maximum(0, radii - (1.0 - r))
        outside_amt[~outside_mask] = 0.0
        
        # inv_radii
        inv_radii = np.zeros_like(radii)
        mask_r_non_zero = radii > 1e-10
        inv_radii[mask_r_non_zero] = 1.0 / radii[mask_r_non_zero]
        
        # Force = -outside_amt * points * inv_radii
        confining_forces = -outside_amt[:, np.newaxis] * points * inv_radii[:, np.newaxis]
        forces += confining_forces
        
        # Update positions
        # Adaptive step size
        current_alpha = alpha * (1.0 - step / max_steps)
        new_points = points + current_alpha * forces
        
        # Check convergence
        displacement = np.max(np.hypot(new_points[:, 0] - points[:, 0], new_points[:, 1] - points[:, 1]))
        points = new_points
        
        if displacement < tol:
            # Check if valid
            if _is_valid(points, r):
                return points, True
            else:
                # Stalled at an invalid local minimum
                # Add jitter
                points += np.random.normal(0, 0.01, points.shape)
                
    return points, _is_valid(points, r)

def _is_valid(points, r):
    """Check if points form a valid packing for radius r."""
    n = len(points)
    # Check container
    radii = np.hypot(points[:, 0], points[:, 1])
    if np.any(radii > 1.0 - r + 1e-9):
        return False
        
    # Check overlaps
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    if np.any(dist < 2.0 * r - 1e-9):
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial points
    # Scale lattice to fit inside unit circle with some margin for r
    # r_max_est ~ 0.125
    # Diameter ~ 0.25
    # Lattice spacing ~ 0.25
    # Let's initialize with a slightly smaller r to ensure fit, then expand
    init_r = 0.1
    points = _hex_lattice_init(N, scale=1.0 - init_r)
    
    best_r = 0.0
    best_points = points
    
    r_low = 0.05
    r_high = 0.13
    
    # Binary search
    while time.time() < deadline:
        r_mid = (r_low + r_high) / 2.0
        
        # Relax
        pts, success = _relax(points, r_mid, max_steps=2000, alpha=0.02)
        
        if success:
            best_r = r_mid
            best_points = pts
            r_low = r_mid
            points = pts  # Use successful config as start for next
        else:
            r_high = r_mid
            # If we fail, keep current points but maybe jitter slightly
            # to help next attempt if r_low is close to r_high
            if r_high - r_low < 1e-4:
                # Stuck, try a different initialization
                points = _hex_lattice_init(N, scale=1.0 - r_low)
                points += np.random.normal(0, 0.005, points.shape)
                # Reset search range slightly? Or just continue
                # For now, just continue binary search
                pass
            
        # Convergence check
        if r_high - r_low < 1e-5:
            break
            
    return best_points
