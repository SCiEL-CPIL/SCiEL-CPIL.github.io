"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 300

# EVOLVE-BLOCK-START
import numpy as np

def _init_hex_grid(N, target_radius=1.0):
    """
    Initialize points on a hexagonal grid, clipped to the unit circle.
    If too many points, subsample. If too few, add random points.
    """
    # Estimate spacing for hex grid
    # Area of hex unit cell for N points in unit circle
    # Area ~ N * (sqrt(3)/2 * s^2) = pi * 1^2
    # s ~ sqrt(2 * pi / (N * sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    
    # Generate grid points
    # Hex grid coordinates:
    # x = (col + 0.5 * (row % 2)) * s
    # y = row * s * sqrt(3)/2
    max_coords = int(np.ceil(1.0 / s)) + 2
    
    points = []
    for row in range(-max_coords, max_coords + 1):
        for col in range(-max_coords, max_coords + 1):
            x = (col + 0.5 * (row % 2)) * s
            y = row * s * np.sqrt(3.0) / 2.0
            if np.sqrt(x*x + y*y) < 0.99: # Keep well inside initially
                points.append([x, y])
                
    points = np.array(points)
    
    # Shuffle to break symmetry
    np.random.shuffle(points)
    
    if len(points) > N:
        points = points[:N]
    elif len(points) < N:
        # Add random points
        num_add = N - len(points)
        random_pts = np.random.uniform(-0.9, 0.9, size=(num_add, 2))
        # Ensure they are inside
        norms = np.linalg.norm(random_pts, axis=1)
        mask = norms > 0.9
        random_pts[mask] = random_pts[mask] / norms[mask, np.newaxis] * 0.9
        points = np.vstack([points, random_pts])
        
    return points

def _compute_forces(points):
    """
    Compute repulsive forces between all pairs.
    F_i = sum_{j != i} (p_i - p_j) / |p_i - p_j|^3
    This is the gradient of the potential sum 1/d_ij.
    """
    N_pts = len(points)
    forces = np.zeros_like(points)
    
    # Vectorized distance calculation
    # diff[i, j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero
    np.fill_diagonal(dist_sq, 1.0)
    
    # Force magnitude is 1 / dist_sq (since F_vec = diff / dist^3 = diff / (dist_sq * dist))
    # Actually, force vector = (p_i - p_j) / |p_i - p_j|^3
    # |p_i - p_j|^3 = (dist_sq * dist_sq)^0.5 * dist_sq = dist_sq^1.5
    # So factor = 1 / dist_sq^1.5
    
    # We want to maximize the minimum distance.
    # A strong repulsion (1/d^2 force) helps push close pairs apart.
    
    inv_dist_cubed = 1.0 / (dist_sq * np.sqrt(dist_sq))
    np.fill_diagonal(inv_dist_cubed, 0.0)
    
    # Force on i from j is proportional to diff[i,j] * inv_dist_cubed[i,j]
    # Sum over j
    forces = np.sum(diff * inv_dist_cubed[:, :, np.newaxis], axis=1)
    
    return forces

def _optimize_forces(points, steps=5000):
    """
    Optimize using force-directed relaxation with adaptive step size.
    """
    pts = points.copy()
    # Start with a relatively large step, decrease it
    step_size = 0.05
    cooling = 0.999
    
    for s in range(steps):
        forces = _compute_forces(pts)
        
        # Add small noise to escape local minima
        noise = np.random.normal(0, 0.001, size=pts.shape)
        
        # Update positions
        pts += step_size * (forces + noise)
        
        # Enforce boundary constraint
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / norms[mask, np.newaxis] * 0.999999
            
        # Cool down
        step_size *= cooling
        
    return pts

def _score(points):
    """
    Calculate the score: min(min_pair_dist/2, min_dist_to_boundary)
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    norms = np.linalg.norm(points, axis=1)
    min_boundary_dist = 1.0 - np.max(norms)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def construct():
    best_points = None
    best_score = -1.0
    
    # Run multiple trials with different initializations
    for trial in range(5):
        # Initialize with hex grid
        init_pts = _init_hex_grid(N)
        
        # Add slight perturbation
        init_pts += np.random.normal(0, 0.005, size=init_pts.shape)
        
        # Ensure inside
        norms = np.linalg.norm(init_pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            init_pts[mask] = init_pts[mask] / norms[mask, np.newaxis] * 0.999999
            
        # Optimize
        optimized = _optimize_forces(init_pts, steps=8000)
        
        # Final polish with smaller steps
        optimized = _optimize_forces(optimized, steps=2000)
        
        score = _score(optimized)
        if score > best_score:
            best_score = score
            best_points = optimized.copy()
            
    # Ensure validity
    if best_points is not None:
        norms = np.linalg.norm(best_points, axis=1)
        mask = norms >= 1.0
        if np.any(mask):
            best_points[mask] = best_points[mask] / norms[mask, np.newaxis] * 0.999999
            
    return best_points
# EVOLVE-BLOCK-END
