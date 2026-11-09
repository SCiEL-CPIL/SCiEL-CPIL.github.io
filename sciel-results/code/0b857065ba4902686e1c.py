import numpy as np
import time

N = 100
CUBE_HALF = 0.5
DEADLINE_S = 115.0

def _grid_score(points):
    """
    Compute the largest radius r for the given points.
    r = min(min_pair_dist / 2, min_boundary_dist)
    """
    if len(points) < 2:
        return 0.5
    # Pairwise distances
    # d[i, j] = ||points[i] - points[j]||
    d = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    
    # Boundary distances
    # Distance to each of the 6 faces
    # x: 0.5 - |x|
    # y: 0.5 - |y|
    # z: 0.5 - |z|
    margins = CUBE_HALF - np.abs(points)
    min_boundary = np.min(margins)
    
    return min(min_pair, min_boundary)

def _initial_grid():
    """
    Create a 5x4x5 grid.
    """
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid

def construct():
    start_time = time.time()
    
    # Initialize
    points = _initial_grid()
    # Add small random noise to break symmetry
    rng = np.random.default_rng(42)
    points = points + rng.normal(0, 0.01, points.shape)
    points = np.clip(points, -CUBE_HALF, CUBE_HALF)
    
    best_points = points.copy()
    best_score = _grid_score(points)
    
    # Optimization parameters
    step_size = 0.01
    step_decay = 0.999
    min_step = 1e-6
    push_radius_factor = 1.5 # Push from neighbors within 1.5 * current_min_dist
    
    # Main loop
    while time.time() - start_time < DEADLINE_S:
        # Calculate current min distance
        d = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1))
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d)
        
        # Calculate wall distances
        margins = CUBE_HALF - np.abs(points)
        min_boundary = np.min(margins)
        
        current_score = min(min_pair, min_boundary)
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
        # If step size is too small, break
        if step_size < min_step:
            break
            
        # Determine push radius
        # We want to push points that are close to each other or to the wall
        # Let's use a fixed threshold relative to the current min distance
        # Or just push all points based on a potential field
        push_radius = max(1.2 * min_pair, 1.2 * min_boundary, 0.05)
        
        forces = np.zeros_like(points)
        
        # Pairwise repulsion
        # Only consider pairs within push_radius
        mask = (d < push_radius) & (d > 1e-9)
        # For each point i, sum forces from j
        # F_i = sum_j (r_i - r_j) / d_ij^2
        # We can vectorize this
        # diff = points[:, None, :] - points[None, :, :]
        # force_vec = diff / (d**2 + 1e-9)**1.5 # Inverse square law magnitude, vector direction
        # Actually, magnitude 1/d^2, direction (r_i - r_j)/d. So vector is (r_i - r_j)/d^3.
        
        diff = points[:, None, :] - points[None, :, :]
        d_cubed = d**3 + 1e-12
        # Set diagonal to 0
        np.fill_diagonal(d_cubed, 1.0) # Avoid div by 0, will be masked anyway
        force_vec = diff / d_cubed[:, :, None]
        
        # Apply mask
        force_vec[~mask] = 0
        
        forces += np.sum(force_vec, axis=1)
        
        # Wall repulsion
        # For each point, if close to a wall, push inward
        # x dimension
        # dist to -0.5 wall: points[:,0] + 0.5
        # dist to +0.5 wall: 0.5 - points[:,0]
        
        # We can calculate a force vector for each dimension
        # fx = 0
        # if x < 0: dist = x + 0.5. If dist < push_radius: fx += 1/dist^2
        # if x > 0: dist = 0.5 - x. If dist < push_radius: fx -= 1/dist^2
        
        # Vectorized wall force
        # dist_left = points[:, 0] + 0.5
        # dist_right = 0.5 - points[:, 0]
        
        # Let's do it for all 3 dims
        for dim in range(3):
            dist_neg = points[:, dim] + CUBE_HALF
            dist_pos = CUBE_HALF - points[:, dim]
            
            mask_neg = dist_neg < push_radius
            mask_pos = dist_pos < push_radius
            
            # Force from negative wall (pushes in +dim direction)
            f_neg = np.zeros(N)
            f_neg[mask_neg] = 1.0 / (dist_neg[mask_neg]**2 + 1e-12)
            
            # Force from positive wall (pushes in -dim direction)
            f_pos = np.zeros(N)
            f_pos[mask_pos] = 1.0 / (dist_pos[mask_pos]**2 + 1e-12)
            
            forces[:, dim] += f_neg - f_pos
            
        # Normalize forces to prevent large jumps?
        # The magnitude of forces can vary.
        # Let's scale by step_size
        points += step_size * forces
        
        # Clip to cube
        points = np.clip(points, -CUBE_HALF, CUBE_HALF)
        
        # Decay step size
        step_size *= step_decay
        
        # Occasionally, if stuck, do a random kick?
        # For now, simple decay.
        
    return best_points
