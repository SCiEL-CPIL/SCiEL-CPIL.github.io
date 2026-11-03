import numpy as np
import time

N = 200

def _compute_radius(pts):
    """Compute the maximum common radius for given centers."""
    if len(pts) == 0:
        return 0.0
    
    # Boundary distances
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    r_boundary = np.minimum(dx, dy).min()
    
    # Pairwise distances
    # For N=200, a full NxN matrix is 40,000 elements, which is small.
    # We can compute the full distance matrix.
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min()
    
    return min(min_pair / 2.0, r_boundary)

def _initial_hex_grid(seed=0):
    """Generate initial hexagonal lattice positions for N circles."""
    rng = np.random.RandomState(seed)
    
    # Try to fit N circles in a hexagonal arrangement
    # Hex grid: row y, col x
    # x = x_spacing * (col + 0.5*(row % 2))
    # y = y_spacing * row
    # y_spacing = x_spacing * sqrt(3)/2
    
    # Estimate spacing
    # Area per circle ~ 1/N. For hex, area per circle = sqrt(3)/2 * s^2.
    # s ~ sqrt(2 / (sqrt(3) * N))
    s_est = np.sqrt(2.0 / (np.sqrt(3) * N))
    
    best_pts = None
    best_r = 0
    
    # Try a few variations of the hex grid spacing and offset
    for offset_seed in range(5):
        rng = np.random.RandomState(seed + offset_seed)
        
        # Generate a large hex grid
        s = s_est * (0.95 + 0.1 * rng.random())
        y_spacing = s * np.sqrt(3) / 2
        
        # Determine number of rows and cols to cover the square
        n_rows = int(1.0 / y_spacing) + 4
        n_cols = int(1.0 / s) + 4
        
        pts = []
        for row in range(n_rows):
            y = -0.5 + y_spacing * (row + 0.5)
            if y > 0.5:
                break
            offset = 0.5 * s if row % 2 == 1 else 0.0
            for col in range(n_cols):
                x = -0.5 + s * (col + 0.5) + offset
                if x > 0.5:
                    break
                pts.append((x, y))
        
        if len(pts) >= N:
            # Shuffle to break symmetry if needed, or just take first N
            # Taking first N biases towards top-left. Better to randomize selection or center.
            # Let's center the selected points.
            pts_arr = np.array(pts[:N])
            # Add small random jitter to break symmetry
            pts_arr += rng.uniform(-0.001, 0.001, size=pts_arr.shape)
            
            # Center the points
            center = pts_arr.mean(axis=0)
            pts_arr -= center
            
            # Check if all inside
            if np.all(np.abs(pts_arr) < 0.5 - 1e-6):
                r = _compute_radius(pts_arr)
                if r > best_r:
                    best_r = r
                    best_pts = pts_arr.copy()
    
    if best_pts is None:
        # Fallback to random
        best_pts = np.random.uniform(-0.4, 0.4, size=(N, 2))
        
    return best_pts

def _dynamics_optimize(pts, time_limit=100.0):
    """
    Optimize packing using repulsive force dynamics.
    """
    start_time = time.time()
    pts = pts.copy()
    n = len(pts)
    
    # Initial radius estimate
    r_curr = _compute_radius(pts)
    
    # Parameters
    damping = 0.9
    step_size = 0.005
    force_k = 1.0
    
    # We want to maximize the minimum distance.
    # We apply forces that push particles away from each other and walls.
    
    for step in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        # 1. Compute Forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        # diff[i, j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        
        # Avoid division by zero
        np.fill_diagonal(dists, 1.0)
        
        # Force magnitude: k / dist^2
        # We only care about close pairs? No, global repulsion helps maintain structure.
        # But for packing, we want to push apart the *closest* ones.
        # Let's use a force that is significant only when dist < 2 * r_curr * 1.5
        # Or just standard 1/r^2.
        
        # To be efficient, we can mask out very distant pairs if needed, but N=200 is small.
        inv_dist_sq = 1.0 / (dists * dists)
        
        # Force on i from j is proportional to (pts[i] - pts[j]) / dist^3 ?
        # F = k * (r_i - r_j) / |r_i - r_j|^3 * k_factor
        # Let's use F_ij = k * (pts[i] - pts[j]) / dists[i,j]^3
        # This is the gradient of -k/dist
        
        # Vectorized force calculation:
        # force_i = sum_j k * (pts[i] - pts[j]) / dists[i,j]^3
        # = k * (pts[i] * sum_j 1/dists[i,j]^3 - sum_j pts[j] / dists[i,j]^3)
        
        inv_dist_cubed = inv_dist_sq / dists
        np.fill_diagonal(inv_dist_cubed, 0.0)
        
        sum_inv_dist_cubed = np.sum(inv_dist_cubed, axis=1)
        weighted_pts = inv_dist_cubed @ pts
        
        force_pair = force_k * (pts * sum_inv_dist_cubed[:, np.newaxis] - weighted_pts)
        
        # Wall repulsion
        # Push inwards if close to wall
        # dx = 0.5 - x, dy = 0.5 - y
        dx = 0.5 - np.abs(pts[:, 0])
        dy = 0.5 - np.abs(pts[:, 1])
        
        # If dx < r_curr, push in x direction
        # Force magnitude proportional to (r_curr - dx) or just constant push?
        # Let's use a potential V = (r_curr - d)^2 if d < r_curr, else 0.
        # Gradient is 2*(r_curr - d) * sign(d)
        
        wall_force_x = np.zeros(n)
        wall_force_y = np.zeros(n)
        
        mask_x = dx < r_curr
        # d = 0.5 - |x|
        # dd/dx = -sign(x)
        # dV/dx = 2*(r_curr - d) * (-sign(x))
        # We want to move in direction of -grad V? No, force is -grad V.
        # If we define V as penalty, force is -grad V.
        # Let's just use a simple push:
        # If x > 0 and 0.5 - x < r_curr, push left (negative x).
        # If x < 0 and 0.5 + x < r_curr, push right (positive x).
        
        # Force magnitude: k_wall * (r_curr - d)
        k_wall = 5.0
        wall_force_x[mask_x] = -k_wall * (r_curr - dx[mask_x]) * np.sign(pts[mask_x, 0])
        
        mask_y = dy < r_curr
        wall_force_y[mask_y] = -k_wall * (r_curr - dy[mask_y]) * np.sign(pts[mask_y, 1])
        
        total_forces = force_pair
        total_forces[:, 0] += wall_force_x
        total_forces[:, 1] += wall_force_y
        
        # 2. Update Positions
        # Velocity Verlet or simple Euler with damping
        # v = damping * v + step_size * F
        # p = p + v
        
        # We don't store velocity, so it's effectively:
        # p_new = p + step_size * damping * F
        # This is slow. Let's use a momentum variable.
        
        if not hasattr(_dynamics_optimize, 'momentum'):
            _dynamics_optimize.momentum = np.zeros_like(pts)
            
        momentum = _dynamics_optimize.momentum
        momentum = damping * momentum + step_size * total_forces
        pts += momentum
        
        # 3. Constraint Projection
        # Keep points inside the square
        pts[:, 0] = np.clip(pts[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
        pts[:, 1] = np.clip(pts[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
        
        # 4. Update Radius
        r_new = _compute_radius(pts)
        if r_new > r_curr:
            r_curr = r_new
            # Slow down step size as we get better?
            step_size *= 0.999
        else:
            # If radius decreased, revert or damp more?
            # For simplicity, just continue. The forces should push back.
            pass
        
        # Cool down step size over time
        step_size *= 0.9995

    _dynamics_optimize.momentum = np.zeros_like(pts)
    return pts

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
    best_pts = None
    best_r = 0.0
    
    # Multi-start
    num_starts = 10
    time_per_start = (deadline - start_time) / num_starts
    
    for i in range(num_starts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize
        pts = _initial_hex_grid(seed=i)
        
        # Optimize
        # Allocate remaining time minus a safety buffer
        alloc_time = min(time_per_start, deadline - time.time() - 2.0)
        if alloc_time < 5.0:
            break
            
        pts_opt = _dynamics_optimize(pts, time_limit=alloc_time)
        
        r = _compute_radius(pts_opt)
        if r > best_r:
            best_r = r
            best_pts = pts_opt.copy()
            
    if best_pts is None:
        # Fallback
        best_pts = _initial_hex_grid(seed=0)
        
    return best_pts
