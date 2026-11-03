import numpy as np
import time
from scipy.optimize import minimize

N = 200
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin

def get_hex_grid(n_points):
    """Generate a hexagonal-like grid of points in [-0.5, 0.5]^2."""
    # Approximate number of points per side
    side = int(np.ceil(np.sqrt(n_points)))
    # Generate a slightly larger grid to ensure we have enough points
    # Hex packing density is higher, so we can use a tighter grid
    # Let's just generate a square grid first, then perturb to hex? 
    # No, let's generate a proper hex grid.
    
    # Spacing s. Area per point ~ s^2 * sqrt(3)/2.
    # Total area 1. N=200.
    # s ~ sqrt(1 / (200 * sqrt(3)/2)) ~ sqrt(1/173) ~ 0.076
    # Let's try s = 0.08
    s = 0.08
    points = []
    y = -0.5
    row = 0
    while len(points) < n_points + 50: # Generate extra
        x_offset = (s / 2) if row % 2 == 1 else 0
        x = -0.5 + x_offset
        while x < 0.5 and len(points) < n_points + 50:
            points.append([x, y])
            x += s
        y += s * np.sqrt(3) / 2
        row += 1
    
    pts = np.array(points)
    # Clip to square
    mask = (pts[:, 0] >= -0.5) & (pts[:, 0] <= 0.5) & (pts[:, 1] >= -0.5) & (pts[:, 1] <= 0.5)
    pts = pts[mask]
    
    # If we have more than n_points, select the ones closest to the center
    if len(pts) > n_points:
        dists = np.sum(pts**2, axis=1)
        indices = np.argsort(dists)
        pts = pts[indices[:n_points]]
    else:
        # If we have less, add random points
        missing = n_points - len(pts)
        random_pts = np.random.uniform(-0.5, 0.5, (missing, 2))
        pts = np.vstack([pts, random_pts])
        
    return pts

def calculate_radius(positions):
    """Calculate the max valid radius for given positions."""
    # Pairwise distances
    # Use scipy.spatial.distance.pdist for efficiency? 
    # For N=200, pdist is fast.
    from scipy.spatial.distance import pdist
    d = pdist(positions)
    min_d = np.min(d) if len(d) > 0 else np.inf
    
    # Boundary distances
    # dist to boundary = min(0.5 - |x|, 0.5 - |y|)
    x = positions[:, 0]
    y = positions[:, 1]
    dx = 0.5 - np.abs(x)
    dy = 0.5 - np.abs(y)
    boundary_dists = np.minimum(dx, dy)
    min_b = np.min(boundary_dists)
    
    r = np.min([min_d / 2.0, min_b])
    return r

def energy_func(params, r_target):
    """
    Energy function for fixed r_target.
    params: (2N,) array of x1, y1, ..., xN, yN
    """
    positions = params.reshape(-1, 2)
    
    # Pairwise distances
    # Compute all pairwise distances
    # d_ij = sqrt((xi-xj)^2 + (yi-yj)^2)
    # Violation = max(0, 2*r_target - d_ij)^2
    # We can vectorize this.
    
    # To avoid O(N^2) memory for large N (though 200 is small), 
    # we can just use a loop or broadcasting.
    # Broadcasting: (N, 1, 2) - (1, N, 2) -> (N, N, 2)
    # This creates a 200x200x2 array, which is 80k floats, very manageable.
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq)
    
    # Mask diagonal (self distance)
    np.fill_diagonal(dist, np.inf)
    
    # Overlap violations
    # We want d >= 2*r_target
    # violation = max(0, 2*r_target - d)
    min_dist_required = 2 * r_target
    violations = np.maximum(0, min_dist_required - dist)
    overlap_energy = np.sum(violations**2)
    
    # Boundary violations
    x = positions[:, 0]
    y = positions[:, 1]
    dx = 0.5 - np.abs(x)
    dy = 0.5 - np.abs(y)
    boundary_dists = np.minimum(dx, dy)
    
    # We want boundary_dist >= r_target
    b_violations = np.maximum(0, r_target - boundary_dists)
    boundary_energy = np.sum(b_violations**2)
    
    return overlap_energy + boundary_energy

def optimize_packing(r_target, init_positions):
    """
    Run local optimization to minimize energy for a fixed r_target.
    Returns (positions, energy).
    """
    x0 = init_positions.flatten()
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Use L-BFGS-B
    try:
        res = minimize(
            energy_func, 
            x0, 
            args=(r_target,), 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return res.x.reshape(-1, 2), res.fun
    except Exception as e:
        # Fallback to Nelder-Mead if L-BFGS-B fails?
        try:
            res = minimize(
                energy_func, 
                x0, 
                args=(r_target,), 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-12}
            )
            return res.x.reshape(-1, 2), res.fun
        except Exception:
            return init_positions, np.inf

def construct():
    global START_TIME
    best_positions = None
    best_radius = 0.0
    
    # Initial guess
    init_pos = get_hex_grid(N)
    
    # Start with a conservative radius
    # Grid radius is approx 0.03 (spacing 0.08 -> r=0.04? No, hex spacing 0.08 -> min dist 0.08 -> r=0.04)
    # Let's start at 0.03 to be safe.
    r_current = 0.03
    
    # Incremental radius increase
    r_step = 0.0002
    
    # We will try a few restarts
    for restart in range(3):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Perturb initial positions for restarts > 0
        if restart == 0:
            pos = get_hex_grid(N)
        else:
            # Add noise to the previous best or a new grid
            pos = get_hex_grid(N) + np.random.normal(0, 0.01, (N, 2))
            # Clip to square
            pos[:, 0] = np.clip(pos[:, 0], -0.49, 0.49)
            pos[:, 1] = np.clip(pos[:, 1], -0.49, 0.49)
            
        r_current = 0.03
        
        while True:
            if time.time() - START_TIME > TIME_LIMIT:
                break
                
            # Optimize for current r_current
            pos, energy = optimize_packing(r_current, pos)
            
            # Check if we found a valid packing (energy close to 0)
            # Also calculate the actual max radius of this configuration
            actual_r = calculate_radius(pos)
            
            if actual_r > best_radius:
                best_radius = actual_r
                best_positions = pos.copy()
                
            # If energy is very low, we can increase r
            if energy < 1e-8:
                r_current += r_step
                # Cap r_current to theoretical max to avoid infinite loop
                if r_current > 0.037:
                    break
            else:
                # If energy is high, we might be stuck or r is too high
                # Try to decrease r slightly if we've been failing? 
                # Or just break and rely on the next restart
                # Let's try to polish the current best radius
                break
                
        # If we broke due to time, stop
        if time.time() - START_TIME > TIME_LIMIT:
            break

    # Final polish on the best found positions with its radius
    if best_positions is not None:
        r_final = calculate_radius(best_positions)
        # One more optimization step with this r_final
        pos, energy = optimize_packing(r_final, best_positions)
        final_r = calculate_radius(pos)
        if final_r > best_radius:
            best_radius = final_r
            best_positions = pos

    if best_positions is None:
        # Fallback to grid
        best_positions = get_hex_grid(N)
        
    return best_positions
