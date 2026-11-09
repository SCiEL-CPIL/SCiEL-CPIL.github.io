"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(pts):
    """Compute the feasible radius for a set of points."""
    # Distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(pts, axis=1)
    min_boundary = np.min(dist_boundary)
    # Minimum pairwise distance / 2
    # Use a vectorized approach but be careful with memory for 300 points
    # 300x300 is 90k, which is fine
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_boundary, min_pair)


def _hex_lattice(n_points):
    """Generate a hexagonal lattice and select the best n_points."""
    # We want to pack n_points. Estimate spacing.
    # Area per point ~ 1 / n_points. Hex area per point = s^2 * sqrt(3)/2.
    # s^2 * sqrt(3)/2 ~ 1/n -> s ~ sqrt(2/(n*sqrt(3)))
    s = np.sqrt(2.0 / (n_points * np.sqrt(3.0)))
    # Try a few spacings around this estimate
    best_pts = None
    best_r = -1
    
    # We need to generate enough points to choose from
    # Radius of lattice needed: roughly sqrt(n_points) * s
    R = 1.0
    # Generate points in a hex grid within [-R, R]
    # Step in y: s * sqrt(3) / 2
    dy = s * np.sqrt(3.0) / 2.0
    dx = s
    
    # Number of rows
    n_rows = int(2 * R / dy) + 2
    n_cols = int(2 * R / dx) + 2
    
    # Create grid
    ys = np.linspace(-R, R, n_rows)
    xs = np.linspace(-R, R, n_cols)
    X, Y = np.meshgrid(xs, ys)
    
    # Apply hex offset
    row_idx = np.arange(n_rows)
    offset = (row_idx % 2) * (dx / 2.0)
    X = X + offset[:, np.newaxis]
    
    points = np.column_stack((X.ravel(), Y.ravel()))
    radii = np.linalg.norm(points, axis=1)
    # Keep points inside unit circle (with some margin to ensure valid radius)
    # Actually, we should keep points strictly inside to allow for radius > 0
    # But for initial generation, let's just keep them inside unit disk.
    # The radius calculation will handle the boundary constraint.
    mask = radii < 1.0
    points = points[mask]
    
    if len(points) < n_points:
        # If we don't have enough points, we need a smaller spacing
        # This is a heuristic; let's just return what we have or try again
        # For now, return the best subset if available, else empty
        if len(points) == 0:
             return np.zeros((0,2))
        # Sort by distance from center to pick the most central ones
        idx = np.argsort(radii[mask])[:n_points]
        return points[idx]
    
    # Sort by distance from center to pick the most central ones
    # This usually gives a good starting packing
    idx = np.argsort(radii[mask])[:n_points]
    return points[idx]


def _refine(pts, max_iters=5000, time_limit=100.0):
    """
    Refine the points to maximize the minimum distance (radius).
    Uses a simple gradient ascent / repulsion model.
    Objective: maximize r = min(min_pair_dist/2, min_boundary_dist)
    Equivalent to maximizing a smooth surrogate or using a repulsive force model.
    
    We will use a force-directed approach:
    - Repulsive force between all pairs: F_ij ~ 1/d^2 or similar
    - Attractive force to keep them inside: not needed if we just repel and project
    - Actually, we want to maximize the minimum distance.
    
    A common heuristic for circle packing is to use a repulsive force that is
    stronger for closer pairs.
    
    Let's use a simple iterative relaxation:
    1. Calculate forces on each point.
       - Repulsion from other points: sum over j != i of (p_i - p_j) / ||p_i - p_j||^2
         This pushes points apart.
       - Wall repulsion: if a point is close to the boundary, push it inward.
         Force = -normal * (1 - ||p||)^2 or similar.
    2. Move points by a small step in the direction of the force.
    3. Project points back inside the unit circle.
    
    To maximize the *minimum* distance, we can use a weighted repulsion
    where closer pairs exert more force.
    
    Alternatively, we can use a gradient-based method on the objective:
    f(p) = min( min_{i<j} d_ij/2, min_i (1 - ||p_i||) )
    This is non-smooth.
    
    Let's stick to the force-directed method which is robust and fast.
    We will tune the step size.
    """
    pts = pts.copy()
    n = len(pts)
    if n == 0:
        return pts
    
    start_time = time.time()
    step_size = 0.01
    decay = 0.999
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Calculate forces
        # Repulsion between points
        # diff[i, j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Avoid division by zero
        dists[np.diag_indices(n)] = np.inf
        
        # Force magnitude: 1 / dist^2
        # But we want to emphasize close pairs more? 1/d^2 does that.
        # Let's use 1/d^2.
        forces = np.zeros_like(pts)
        for i in range(n):
            # Force on i from j
            # F_ij = (pts[i] - pts[j]) / dist_ij^2
            # This is a vector sum
            # We can do this vectorized per i or fully vectorized
            pass
        
        # Vectorized force calculation
        # weights[i, j] = 1 / dists[i, j]^2
        weights = 1.0 / (dists ** 2)
        # Force on i = sum_j (pts[i] - pts[j]) * weights[i, j]
        # This is equivalent to:
        # F_i = sum_j weights[i, j] * pts[i] - sum_j weights[i, j] * pts[j]
        # F_i = pts[i] * sum_j weights[i, j] - sum_j weights[i, j] * pts[j]
        
        # Sum of weights for each i
        sum_w = np.sum(weights, axis=1)
        # Weighted sum of pts[j] for each i
        # This is a matrix multiplication: weights @ pts
        weighted_pts = weights @ pts
        
        F_rep = pts * sum_w[:, np.newaxis] - weighted_pts
        
        # Wall repulsion
        # Push points inward if they are close to the boundary
        # dist_to_wall = 1 - ||pts||
        # Force = -pts / ||pts|| * (1 - ||pts||)^2  (if ||pts|| > 0)
        # Let's use a simpler model: force is proportional to (1 - ||pts||)
        # and directed towards center.
        norms = np.linalg.norm(pts, axis=1)
        # Avoid division by zero
        safe_norms = np.where(norms > 1e-10, norms, 1.0)
        # Unit vectors from center to point
        unit_vecs = pts / safe_norms[:, np.newaxis]
        # Wall distance
        wall_dist = 1.0 - norms
        # Force magnitude: stronger when close to wall
        # Let's use (1 - wall_dist) ? No, wall_dist is small when close.
        # Force should be large when wall_dist is small.
        # F_wall = -unit_vecs * (1 - wall_dist) * strength?
        # Or F_wall = -unit_vecs * (1 / (1 + wall_dist))?
        # Let's try F_wall = -unit_vecs * (1.0 - norms) ** 2 * 10.0
        # This is zero at center, max at boundary.
        # Wait, if norms is close to 1, (1-norms) is small.
        # We want force to be large when norms is close to 1.
        # So force magnitude should be large when wall_dist is small.
        # F_wall = -unit_vecs * (1.0 / (1.0 + wall_dist))?
        # Let's use a simple linear repulsion from the wall:
        # F_wall = -unit_vecs * (1.0 - norms) * 5.0
        # This is zero at center, max at boundary.
        # But it's linear. Maybe quadratic is better to keep them away?
        # Let's use: F_wall = -unit_vecs * (1.0 - norms) * 10.0
        
        F_wall = -unit_vecs * (1.0 - norms)[:, np.newaxis] * 10.0
        
        F_total = F_rep + F_wall
        
        # Update positions
        pts += step_size * F_total
        
        # Project back to unit circle
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / norms[mask, np.newaxis]
            
        # Decay step size
        step_size *= decay
        
    return pts


def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Generate initial hex lattice
    pts = _hex_lattice(N)
    
    if len(pts) < N:
        # Fallback: simple grid or random
        # This shouldn't happen with our lattice generator
        return np.zeros((N, 2))
        
    # 2. Refine
    # Use a time limit for refinement
    # We have ~110s total. Lattice gen is fast.
    # Let's give refinement 100s.
    pts = _refine(pts, max_iters=10000, time_limit=100.0)
    
    # 3. Check time and return
    if time.time() > deadline:
         # Return whatever we have
         pass
         
    return pts

# EVOLVE-BLOCK-END
