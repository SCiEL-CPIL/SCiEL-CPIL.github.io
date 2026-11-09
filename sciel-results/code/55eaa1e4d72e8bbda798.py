"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(points):
    """Compute the feasible radius for a set of points."""
    if points is None or len(points) == 0:
        return 0.0
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dist_boundary)
    from scipy.spatial.distance import pdist
    d = pdist(points)
    min_dist = np.min(d)
    min_pair = min_dist / 2.0
    return min(min_boundary, min_pair)

def _initial_hexagonal():
    """Generate a hexagonal lattice packing as starting point."""
    est_r = np.sqrt(0.9069 / N)
    est_s = 2 * est_r
    
    best_points = None
    best_r = 0.0
    
    for spacing in np.linspace(est_s * 0.95, est_s * 1.05, 10):
        a = spacing
        h = a * np.sqrt(3) / 2
        points = []
        max_range = int(np.ceil(1.0 / h)) + 2
        for j in range(-max_range, max_range + 1):
            for i in range(-max_range, max_range + 1):
                x = (i + 0.5 * (j % 2)) * a
                y = j * h
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < N:
            continue
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        selected = points[idx]
        r = _compute_radius(selected)
        if r > best_r:
            best_r = r
            best_points = selected.copy()
    
    if best_points is None:
        best_points = np.zeros((N, 2))
        best_r = 0.01
    
    return best_points, best_r

def _ring_parameterization(points):
    """
    Convert Cartesian points to ring-based parameterization.
    Returns: list of rings, each ring is (radius, [angles])
    """
    # Compute distances from origin
    dists = np.linalg.norm(points, axis=1)
    
    # Sort by distance
    order = np.argsort(dists)
    sorted_points = points[order]
    sorted_dists = dists[order]
    
    # Group into rings based on distance
    # Use a tolerance to group points at similar radii
    rings = []
    if len(sorted_points) == 0:
        return rings
    
    current_radius = sorted_dists[0]
    current_angles = []
    
    # Calculate angles
    angles = np.arctan2(sorted_points[:, 1], sorted_points[:, 0])
    
    for i in range(len(sorted_points)):
        if i == 0:
            current_radius = sorted_dists[i]
            current_angles = [angles[i]]
        else:
            # Check if this point is close enough to current ring radius
            if abs(sorted_dists[i] - current_radius) < 0.05:  # tolerance
                current_angles.append(angles[i])
            else:
                # Save previous ring
                rings.append((current_radius, np.array(current_angles)))
                # Start new ring
                current_radius = sorted_dists[i]
                current_angles = [angles[i]]
    
    # Save last ring
    rings.append((current_radius, np.array(current_angles)))
    
    return rings

def _reconstruct_from_rings(rings):
    """
    Reconstruct Cartesian points from ring parameterization.
    rings: list of (radius, [angles])
    """
    points = []
    for radius, angles in rings:
        for angle in angles:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            # Ensure inside unit circle
            norm = np.sqrt(x*x + y*y)
            if norm > 1.0:
                x = x / norm
                y = y / norm
            points.append([x, y])
    
    return np.array(points)

def _soft_min_objective(params, rings_structure, beta=50.0):
    """
    Objective function for ring-based optimization.
    params: flattened array of all angles (radii are fixed from initialization)
    rings_structure: list of (radius, n_angles)
    """
    # Reconstruct points from current parameters
    points = []
    idx = 0
    for radius, n_angles in rings_structure:
        for j in range(n_angles):
            angle = params[idx]
            idx += 1
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    points = np.array(points)
    
    # Compute pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(points)
    
    # Compute boundary distances
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    
    # Soft min of all constraints
    all_constraints = np.concatenate([d / 2.0, dist_boundary])
    
    # Soft min approximation
    min_val = np.min(all_constraints)
    # Use soft min to make differentiable
    soft_min = -np.log(np.sum(np.exp(-beta * all_constraints))) / beta
    
    # We want to maximize the minimum, so minimize negative soft_min
    return -soft_min

def _soft_min_gradient(params, rings_structure, beta=50.0):
    """
    Gradient of the soft-min objective.
    """
    # Reconstruct points
    points = []
    idx = 0
    for radius, n_angles in rings_structure:
        for j in range(n_angles):
            angle = params[idx]
            idx += 1
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    points = np.array(points)
    n = len(points)
    
    # Compute pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(points)
    
    # Compute boundary distances
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    
    all_constraints = np.concatenate([d / 2.0, dist_boundary])
    
    # Soft min weights
    exp_vals = np.exp(-beta * all_constraints)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    # Gradient with respect to angles
    grad = np.zeros(len(params))
    
    # For each pair distance
    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            constraint_val = d[pair_idx] / 2.0
            weight = weights[pair_idx]
            
            # Distance between points i and j
            diff = points[i] - points[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                pair_idx += 1
                continue
            
            # Gradient of distance w.r.t. angle of point i
            # x_i = r_i * cos(theta_i), y_i = r_i * sin(theta_i)
            # d(dist)/d(theta_i) = (x_j - x_i) * r_i * sin(theta_i) + (y_j - y_i) * r_i * cos(theta_i) / dist
            # But we need to map back to parameter index
            
            # Find the parameter index for point i and j
            # This is complex; for simplicity, use numerical gradient or simplify
            pair_idx += 1
    
    # For boundary constraints
    for i in range(n):
        constraint_val = dist_boundary[i]
        weight = weights[n*(n-1)//2 + i]
        
        # Gradient of (1 - |p_i|) w.r.t. angle
        # |p_i| = r_i (fixed), so boundary constraint doesn't depend on angle if radius is fixed
        # But if we allow radius variation, it would. Here radii are fixed.
        # So boundary gradient w.r.t. angle is 0
    
    # Since the above is complex, use a simpler approach:
    # Just return zeros and let the optimizer handle it, or use finite differences
    
    # For this implementation, we'll use a simpler gradient approach
    # The key insight: changing angle of point i affects distances to all other points
    # and the boundary constraint
    
    # Simplified: compute gradient numerically for the first few iterations
    # But for performance, let's use an analytic approximation
    
    # Actually, let's just return the gradient of the soft min w.r.t. the constraints
    # and chain rule through the angle changes
    
    # This is getting complex. Let's use a different approach:
    # Optimize each ring's angles independently with a simpler objective
    
    return grad

def _optimize_ring_angles(points, deadline):
    """
    Optimize the angles within each ring using L-BFGS-B.
    """
    # Convert to ring parameterization
    rings = _ring_parameterization(points)
    
    # Extract initial angles and structure
    params_init = []
    rings_structure = []
    for radius, angles in rings:
        rings_structure.append((radius, len(angles)))
        params_init.extend(angles.tolist())
    
    params_init = np.array(params_init)
    
    if len(params_init) == 0:
        return points, _compute_radius(points)
    
    # Define objective
    def objective(params):
        return _soft_min_objective(params, rings_structure, beta=50.0)
    
    # Run optimization
    result = minimize(
        objective,
        params_init,
        method='L-BFGS-B',
        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    # Reconstruct points from optimized parameters
    if result.success or result.fun < objective(params_init):
        optimized_params = result.x
        points_opt = []
        idx = 0
        for radius, n_angles in rings_structure:
            for j in range(n_angles):
                angle = optimized_params[idx]
                idx += 1
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                norm = np.sqrt(x*x + y*y)
                if norm > 1.0:
                    x = x / norm
                    y = y / norm
                points_opt.append([x, y])
        points_opt = np.array(points_opt)
        r_opt = _compute_radius(points_opt)
        return points_opt, r_opt
    
    return points, _compute_radius(points)

def _optimize_radii(points, deadline):
    """
    Optimize the ring radii while keeping angles fixed.
    """
    rings = _ring_parameterization(points)
    
    # Extract radii and angles
    radii_init = []
    angles_list = []
    for radius, angles in rings:
        radii_init.append(radius)
        angles_list.append(angles)
    
    radii_init = np.array(radii_init)
    
    if len(radii_init) == 0:
        return points, _compute_radius(points)
    
    def objective(radii):
        # Reconstruct points
        points_new = []
        for i, (radius, angles) in enumerate(zip(radii, angles_list)):
            for angle in angles:
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                norm = np.sqrt(x*x + y*y)
                if norm > 1.0:
                    x = x / norm
                    y = y / norm
                points_new.append([x, y])
        
        points_new = np.array(points_new)
        r = _compute_radius(points_new)
        return -r  # Minimize negative radius
    
    # Run optimization
    result = minimize(
        objective,
        radii_init,
        method='L-BFGS-B',
        bounds=[(0, 1)] * len(radii_init),
        options={'maxiter': 50, 'ftol': 1e-10}
    )
    
    if result.success or result.fun < objective(radii_init):
        optimized_radii = result.x
        points_opt = []
        for i, (radius, angles) in enumerate(zip(optimized_radii, angles_list)):
            for angle in angles:
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                norm = np.sqrt(x*x + y*y)
                if norm > 1.0:
                    x = x / norm
                    y = y / norm
                points_opt.append([x, y])
        points_opt = np.array(points_opt)
        r_opt = _compute_radius(points_opt)
        return points_opt, r_opt
    
    return points, _compute_radius(points)

def _refine_local(points, deadline):
    """
    Local refinement using small perturbations in Cartesian space.
    """
    best_points = points.copy()
    best_r = _compute_radius(points)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(N):
            if time.time() >= deadline - 5:
                break
            for dx, dy in [(-0.0005, 0), (0.0005, 0), (0, -0.0005), (0, 0.0005),
                          (-0.0003, 0.0003), (0.0003, -0.0003)]:
                new_points = best_points.copy()
                new_points[i] = [best_points[i][0] + dx, best_points[i][1] + dy]
                if np.linalg.norm(new_points[i]) > 1.0:
                    continue
                r_new = _compute_radius(new_points)
                if r_new > best_r + 1e-10:
                    best_r = r_new
                    best_points = new_points
                    improved = True
    
    return best_points, best_r

def construct():
    """Construct the optimal packing."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Generate initial hexagonal configuration
    hex_points, hex_r = _initial_hexagonal()
    
    # Step 1: Optimize angles within rings
    if time.time() < deadline - 30:
        angle_points, angle_r = _optimize_ring_angles(hex_points, deadline)
        if angle_r > hex_r:
            hex_points, hex_r = angle_points, angle_r
    
    # Step 2: Optimize ring radii
    if time.time() < deadline - 20:
        radius_points, radius_r = _optimize_radii(hex_points, deadline)
        if radius_r > hex_r:
            hex_points, hex_r = radius_points, radius_r
    
    # Step 3: Alternate between angle and radius optimization
    for iteration in range(5):
        if time.time() >= deadline - 15:
            break
        
        # Optimize angles
        if time.time() < deadline - 15:
            angle_points, angle_r = _optimize_ring_angles(hex_points, deadline)
            if angle_r > hex_r:
                hex_points, hex_r = angle_points, angle_r
        
        # Optimize radii
        if time.time() < deadline - 10:
            radius_points, radius_r = _optimize_radii(hex_points, deadline)
            if radius_r > hex_r:
                hex_points, hex_r = radius_points, radius_r
    
    # Step 4: Final local refinement
    if time.time() < deadline - 5:
        final_points, final_r = _refine_local(hex_points, deadline)
        if final_r > hex_r:
            hex_points, hex_r = final_points, final_r
    
    return hex_points
# EVOLVE-BLOCK-END
