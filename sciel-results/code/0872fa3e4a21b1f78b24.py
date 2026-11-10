import numpy as np
import time

N = 50

def _calc_min_radius(points):
    # min distance between pairs
    # min distance to boundary
    # points is (N, 2)
    dists = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    min_boundary_dist = np.min(1.0 - radii)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def _compute_gradient(points, r_target):
    # We want to move points to increase the minimum radius.
    # This is equivalent to resolving violations for a slightly larger r.
    # Or simply, compute gradient of the "violation" energy for r = current_r + epsilon.
    
    # Let's use the violation energy gradient approach.
    # Energy E = sum max(0, 2r - d_ij)^2 + sum max(0, |c_i| - (1-r))^2
    # We want to minimize E for a fixed r.
    # If E > 0, we have violations.
    # If E == 0, we are feasible.
    
    # To find the direction to increase r, we can look at the gradient of the active constraints.
    # Let d_min = min d_ij. Let b_min = min (1 - |c_i|).
    # r = min(d_min/2, b_min).
    
    # If d_min/2 < b_min:
    #   The binding constraint is the pair distance.
    #   We want to increase d_min.
    #   Gradient direction is sum of (c_i - c_j)/d_ij for active pairs.
    # If b_min < d_min/2:
    #   The binding constraint is the boundary.
    #   We want to increase b_min (move inwards).
    #   Gradient direction is sum of -c_i/|c_i| for active points.
    # If both:
    #   Combine them.
    
    dists = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    boundary_dists = 1.0 - radii
    
    d_min = np.min(dists)
    b_min = np.min(boundary_dists)
    
    r = min(d_min / 2.0, b_min)
    
    grad = np.zeros_like(points)
    
    # Threshold for "active"
    tol = 1e-6
    
    if d_min / 2.0 <= b_min + tol:
        # Pair constraints are active or co-active
        # Find all pairs with dist <= d_min + tol
        # This is O(N^2)
        for i in range(N):
            for j in range(i + 1, N):
                if dists[i, j] <= d_min + tol:
                    d = dists[i, j]
                    if d < 1e-10: continue
                    # Gradient of d_ij wrt c_i is (c_i - c_j) / d
                    # We want to increase d, so move c_i away from c_j
                    v = (points[i] - points[j]) / d
                    grad[i] += v
                    grad[j] -= v
                    
    if b_min <= d_min / 2.0 + tol:
        # Boundary constraints are active
        for i in range(N):
            if boundary_dists[i] <= b_min + tol:
                r_i = radii[i]
                if r_i < 1e-10: continue
                # Gradient of b_i = 1 - |c_i| wrt c_i is -c_i / |c_i|
                # We want to increase b_i, so move in direction -c_i/|c_i|
                v = -points[i] / r_i
                grad[i] += v
                
    # Normalize gradient to unit length for stable stepping
    norm = np.linalg.norm(grad)
    if norm > 1e-10:
        grad /= norm
        
    return grad, r

def construct():
    start_time = time.time()
    deadline = 110.0 # 120s limit, keep margin
    
    # 1. Initialization: Hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            points = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98
        
    current_r = _calc_min_radius(points)
    best_points = points.copy()
    best_r = current_r
    
    step_size = 1e-4
    
    while time.time() - start_time < deadline:
        grad, current_r = _compute_gradient(points, current_r)
        
        if np.linalg.norm(grad) < 1e-10:
            break
            
        # Try a step
        new_points = points + step_size * grad
        
        # Check if new points are valid (inside circle, no overlaps)
        # Actually, the gradient points in the direction of increasing min radius.
        # So new_points should have a larger min radius if step is small enough.
        
        new_r = _calc_min_radius(new_points)
        
        if new_r > current_r:
            points = new_points
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
            step_size *= 1.05 # Increase step if improving
            if step_size > 1e-2:
                step_size = 1e-2
        else:
            step_size *= 0.8 # Decrease step if not improving
            if step_size < 1e-7:
                break
                
    return best_points
