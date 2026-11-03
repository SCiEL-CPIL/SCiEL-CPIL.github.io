import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _get_circumcenters_and_radii(points, hull):
    """
    Compute the circumcenters and circumradii for all facets of the convex hull.
    Returns:
        centers: array of shape (M, 3) with circumcenters on the unit sphere.
        radii: array of shape (M,) with angular circumradii.
        facets: array of shape (M, 3) with indices of vertices for each facet.
    """
    facets = hull.simplices
    M = len(facets)
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    for k in range(M):
        tri = facets[k]
        p1 = points[tri[0]]
        p2 = points[tri[1]]
        p3 = points[tri[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            radii[k] = 0.0
            centers[k] = p1
            continue
            
        c = cross / norm_cross
        
        # Ensure the circumcenter is on the correct side (outward normal)
        # For a convex hull containing the origin, the outward normal should
        # have positive dot product with the centroid of the triangle
        centroid = (p1 + p2 + p3) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
            
        dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        
        radii[k] = dist
        centers[k] = c
        
    return centers, radii, facets

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    """
    try:
        hull = ConvexHull(points)
        centers, radii, _ = _get_circumcenters_and_radii(points, hull)
        return np.degrees(np.max(radii))
    except Exception:
        return 180.0

def _soft_max(radii, beta=100.0):
    """
    Soft-max approximation of max using log-sum-exp.
    As beta -> infinity, this approaches the true max.
    """
    # Numerically stable log-sum-exp
    max_r = np.max(radii)
    if max_r < -700:  # Avoid underflow
        return max_r
    shifted = radii - max_r
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return max_r + log_sum

def _compute_gradients(points, centers, radii, facets, beta=100.0):
    """
    Compute the gradient of the soft-max covering radius with respect to point positions.
    
    The soft-max is: S = max_r + (1/beta) * log(sum(exp(beta * (r_k - max_r))))
    
    dS/dr_k = (1/beta) * exp(beta * (r_k - max_r)) / sum(exp(beta * (r_j - max_r)))
    
    Then we need dr_k/dp_i for each point p_i that is a vertex of facet k.
    
    For a spherical triangle with circumcenter c and circumradius r:
    r = arccos(c · p1)
    dr/dp1 = -c / sqrt(1 - (c·p1)^2)  (tangent component)
    
    But we need to be careful: c itself depends on p1, p2, p3.
    For simplicity and stability, we'll use a finite-difference-like approach
    or approximate the gradient assuming c is fixed (first-order).
    
    Actually, let's use a simpler approach: the gradient of r_k with respect to p_i
    (where p_i is a vertex of facet k) is approximately:
    -c / sin(r_k)  projected onto the tangent plane at p_i
    
    This is because r = arccos(c·p), so dr/dp = -c/sin(r) (in the tangent space).
    """
    M = len(facets)
    grads = np.zeros_like(points)
    
    if M == 0:
        return grads
    
    max_r = np.max(radii)
    # Compute weights for soft-max
    shifted = radii - max_r
    exp_shifted = np.exp(beta * shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    # For each facet, compute the contribution to the gradient
    for k in range(M):
        w = weights[k]
        if w < 1e-10:
            continue
            
        r_k = radii[k]
        c = centers[k]
        tri = facets[k]
        
        # For each vertex of this facet, compute the gradient contribution
        for idx in tri:
            p = points[idx]
            
            # dr/dp = -c / sin(r) in the tangent plane
            # sin(r) = sqrt(1 - cos(r)^2) = sqrt(1 - (c·p)^2)
            dot_cp = np.dot(c, p)
            sin_r = np.sqrt(max(1e-12, 1.0 - dot_cp * dot_cp))
            
            # Gradient of r with respect to p (in R^3)
            grad_r = -c / sin_r
            
            # Project to tangent plane at p
            grad_tangent = grad_r - np.dot(grad_r, p) * p
            
            # Accumulate weighted gradient
            grads[idx] += w * grad_tangent
    
    return grads

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    points += rng.normal(0, 0.01, points.shape)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Optimization parameters
    step_size = 0.02
    min_step_size = 1e-7
    beta = 50.0  # Soft-max sharpness
    stagnant_count = 0
    max_stagnant = 30
    polish_phase = False
    
    while time.time() < deadline:
        
        try:
            hull = ConvexHull(points)
            centers, radii, facets = _get_circumcenters_and_radii(points, hull)
        except Exception:
            # If hull fails, add noise and retry
            points += rng.normal(0, 0.01, points.shape)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue

        current_radius_rad = np.max(radii)
        current_radius_deg = np.degrees(current_radius_rad)
        
        if current_radius_deg < best_radius - 1e-9:
            best_radius = current_radius_deg
            best_points = points.copy()
            stagnant_count = 0
        else:
            stagnant_count += 1
            
            if stagnant_count > max_stagnant:
                # Random kick to escape local minimum
                kick = rng.normal(0, 0.03, points.shape)
                points += kick
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                step_size = 0.02
                stagnant_count = 0
                continue
        
        # Check if we should enter polish phase
        if best_radius < 11.35 and not polish_phase:
            polish_phase = True
            step_size = 0.005
            beta = 100.0
        
        # Compute gradients using soft-max
        grads = _compute_gradients(points, centers, radii, facets, beta)
        
        # Project gradients to tangent plane (should already be tangent, but ensure)
        dot_fn = np.sum(grads * points, axis=1, keepdims=True)
        tangent_grads = grads - dot_fn * points
        
        # Normalize gradient magnitude for consistent step size
        grad_norm = np.linalg.norm(tangent_grads, axis=1)
        max_grad = np.max(grad_norm)
        if max_grad > 1e-12:
            tangent_grads = tangent_grads / max_grad
        
        # Line search: try a few step sizes
        best_trial = None
        best_trial_radius = current_radius_deg
        
        for factor in [1.0, 0.5, 0.25, 0.1]:
            trial_step = step_size * factor
            trial_points = points - trial_step * tangent_grads
            trial_points /= np.linalg.norm(trial_points, axis=1, keepdims=True)
            
            if not np.all(np.isfinite(trial_points)):
                continue
            
            try:
                trial_hull = ConvexHull(trial_points)
                trial_centers, trial_radii, _ = _get_circumcenters_and_radii(trial_points, trial_hull)
                trial_radius = np.degrees(np.max(trial_radii))
            except Exception:
                continue
            
            if trial_radius < best_trial_radius:
                best_trial_radius = trial_radius
                best_trial = trial_points
        
        if best_trial is not None:
            points = best_trial
            stagnant_count = 0
            # Reduce step size slightly for next iteration
            step_size = max(step_size * 0.99, min_step_size)
        else:
            # No improvement, reduce step size
            step_size = max(step_size * 0.9, min_step_size)
            stagnant_count += 1
            
            if stagnant_count > max_stagnant:
                # Random kick
                kick = rng.normal(0, 0.02, points.shape)
                points += kick
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                step_size = 0.01
                stagnant_count = 0
    
    # Final polish: a few iterations with very small steps
    if time.time() < deadline - 5.0:
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            try:
                hull = ConvexHull(points)
                centers, radii, facets = _get_circumcenters_and_radii(points, hull)
            except Exception:
                break
                
            grads = _compute_gradients(points, centers, radii, facets, beta=200.0)
            dot_fn = np.sum(grads * points, axis=1, keepdims=True)
            tangent_grads = grads - dot_fn * points
            
            grad_norm = np.linalg.norm(tangent_grads, axis=1)
            max_grad = np.max(grad_norm)
            if max_grad < 1e-12:
                break
            tangent_grads = tangent_grads / max_grad
            
            trial_points = points - 0.001 * tangent_grads
            trial_points /= np.linalg.norm(trial_points, axis=1, keepdims=True)
            
            if not np.all(np.isfinite(trial_points)):
                break
            
            try:
                trial_hull = ConvexHull(trial_points)
                trial_centers, trial_radii, _ = _get_circumcenters_and_radii(trial_points, trial_hull)
                trial_radius = np.degrees(np.max(trial_radii))
            except Exception:
                break
            
            if trial_radius < _compute_covering_radius(points) - 1e-10:
                points = trial_points
    
    return best_points
