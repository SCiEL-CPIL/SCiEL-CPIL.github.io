import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def compute_covering_radius_and_facets(points):
    """
    Computes the covering radius and returns the radii and indices for each facet.
    points: (N, 3) array of unit vectors.
    Returns (max_radius, radii, simplices, centers)
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return np.pi, np.array([]), np.array([]), np.array([])
    
    if len(simplices) == 0:
        return np.pi, np.array([]), np.array([]), np.array([])
        
    # Extract vertices for all facets
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    # Compute normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    # Avoid division by zero
    valid = norms > 1e-10
    if not np.all(valid):
        norms[~valid] = 1.0
        n[~valid] = [0, 0, 1]
        
    u = n / norms[:, np.newaxis]
    
    # Orient normals to point outwards (away from origin)
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius is angle between u and a
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    max_radius = np.max(radii)
    return max_radius, radii, simplices, u

def _facet_circumradii_with_fixed_simplices(points, simplices):
    """
    Compute circumradii for a fixed set of simplices (triangles on sphere).
    Returns (radii, centers) where radii is (M,) and centers is (M, 3).
    If a triangle is degenerate or invalid, radius is set to pi.
    """
    M = len(simplices)
    radii = np.full(M, np.pi)
    centers = np.zeros((M, 3))
    
    if M == 0:
        return radii, centers
    
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    # Compute normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    valid = norms > 1e-8
    if not np.any(valid):
        return radii, centers
    
    u = np.zeros_like(n)
    u[valid] = n[valid] / norms[valid, np.newaxis]
    
    # Orient normals to point outwards
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    centers = u
    
    # Circumradius is angle between u and a
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii[valid] = np.arccos(cos_theta[valid])
    
    return radii, centers

def _smooth_surrogate(points, simplices, beta=50.0):
    """
    Smooth surrogate for max circumradius using log-sum-exp.
    Returns (value, gradient) where value is the smooth max and gradient is (N, 3).
    """
    radii, centers = _facet_circumradii_with_fixed_simplices(points, simplices)
    M = len(radii)
    
    if M == 0:
        return np.pi, np.zeros_like(points)
    
    # Log-sum-exp for smooth max
    max_r = np.max(radii)
    # Shift for numerical stability
    shifted = beta * (radii - max_r)
    exp_shifted = np.exp(shifted)
    log_sum = np.log(np.sum(exp_shifted))
    
    value = max_r + (1.0 / beta) * log_sum
    
    # Gradient of log-sum-exp: d/dx_i = exp(beta*x_i) / sum(exp(beta*x_j))
    weights = exp_shifted / np.sum(exp_shifted)
    
    # Now compute gradient with respect to point positions
    gradient = np.zeros_like(points)
    
    for m in range(M):
        if weights[m] < 1e-10:
            continue
        
        simplex = simplices[m]
        i_idx, j_idx, k_idx = simplex
        
        r_m = radii[m]
        if r_m < 1e-8 or r_m > np.pi - 1e-8:
            continue
        
        sin_r = np.sin(r_m)
        center = centers[m]
        
        # Gradient contribution from each vertex
        # For a triangle on the sphere, the circumcenter is determined by the three vertices
        # A simplified gradient: move each vertex away from the center if the facet is large
        grad_i = -sin_r * center
        grad_j = -sin_r * center
        grad_k = -sin_r * center
        
        weight = weights[m] / beta
        
        gradient[i_idx] += weight * grad_i
        gradient[j_idx] += weight * grad_j
        gradient[k_idx] += weight * grad_k
    
    # Project gradient to tangent space (remove radial component)
    for i in range(N):
        p = points[i]
        radial = np.dot(gradient[i], p)
        gradient[i] -= radial * p
    
    return value, gradient

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return points / norms

def _lloyd_step(points, simplices, centers, radii, alpha=0.1):
    """
    Perform a Lloyd-like step: move each point toward the circumcenter
    of its largest adjacent facet, weighted by the facet's radius.
    """
    N = len(points)
    updates = np.zeros_like(points)
    
    # For each point, accumulate weighted centroid of adjacent facet centers
    # Weight by the facet's circumradius (larger facets pull more)
    for m in range(len(simplices)):
        simplex = simplices[m]
        i_idx, j_idx, k_idx = simplex
        
        r_m = radii[m]
        center = centers[m]
        
        # Weight: larger radius means more pull
        w = r_m / max(np.max(radii), 1e-10)
        
        updates[i_idx] += w * (center - points[i_idx])
        updates[j_idx] += w * (center - points[j_idx])
        updates[k_idx] += w * (center - points[k_idx])
    
    # Normalize updates and apply
    for i in range(N):
        norm_u = np.linalg.norm(updates[i])
        if norm_u > 1e-10:
            updates[i] /= norm_u
    
    points_new = points + alpha * updates
    return _normalize(points_new)

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    best_points = None
    best_radius = np.pi
    
    # Initial configuration: Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Get initial hull structure
    max_radius, radii, simplices, centers = compute_covering_radius_and_facets(init_fib)
    
    if len(simplices) == 0:
        return init_fib
    
    M = len(simplices)
    
    # Phase 1: Lloyd-like iterations with periodic hull updates
    points = init_fib.copy()
    alpha = 0.05  # Step size for Lloyd step
    
    for iteration in range(3000):
        if time.time() - start_time > time_limit - 30.0:
            break
        
        # Compute current radii and centers
        radii, centers = _facet_circumradii_with_fixed_simplices(points, simplices)
        
        # Perform Lloyd step
        points_new = _lloyd_step(points, simplices, centers, radii, alpha)
        
        # Check if this improves the covering radius
        new_max, new_radii, new_simplices, new_centers = compute_covering_radius_and_facets(points_new)
        
        if new_max < best_radius:
            best_radius = new_max
            best_points = points_new.copy()
        
        # Update structure if changed significantly
        if len(new_simplices) > 0:
            if len(new_simplices) != M or np.mean(np.abs(new_radii - radii)) > 0.03:
                simplices = new_simplices
                M = len(simplices)
        
        points = points_new
        
        # Adapt step size
        if iteration % 100 == 0:
            alpha *= 0.97
    
    # Phase 2: Gradient-based optimization with smooth surrogate
    if time.time() - start_time < time_limit - 15.0:
        points = best_points.copy() if best_points is not None else points.copy()
        
        # Use L-BFGS-B with the smooth surrogate
        def objective_and_grad(flat_params):
            pts = flat_params.reshape(N, 3)
            pts = _normalize(pts)
            
            # Recompute hull if structure might have changed
            max_r, rad, simp, cent = compute_covering_radius_and_facets(pts)
            
            if len(simp) == 0:
                return np.pi, np.zeros(N * 3)
            
            value, grad = _smooth_surrogate(pts, simp, beta=80.0)
            return value, grad.reshape(-1)
        
        x0 = points.flatten()
        result = minimize(objective_and_grad, x0, method='L-BFGS-B',
                         jac=True, options={'maxiter': 200, 'gtol': 1e-6})
        
        if result.fun < best_radius:
            best_radius = result.fun
            best_points = _normalize(result.x.reshape(N, 3))
    
    # Phase 3: Final polishing - target worst facets directly
    if time.time() - start_time < time_limit - 5.0 and best_points is not None:
        polish_points = best_points.copy()
        
        alpha = 0.01
        for step in range(300):
            if time.time() - start_time > time_limit - 2.0:
                break
            
            max_r, radii_p, simplices_p, centers_p = compute_covering_radius_and_facets(polish_points)
            
            if len(simplices_p) == 0:
                break
            
            if max_r < best_radius:
                best_radius = max_r
                best_points = polish_points.copy()
            
            # Move vertices of largest facets toward circumcenters
            threshold = np.percentile(radii_p, 60)
            mask = radii_p > threshold
            
            if not np.any(mask):
                max_idx = np.argmax(radii_p)
                mask = np.zeros(len(radii_p), dtype=bool)
                mask[max_idx] = True
            
            updates = np.zeros_like(polish_points)
            
            for m in np.where(mask)[0]:
                simplex = simplices_p[m]
                i_idx, j_idx, k_idx = simplex
                
                a = polish_points[i_idx]
                b = polish_points[j_idx]
                c = polish_points[k_idx]
                
                center = centers_p[m]
                
                step_size = alpha * radii_p[m]
                
                updates[i_idx] += step_size * (center - a)
                updates[j_idx] += step_size * (center - b)
                updates[k_idx] += step_size * (center - c)
            
            polish_points += updates
            polish_points = _normalize(polish_points)
    
    if best_points is None:
        best_points = init_fib
    
    return best_points
