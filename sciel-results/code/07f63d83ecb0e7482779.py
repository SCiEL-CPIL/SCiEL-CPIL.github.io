import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms

def _facet_circumradii(points, simplices):
    """
    Compute circumradii and circumcenters for all hull facets (vectorized).
    Returns:
        radii: (M,) array of angular circumradii
        centers: (M, 3) array of circumcenters on sphere
        mask: (M,) boolean array of valid facets
    """
    M = len(simplices)
    p1 = points[simplices[:, 0]]
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    norm_cross = np.linalg.norm(cross, axis=1)
    
    valid = norm_cross > 1e-14
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    if np.any(valid):
        c = cross[valid] / norm_cross[valid, np.newaxis]
        
        # Ensure the circumcenter is on the same side of the plane as the centroid
        # (i.e., pointing outwards from the origin for a convex hull containing origin)
        centroid = (p1[valid] + p2[valid] + p3[valid]) / 3.0
        dot_c_centroid = np.sum(c * centroid, axis=1)
        flip = dot_c_centroid < 0
        c[flip] = -c[flip]
        
        centers[valid] = c
        
        # Calculate angular distance
        dot_val = np.clip(np.sum(c * p1[valid], axis=1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        # The covering radius is the max angular distance.
        # For a convex hull of points on the sphere, the "cap" is the region outside the triangle.
        # The circumcenter of the triangle on the sphere is the center of the minimal cap covering the triangle.
        # The radius is the distance from the circumcenter to the vertices.
        radii[valid] = dist
    
    return radii, centers, valid

def _covering_radius_rad(points):
    """
    Compute the covering radius of a set of points on the unit sphere (in radians).
    """
    try:
        hull = ConvexHull(points)
        radii, centers, valid = _facet_circumradii(points, hull.simplices)
        if not np.any(valid):
            return np.pi
        return float(np.max(radii[valid]))
    except Exception:
        return np.pi

def _smooth_max_objective_and_grad(points, beta=10.0):
    """
    Compute the soft-max (log-sum-exp) surrogate for the max circumradius
    and its gradient with respect to point positions.
    
    Returns:
        obj: float value of the soft-max objective
        grad: (N, 3) array of gradients
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull fails, return a large penalty and zero gradient
        return 10.0, np.zeros_like(points)
    
    radii, centers, valid = _facet_circumradii(points, hull.simplices)
    M = len(radii)
    
    if M == 0 or not np.any(valid):
        return 0.0, np.zeros_like(points)
    
    # Use only valid facets
    r_valid = radii[valid]
    c_valid = centers[valid]
    f_valid = hull.simplices[valid]
    
    if len(r_valid) == 0:
        return 0.0, np.zeros_like(points)
    
    r_max = np.max(r_valid)
    if r_max < 1e-15:
        return 0.0, np.zeros_like(points)
    
    # Log-sum-exp trick for stability
    shifted = beta * (r_valid - r_max)
    exps = np.exp(shifted)
    log_sum_exp = r_max + np.log(np.sum(exps)) / beta
    
    weights = exps / np.sum(exps)
    
    grad = np.zeros_like(points)
    
    for k in range(len(f_valid)):
        w_k = weights[k]
        if w_k < 1e-8:
            continue
        
        tri = f_valid[k]
        c = c_valid[k]
        r_k = r_valid[k]
        
        if r_k < 1e-12:
            continue
        
        # Gradient of r = arccos(c . p) w.r.t. p
        # d/dp arccos(c.p) = - (c - (c.p)p) / sqrt(1 - (c.p)^2)
        # Note: (c - (c.p)p) is the tangent component of c.
        # Its norm is sin(r).
        # So the gradient is - (c - (c.p)p) / sin(r)
        
        for idx in tri:
            p = points[idx]
            dot_cp = np.dot(c, p)
            sin_r = np.sqrt(max(0.0, 1.0 - dot_cp * dot_cp))
            if sin_r > 1e-12:
                # Direction to move p to decrease r is towards c (tangent component)
                # grad_r = - (c - dot_cp * p) / sin_r
                # We want to minimize, so we move in direction -grad_r = (c - dot_cp * p) / sin_r
                # But we are adding to grad, and later doing points - lr * grad.
                # So grad should be d(objective)/dp.
                # d(objective)/dp = sum w_k * d(r_k)/dp
                # d(r_k)/dp = - (c - dot_cp * p) / sin_r
                grad_p = -(c - dot_cp * p) / sin_r
                # Project to tangent plane (should be, but just in case)
                grad_p -= np.dot(grad_p, p) * p
                grad[idx] += w_k * grad_p
    
    return float(log_sum_exp), grad

def _nelder_mead_polish(points, max_facets=20, max_iter=100, time_limit=5.0):
    """
    Polish using Nelder-Mead on the vertices of the largest facets.
    """
    start = time.time()
    best_points = points.copy()
    best_radius = _covering_radius_rad(points)
    
    try:
        hull = ConvexHull(points)
        radii, centers, valid = _facet_circumradii(points, hull.simplices)
        if not np.any(valid):
            return best_points
    except Exception:
        return best_points
    
    r_valid = radii[valid]
    f_valid = hull.simplices[valid]
    
    if len(r_valid) == 0:
        return best_points
    
    # Select top K facets
    top_k = min(max_facets, len(r_valid))
    top_indices = np.argsort(r_valid)[-top_k:]
    
    # Get unique vertices involved in these facets
    vertex_indices = set()
    for idx in top_indices:
        for v in f_valid[idx]:
            vertex_indices.add(v)
    
    vertex_indices = list(vertex_indices)
    
    if len(vertex_indices) == 0:
        return best_points
    
    def objective(x_flat):
        if time.time() - start > time_limit:
            return np.pi
        x = x_flat.reshape(-1, 3)
        new_points = points.copy()
        for i, idx in enumerate(vertex_indices):
            new_points[idx] = x[i]
        new_points = _normalize(new_points)
        if not np.all(np.isfinite(new_points)):
            return np.pi
        return _covering_radius_rad(new_points)
    
    x0 = points[vertex_indices].flatten()
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
        if result.fun < best_radius:
            x_opt = result.x.reshape(-1, 3)
            best_points = points.copy()
            for i, idx in enumerate(vertex_indices):
                best_points[idx] = x_opt[i]
            best_points = _normalize(best_points)
            best_radius = result.fun
    except Exception:
        pass
    
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def time_left():
        return deadline - time.time()

    rng = np.random.default_rng(42)
    
    # Phase 0: Initialization
    # Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    fib_pts = _normalize(fib_pts)
    
    points = fib_pts.copy()
    best_points = points.copy()
    best_radius = _covering_radius_rad(points)
    
    # Phase 1: Gradient Descent with Soft-Max Objective
    beta = 5.0  # Start with low beta (smoother), increase later
    lr = 0.01
    no_improve_count = 0
    max_no_improve = 20
    
    while time_left() > 30.0:
        obj, grad = _smooth_max_objective_and_grad(points, beta)
        
        if not np.all(np.isfinite(obj)):
            # Reset if something went wrong
            points = best_points.copy()
            beta = 5.0
            lr = 0.01
            continue
            
        # Project gradient to tangent space
        grad_tangent = np.zeros_like(points)
        for i in range(N):
            g = grad[i]
            g_tan = g - np.dot(g, points[i]) * points[i]
            norm_g = np.linalg.norm(g_tan)
            if norm_g > 1e-12:
                grad_tangent[i] = g_tan / norm_g # Normalize gradient for stable step size
        
        # Take a step
        new_points = points - lr * grad_tangent
        new_points = _normalize(new_points)
        
        # Evaluate new points
        current_radius = _covering_radius_rad(new_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points
            no_improve_count = 0
            # Increase beta to sharpen the max
            beta = min(beta * 1.05, 50.0)
            # Slightly increase learning rate if stable
            if no_improve_count == 0 and lr < 0.02:
                 lr *= 1.02
        else:
            no_improve_count += 1
            lr *= 0.8
            if no_improve_count > max_no_improve:
                # Restart with higher beta or kick
                points = best_points.copy()
                beta = min(beta * 1.5, 100.0)
                lr = 0.005
                no_improve_count = 0
                
                # Occasionally kick to escape local minima
                if no_improve_count % 2 == 0:
                     kick = rng.normal(0, 0.01, points.shape)
                     points += kick
                     points = _normalize(points)
                     current_radius = _covering_radius_rad(points)
                     if current_radius < best_radius:
                         best_radius = current_radius
                         best_points = points.copy()

    # Phase 2: Polishing with Nelder-Mead on worst facets
    points = best_points.copy()
    
    # Multiple rounds of polishing
    polish_time_budget = time_left()
    if polish_time_budget > 10.0:
        # First round: Top 10 facets
        points = _nelder_mead_polish(points, max_facets=10, max_iter=50, time_limit=min(10.0, polish_time_budget * 0.4))
        current_radius = _covering_radius_rad(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
    if time_left() > 10.0:
        # Second round: Top 20 facets, more iterations
        points = best_points.copy()
        points = _nelder_mead_polish(points, max_facets=20, max_iter=100, time_limit=min(20.0, time_left() - 5.0))
        current_radius = _covering_radius_rad(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()

    # Final check
    return best_points
