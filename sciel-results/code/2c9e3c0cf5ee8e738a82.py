import numpy as np
from scipy.spatial import ConvexHull
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
        
        centroid = (p1[valid] + p2[valid] + p3[valid]) / 3.0
        dot_c_centroid = np.sum(c * centroid, axis=1)
        flip = dot_c_centroid < 0
        c[flip] = -c[flip]
        
        centers[valid] = c
        dot_val = np.clip(np.sum(c * p1[valid], axis=1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        radii[valid] = np.where(dist > np.pi / 2, np.pi - dist, dist)
    
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

def _covering_radius_deg(points):
    return _covering_radius_rad(points) * 180.0 / np.pi

def _soft_max_objective_and_grad(points, beta=50.0):
    """
    Compute the soft-max (log-sum-exp) surrogate for the max circumradius
    and its gradient with respect to point positions.
    Optimized for speed.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 1.0, np.zeros_like(points)
    
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
    
    shifted = beta * (r_valid - r_max)
    log_sum_exp = r_max + np.log(np.sum(np.exp(shifted))) / beta
    
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    grad = np.zeros_like(points)
    
    # Vectorized gradient accumulation is tricky with variable indices, 
    # but N is small (130) and M is ~260, so a tight loop is acceptable 
    # if we avoid redundant calculations.
    
    for k in range(len(f_valid)):
        w_k = weights[k]
        if w_k < 1e-8: # Skip negligible weights
            continue
        
        tri = f_valid[k]
        c = c_valid[k]
        r_k = r_valid[k]
        
        if r_k < 1e-12:
            continue
        
        # Gradient of r_k w.r.t p_j for j in tri
        # d(arccos(c.p))/dp = - (c - (c.p)p) / sin(r)
        
        # We compute the gradient for the 3 vertices of this triangle
        p1 = points[tri[0]]
        p2 = points[tri[1]]
        p3 = points[tri[2]]
        
        # Calculate dot products
        d1 = np.dot(c, p1)
        d2 = np.dot(c, p2)
        d3 = np.dot(c, p3)
        
        # Calculate sin(r) for each (should be same for all 3 if c is circumcenter)
        # sin(r) = sqrt(1 - d^2)
        # But due to numerical errors, d might vary slightly. 
        # Using the average or just one is fine. Let's use the one for p1.
        sin_r_sq = 1.0 - d1 * d1
        if sin_r_sq < 1e-12:
            continue
        sin_r = np.sqrt(sin_r_sq)
        
        # Gradient vectors
        g1 = -(c - d1 * p1) / sin_r
        g2 = -(c - d2 * p2) / sin_r
        g3 = -(c - d3 * p3) / sin_r
        
        # Project to tangent plane (should be close to zero, but ensure stability)
        g1 -= np.dot(g1, p1) * p1
        g2 -= np.dot(g2, p2) * p2
        g3 -= np.dot(g3, p3) * p3
        
        # Accumulate
        grad[tri[0]] += w_k * g1
        grad[tri[1]] += w_k * g2
        grad[tri[2]] += w_k * g3
    
    return float(log_sum_exp), grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    def time_left():
        return deadline - time.time()

    rng = np.random.default_rng(42)
    
    # Phase 0: Generate initializations (Reduced to 2 to save time)
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    fib_pts = _normalize(fib_pts)
    
    # 2. Random points
    rand_pts = rng.normal(0, 1, (N, 3))
    rand_pts = _normalize(rand_pts)
    
    inits = [fib_pts, rand_pts]
    
    # Evaluate initializations and pick the best
    best_points = None
    best_radius = np.pi
    
    for init in inits:
        if time_left() < 10.0:
            break
        rad = _covering_radius_rad(init)
        if rad < best_radius:
            best_radius = rad
            best_points = init.copy()
    
    if best_points is None:
        best_points = fib_pts
        best_radius = _covering_radius_rad(fib_pts)
    
    # Phase 1: Main optimization loop
    points = best_points.copy()
    
    # Strategy parameters
    gradient_step = 0.02
    beta = 50.0
    no_improve_count = 0
    max_no_improve = 15 # Reduced to switch strategies faster
    
    # We will primarily use gradient descent. 
    # If stuck, we do a small random kick.
    
    while time_left() > 10.0:
        # Gradient descent on soft-max
        obj, grad = _soft_max_objective_and_grad(points, beta)
        
        # Project gradient to tangent plane and normalize
        grad_tangent = np.zeros_like(points)
        for i in range(N):
            g = grad[i]
            g_tan = g - np.dot(g, points[i]) * points[i]
            norm_g = np.linalg.norm(g_tan)
            if norm_g > 1e-12:
                grad_tangent[i] = g_tan / norm_g
        
        new_points = points - gradient_step * grad_tangent
        new_points = _normalize(new_points)
        
        current_radius = _covering_radius_rad(new_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points
            no_improve_count = 0
            # Slowly increase step and beta
            gradient_step = min(gradient_step * 1.01, 0.05)
            beta = min(beta * 1.01, 150.0)
        else:
            no_improve_count += 1
            gradient_step = max(gradient_step * 0.9, 1e-6)
            
        if no_improve_count > max_no_improve:
            # Random kick to escape local minimum
            kick = rng.normal(0, 0.05, points.shape)
            points = points + kick
            points = _normalize(points)
            no_improve_count = 0
            gradient_step = 0.02 # Reset step size
            
            current_radius = _covering_radius_rad(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
            
            # If we are still stuck after a kick, try increasing beta to focus on max
            if no_improve_count > max_no_improve: # This won't trigger immediately, but if we loop again
                pass 

    # Phase 2: High-precision polish
    points = best_points.copy()
    polish_step = 0.002
    polish_beta = 100.0
    no_polish_improve = 0
    
    while time_left() > 5.0 and no_polish_improve < 20:
        obj, grad = _soft_max_objective_and_grad(points, polish_beta)
        
        grad_tangent = np.zeros_like(points)
        for i in range(N):
            g = grad[i]
            g_tan = g - np.dot(g, points[i]) * points[i]
            norm_g = np.linalg.norm(g_tan)
            if norm_g > 1e-12:
                grad_tangent[i] = g_tan / norm_g
        
        new_points = points - polish_step * grad_tangent
        new_points = _normalize(new_points)
        
        current_radius = _covering_radius_rad(new_points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points
            no_polish_improve = 0
            polish_step = min(polish_step * 1.02, 0.005)
        else:
            polish_step *= 0.7
            if polish_step < 1e-8:
                break
            no_polish_improve += 1
            points = best_points.copy()

    return best_points
