import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _compute_facet_radii(pts):
    """
    Compute the circumradius (in radians) for each facet of the convex hull.
    Returns:
        radii: array of shape (n_facets,)
        simplices: array of shape (n_facets, 3)
        normals: array of shape (n_facets, 3)
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, None
    
    simplices = hull.simplices
    n_facets = len(simplices)
    radii = np.zeros(n_facets)
    normals = np.zeros((n_facets, 3))
    
    # Vectorize circumcenter calculation
    # For each facet (a, b, c), normal n = (a-b) x (a-c)
    # The circumcenter on the sphere is along n.
    # rad = arccos(dot(n_normalized, a))
    
    for i in range(n_facets):
        tri = simplices[i]
        a = pts[tri[0]]
        b = pts[tri[1]]
        c = pts[tri[2]]
        
        n = np.cross(a - b, a - c)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            radii[i] = 0.0
            continue
        n = n / n_norm
        # Ensure normal points outward
        centroid = (a + b + c) / 3.0
        if np.dot(n, centroid) < 0:
            n = -n
        
        normals[i] = n
        dot_val = np.clip(np.dot(n, a), -1.0, 1.0)
        radii[i] = np.arccos(dot_val)
        
    return radii, simplices, normals

def _soft_max_gradient(pts, temperature=0.05):
    """
    Compute the gradient of the soft-max of facet circumradii.
    This provides a smoother objective than the hard max.
    """
    radii, simplices, normals = _compute_facet_radii(pts)
    if radii is None:
        return np.zeros_like(pts), 0.0
    
    # Soft-max weights
    # Shift by max for numerical stability
    max_rad = np.max(radii)
    exp_radii = np.exp((radii - max_rad) / temperature)
    sum_exp = np.sum(exp_radii)
    weights = exp_radii / sum_exp
    
    # Soft-max value
    soft_max_val = max_rad + temperature * np.log(sum_exp)
    
    # Gradient calculation
    # d(soft_max)/d(pts[i]) = sum_j weights[j] * d(radii[j])/d(pts[i])
    # For a facet j with vertices (a, b, c) and normal n_j:
    # radii[j] = arccos(n_j . a)
    # d(radii)/d(a) = -n_j / sqrt(1 - (n_j . a)^2)
    # But we are moving points on the sphere, so we project the gradient.
    
    grad = np.zeros_like(pts)
    for j in range(len(simplices)):
        w = weights[j]
        if w < 1e-8:
            continue
        tri = simplices[j]
        n = normals[j]
        rad = radii[j]
        
        # The gradient of arccos(x) is -1/sqrt(1-x^2)
        # Here x = n . a
        # The term 1/sqrt(1-x^2) is 1/sin(rad)
        # So d(rad)/d(a) = -n / sin(rad)
        # However, we want to minimize the radius, so we move a towards n.
        # The force on a should be proportional to n.
        
        # Let's use the direct geometric intuition:
        # To reduce the circumradius of a facet, move its vertices towards the circumcenter (normal).
        # The "force" is the projection of the normal onto the tangent space.
        # Since n is already the circumcenter direction, and a is on the sphere,
        # the direction to move a is simply n - (n.a)a = n - cos(rad)a.
        # But n.a = cos(rad), so n - cos(rad)a is the vector from a to n projected on tangent.
        # Actually, the gradient of the distance from a to n on the sphere is -n (projected).
        
        # Let's just use the normal direction scaled by weight.
        # This is a subgradient-like approach.
        for k in range(3):
            idx = tri[k]
            a = pts[idx]
            # Project normal onto tangent plane at a
            # Tangent projection of n is n - (n.a)a
            proj = n - np.dot(n, a) * a
            grad[idx] += w * proj
            
    return grad, soft_max_val

def _local_optimize(pts, max_time, initial_step=0.02, momentum=0.9, temperature=0.05):
    """
    Local optimization using soft-max gradient descent.
    """
    velocities = np.zeros_like(pts)
    step_size = initial_step
    best_pts = pts.copy()
    
    # Initial radius
    radii, _, _ = _compute_facet_radii(pts)
    if radii is None:
        return pts, 180.0
    best_rad = np.max(radii)
    
    it = 0
    while time.time() < max_time:
        it += 1
        
        grad, soft_val = _soft_max_gradient(pts, temperature)
        
        # Update velocities
        velocities = momentum * velocities - step_size * grad
        new_pts = pts + velocities
        new_pts = _normalize(new_pts)
        
        # Check improvement
        new_radii, _, _ = _compute_facet_radii(new_pts)
        if new_radii is None:
            break
        new_rad = np.max(new_radii)
        
        if new_rad < best_rad:
            best_rad = new_rad
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            # Slowly decrease temperature to sharpen the max
            temperature *= 0.999
        else:
            # Revert and adjust step size
            step_size *= 0.9
            velocities *= 0.5
            if step_size < 1e-6:
                break
        
        # Add small noise
        if it % 10 == 0:
            noise = np.random.normal(0, step_size * 0.05, pts.shape)
            pts = _normalize(pts + noise)
            
    return best_pts, best_rad

def construct():
    start_time = time.time()
    max_time = 145.0  # Safety margin
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    best_points = points.copy()
    radii, _, _ = _compute_facet_radii(best_points)
    best_rad = np.max(radii)
    
    # Multiple restarts
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() - start_time > max_time - 5.0:
            break
        
        if restart == 0:
            pts = points.copy()
        else:
            # Perturbed initialization
            pts = points.copy()
            noise = np.random.normal(0, 0.1 * (restart + 1), pts.shape)
            pts = _normalize(pts + noise)
        
        remaining_time = max_time - (time.time() - start_time)
        if remaining_time < 5.0:
            break
        
        opt_pts, opt_rad = _local_optimize(pts, start_time + remaining_time - 2.0, 
                                           initial_step=0.02, momentum=0.9, temperature=0.05)
        
        if opt_rad < best_rad:
            best_rad = opt_rad
            best_points = opt_pts.copy()
            
    # Final polish with very small step and low temperature
    if time.time() - start_time < max_time - 10.0:
        polish_pts, polish_rad = _local_optimize(best_points, max_time - 2.0, 
                                                  initial_step=0.005, momentum=0.8, temperature=0.01)
        if polish_rad < best_rad:
            best_rad = polish_rad
            best_points = polish_pts.copy()
    
    return best_points
