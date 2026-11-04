"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _compute_facet_data(points):
    """
    Computes the circumradius and circumcenter direction for each facet of the convex hull.
    Returns:
        radii: array of shape (M,)
        centers: array of shape (M, 3) unit vectors
        simplices: array of shape (M, 3)
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return np.array([]), np.array([]), np.array([])
    
    if len(simplices) == 0:
        return np.array([]), np.array([]), np.array([])
        
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    # Compute normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    valid = norms > 1e-10
    
    # Initialize outputs
    M = len(simplices)
    radii = np.full(M, np.pi)
    centers = np.zeros((M, 3))
    centers[~valid] = [0, 0, 1]
    
    if not np.all(valid):
        if np.any(valid):
            a_v = a[valid]
            b_v = b[valid]
            c_v = c[valid]
            n_v = n[valid]
            norms_v = norms[valid]
            
            u = n_v / norms_v[:, np.newaxis]
            centroids = (a_v + b_v + c_v) / 3.0
            dots = np.sum(u * centroids, axis=1)
            u[dots < 0] *= -1
            
            cos_theta = np.sum(u * a_v, axis=1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            radii[valid] = np.arccos(cos_theta)
            centers[valid] = u
            
        return radii, centers, simplices

    # All valid
    u = n / norms[:, np.newaxis]
    
    # Orient normals outwards (away from origin)
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    return radii, u, simplices

def _covering_radius_deg(points):
    """Compute the covering radius in degrees for a set of unit points."""
    if len(points) < 4:
        return 180.0
    radii, centers, simplices = _compute_facet_data(points)
    if len(radii) == 0:
        return 180.0
    return np.degrees(np.max(radii))

def _lloyd_spherical(points, alpha=0.5):
    """
    Perform one step of Lloyd's algorithm.
    Move each point towards the centroid of its Voronoi cell (average of adjacent facet circumcenters).
    """
    radii, centers, simplices = _compute_facet_data(points)
    
    if len(radii) == 0:
        return points

    updates = np.zeros_like(points)
    counts = np.zeros(len(points))
    
    for m in range(len(centers)):
        idx = simplices[m]
        i_idx, j_idx, k_idx = idx
        updates[i_idx] += centers[m]
        updates[j_idx] += centers[m]
        updates[k_idx] += centers[m]
        counts[i_idx] += 1
        counts[j_idx] += 1
        counts[k_idx] += 1
    
    valid = counts > 0
    if not np.any(valid):
        return points
        
    updates[valid] /= counts[valid, np.newaxis]
    
    new_points = points.copy()
    new_points[valid] = points[valid] + alpha * (updates[valid] - points[valid])
    
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    new_points /= norms
    
    return new_points

def _facet_push_step(points, radii, centers, simplices, T=0.1, percentile=80):
    """
    Push vertices of large facets towards the circumcenter.
    """
    if len(radii) == 0:
        return points
        
    threshold = np.percentile(radii, percentile)
    mask = radii > threshold
    
    if not np.any(mask):
        max_idx = np.argmax(radii)
        mask = np.zeros(len(radii), dtype=bool)
        mask[max_idx] = True
        
    updates = np.zeros_like(points)
    
    for m in np.where(mask)[0]:
        idx = simplices[m]
        i_idx, j_idx, k_idx = idx
        center = centers[m]
        radius = radii[m]
        
        step = T * radius
        
        updates[i_idx] += step * (center - points[i_idx])
        updates[j_idx] += step * (center - points[j_idx])
        updates[k_idx] += step * (center - points[k_idx])
        
    new_points = points + updates
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    new_points /= norms
    
    return new_points

def _smooth_objective_and_grad(points, beta=100.0):
    """
    Compute a smooth surrogate for the max facet radius and its gradient.
    Uses log-sum-exp of squared radii.
    Returns:
        value: float
        grad: array of shape (N, 3)
    """
    radii, centers, simplices = _compute_facet_data(points)
    if len(radii) == 0:
        return 0.0, np.zeros_like(points)
    
    # Adaptive beta: larger when radii are spread out, smaller when concentrated
    r_max = np.max(radii)
    r_min = np.min(radii)
    spread = r_max - r_min
    # beta = max(50.0, min(200.0, 50.0 / max(spread, 0.01)))
    
    max_r2 = np.max(radii**2)
    scaled = beta * (radii**2 - max_r2)
    
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp
    
    log_sum_exp_val = beta * max_r2 + max_scaled + np.log(sum_exp)
    
    value = np.sqrt(log_sum_exp_val / beta)
    
    grad_points = np.zeros_like(points)
    
    for m in range(len(radii)):
        if weights[m] < 1e-12:
            continue
            
        i_idx, j_idx, k_idx = simplices[m]
        center = centers[m]
        r = radii[m]
        
        sin_r = np.sin(r)
        if sin_r < 1e-6:
            continue
            
        # Gradient of r wrt point a: dr/da = -(u · da)/sin(r) projected to tangent
        # For the surrogate f = sqrt(LSE(beta*r^2)/beta), df/da = sum w_i * 2*beta*r_i/sin(r_i) * (-u_i/sin(r_i)) * (tangent projection)
        # Simplified: move points away from circumcenter direction for large facets
        coeff = weights[m] * 2.0 * beta * r / (sin_r * sin_r)
        
        # Project center to tangent space at each vertex
        for idx in [i_idx, j_idx, k_idx]:
            p = points[idx]
            tangent_center = center - np.dot(center, p) * p
            grad_points[idx] += coeff * tangent_center
        
    # Normalize gradient magnitude
    gnorm = np.linalg.norm(grad_points)
    if gnorm > 1e-10:
        grad_points /= gnorm
        
    return value, grad_points

def _refine(points, max_iter=2000, tol=1e-10, deadline=None):
    """
    Refine points to minimize the covering radius using a hybrid approach:
    1. Lloyd's algorithm steps
    2. Facet push steps
    3. Gradient descent on smooth surrogate
    """
    points = points.copy()
    best_cr = _covering_radius_deg(points)
    best_points = points.copy()
    
    if deadline is None:
        deadline = time.time() + 100
    
    alpha = 0.5
    T = 0.1
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Alternate between Lloyd and Facet Push
        if it % 2 == 0:
            points = _lloyd_spherical(points, alpha)
        else:
            radii, centers, simplices = _compute_facet_data(points)
            if len(radii) > 0:
                points = _facet_push_step(points, radii, centers, simplices, T, percentile=80)
            
        # Check progress
        cr = _covering_radius_deg(points)
        if cr < best_cr - 1e-10:
            best_cr = cr
            best_points = points.copy()
        else:
            # Revert and try smaller step
            points = best_points.copy()
            alpha *= 0.9
            T *= 0.9
            if alpha < 1e-6:
                break
        
        # Occasional gradient step
        if it % 10 == 0:
            val, grad = _smooth_objective_and_grad(points, beta=100.0)
            if np.linalg.norm(grad) > 1e-8:
                points = points - 0.01 * grad
                norms = np.linalg.norm(points, axis=1, keepdims=True)
                points /= norms
                cr = _covering_radius_deg(points)
                if cr < best_cr:
                    best_cr = cr
                    best_points = points.copy()
                else:
                    points = best_points.copy()
    
    return best_points, best_cr

def construct():
    global start_time
    start_time = time.time()
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Refine
    points, best_cr = _refine(points, max_iter=500, deadline=time.time() + 110)
    
    # Check time
    if time.time() - start_time > 115:
        return points
    
    # Try a few random restarts if time permits
    for restart in range(3):
        if time.time() - start_time > 115:
            break
        # Perturb the current best
        perturb = points + 0.1 * np.random.randn(N, 3)
        perturb = perturb / np.linalg.norm(perturb, axis=1, keepdims=True)
        new_points, new_cr = _refine(perturb, max_iter=100, deadline=time.time() + 115)
        if new_cr < best_cr:
            points = new_points
            best_cr = new_cr
    
    return points
# EVOLVE-BLOCK-END
