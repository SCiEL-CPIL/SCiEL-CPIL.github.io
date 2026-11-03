import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
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
    if n_facets == 0:
        return None, None, None
        
    radii = np.zeros(n_facets)
    normals = np.zeros((n_facets, 3))
    
    # Vectorized circumcenter calculation
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    
    n = np.cross(a - b, a - c)
    n_norms = np.linalg.norm(n, axis=1)
    valid = n_norms > 1e-10
    
    if not np.all(valid):
        # Handle degenerate cases
        for i in range(n_facets):
            if not valid[i]:
                radii[i] = 0.0
                normals[i] = [0, 0, 1]
                continue
            
            tri = simplices[i]
            a_i = pts[tri[0]]
            b_i = pts[tri[1]]
            c_i = pts[tri[2]]
            n_i = np.cross(a_i - b_i, a_i - c_i)
            n_norm_i = np.linalg.norm(n_i)
            if n_norm_i < 1e-10:
                radii[i] = 0.0
                normals[i] = [0, 0, 1]
                continue
            n_i = n_i / n_norm_i
            centroid = (a_i + b_i + c_i) / 3.0
            if np.dot(n_i, centroid) < 0:
                n_i = -n_i
            normals[i] = n_i
            dot_val = np.clip(np.dot(n_i, a_i), -1.0, 1.0)
            radii[i] = np.arccos(dot_val)
        
        return radii, simplices, normals
    
    n = n / n_norms[:, np.newaxis]
    
    # Ensure normals point outward
    centroids = (a + b + c) / 3.0
    dots = np.sum(n * centroids, axis=1)
    n[dots < 0] *= -1
    
    normals = n
    
    dot_val = np.clip(np.sum(n * a, axis=1), -1.0, 1.0)
    radii = np.arccos(dot_val)
    
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
    max_rad = np.max(radii)
    exp_radii = np.exp((radii - max_rad) / temperature)
    sum_exp = np.sum(exp_radii)
    weights = exp_radii / sum_exp
    
    # Soft-max value
    soft_max_val = max_rad + temperature * np.log(sum_exp)
    
    grad = np.zeros_like(pts)
    for j in range(len(simplices)):
        w = weights[j]
        if w < 1e-8:
            continue
        tri = simplices[j]
        n = normals[j]
        
        for k in range(3):
            idx = tri[k]
            a = pts[idx]
            # Project normal onto tangent plane at a
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
    
    radii, _, _ = _compute_facet_radii(pts)
    if radii is None:
        return pts, 180.0
    best_rad = np.max(radii)
    
    it = 0
    no_improve = 0
    while time.time() < max_time:
        it += 1
        
        grad, soft_val = _soft_max_gradient(pts, temperature)
        
        velocities = momentum * velocities - step_size * grad
        new_pts = pts + velocities
        new_pts = _normalize(new_pts)
        
        new_radii, _, _ = _compute_facet_radii(new_pts)
        if new_radii is None:
            break
        new_rad = np.max(new_radii)
        
        if new_rad < best_rad:
            best_rad = new_rad
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            no_improve = 0
            temperature *= 0.999
        else:
            no_improve += 1
            step_size *= 0.95
            velocities *= 0.5
            if step_size < 1e-7 or no_improve > 100:
                break
        
        # Add small noise occasionally to escape local minima
        if it % 20 == 0 and no_improve < 50:
            noise = np.random.normal(0, step_size * 0.1, pts.shape)
            pts = _normalize(pts + noise)
            
    return best_pts, best_rad

def _polish_largest_facets(points, max_facets=30, max_iter=1000, time_limit=None):
    """
    Polish the configuration by minimizing the max facet radius using Nelder-Mead
    on the vertices of the largest facets.
    """
    best_points = points.copy()
    best_radius, radii, simplices, centers = _compute_covering_radius_and_facets(points)
    
    if len(simplices) == 0:
        return best_points
    
    # Get indices of vertices in the largest facets
    top_k = min(max_facets, len(radii))
    top_indices = np.argsort(radii)[-top_k:]
    
    # Collect unique vertex indices from these facets
    vertex_indices = set()
    for idx in top_indices:
        for v in simplices[idx]:
            vertex_indices.add(v)
    
    vertex_indices = sorted(list(vertex_indices))
    
    if len(vertex_indices) == 0:
        return best_points
    
    def objective(x_flat):
        if time_limit is not None and time.time() > time_limit:
            return np.pi
        x = x_flat.reshape(-1, 3)
        new_points = points.copy()
        for i, idx in enumerate(vertex_indices):
            new_points[idx] = x[i]
        new_points = _normalize(new_points)
        if not np.all(np.isfinite(new_points)):
            return np.pi
        r, _, _, _ = _compute_covering_radius_and_facets(new_points)
        return r
    
    x0 = points[vertex_indices].flatten()
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-14})
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

def _compute_covering_radius_and_facets(points):
    """
    Computes the covering radius and returns the radii and indices for each facet.
    points: (N, 3) array of unit vectors.
    """
    radii, simplices, normals = _compute_facet_radii(points)
    if radii is None:
        return np.pi, np.array([]), np.array([]), np.array([])
    max_radius = np.max(radii)
    return max_radius, radii, simplices, normals

def construct():
    start_time = time.time()
    max_time = 145.0  # Safety margin before 150s kill
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    best_points = points.copy()
    radii, _, _ = _compute_facet_radii(best_points)
    best_rad = np.max(radii) if radii is not None else 180.0
    
    # Multiple restarts with different initializations
    n_restarts = 12
    for restart in range(n_restarts):
        if time.time() - start_time > max_time - 15.0:
            break
        
        if restart == 0:
            pts = points.copy()
        elif restart == 1:
            # Random points
            np.random.seed(42)
            pts = np.random.randn(N, 3)
            pts = _normalize(pts)
        elif restart == 2:
            # Fibonacci + noise
            np.random.seed(123)
            pts = points + np.random.randn(N, 3) * 0.1
            pts = _normalize(pts)
        else:
            # Perturbed Fibonacci with varying noise
            np.random.seed(restart * 100 + 7)
            noise_scale = 0.05 + 0.05 * (restart % 4)
            pts = points + np.random.randn(N, 3) * noise_scale
            pts = _normalize(pts)
        
        remaining_time = max_time - (time.time() - start_time)
        if remaining_time < 5.0:
            break
        
        # Phase 1: Gradient-based local optimization
        opt_pts, opt_rad = _local_optimize(pts, start_time + remaining_time - 10.0, 
                                           initial_step=0.02, momentum=0.9, temperature=0.05)
        
        if opt_rad < best_rad:
            best_rad = opt_rad
            best_points = opt_pts.copy()
        
        # Phase 2: Nelder-Mead polishing on largest facets
        if time.time() - start_time < max_time - 5.0:
            polish_time = max_time - 5.0
            polished = _polish_largest_facets(best_points, max_facets=25, max_iter=500, time_limit=polish_time)
            pol_radius, _, _, _ = _compute_covering_radius_and_facets(polished)
            if pol_radius < best_rad:
                best_rad = pol_radius
                best_points = polished.copy()
    
    # Final polish with very small step and low temperature
    if time.time() - start_time < max_time - 3.0:
        polish_pts, polish_rad = _local_optimize(best_points, max_time - 2.0, 
                                                  initial_step=0.003, momentum=0.8, temperature=0.01)
        if polish_rad < best_rad:
            best_rad = polish_rad
            best_points = polish_pts.copy()
    
    # Final Nelder-Mead polish
    if time.time() - start_time < max_time - 1.0:
        polished = _polish_largest_facets(best_points, max_facets=20, max_iter=300, time_limit=max_time - 1.0)
        pol_radius, _, _, _ = _compute_covering_radius_and_facets(polished)
        if pol_radius < best_rad:
            best_rad = pol_radius
            best_points = polished.copy()
    
    return best_points
