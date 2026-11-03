import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def compute_covering_radius_and_facets(points):
    """
    Computes the covering radius and returns the radii and indices for each facet.
    points: (N, 3) array of unit vectors.
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

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return points / norms

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
    
    # Try multiple initializations if time allows
    # For now, just one robust run, but we can add restarts later if needed.
    # Let's stick to one strong run with a good algorithm.
    
    points = init_fib.copy()
    
    # Lloyd-like iteration
    # Move points towards the circumcenters of the largest facets they belong to.
    
    max_iter = 2000
    alpha = 0.1 # Step size
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit - 5.0:
            break
        
        max_radius, radii, simplices, centers = compute_covering_radius_and_facets(points)
        
        if len(simplices) == 0:
            break
            
        if max_radius < best_radius:
            best_radius = max_radius
            best_points = points.copy()
            
        # Calculate update for each point
        updates = np.zeros_like(points)
        
        # We want to minimize the max radius.
        # Strategy: Move vertices of facets with radius > threshold towards the facet center.
        # The strength of the move is proportional to how much the radius exceeds the mean or a threshold.
        
        mean_radius = np.mean(radii)
        # Use a threshold slightly above the mean to focus on the "worst" facets
        threshold = mean_radius + 0.5 * (max_radius - mean_radius)
        
        # For each facet, if it's "large", add a pull towards its center for its vertices.
        # We accumulate these pulls.
        
        for m in range(len(simplices)):
            r_m = radii[m]
            if r_m > threshold:
                # Weight is proportional to excess radius
                weight = (r_m - threshold)
                
                simplex = simplices[m]
                i_idx, j_idx, k_idx = simplex
                
                center = centers[m]
                
                # Vector from vertex to center
                # Note: center is a unit vector.
                # Moving towards center on the sphere means adding a component in the direction of center.
                # p_new = p + alpha * weight * (center - p)
                # This is a chordal step. We normalize afterwards.
                
                updates[i_idx] += weight * (center - points[i_idx])
                updates[j_idx] += weight * (center - points[j_idx])
                updates[k_idx] += weight * (center - points[k_idx])
        
        # Apply update
        points = points + alpha * updates
        points = _normalize(points)
        
        # Optional: Small random perturbation to escape local minima?
        # Might be dangerous. Let's rely on the dynamics.
        
    if best_points is None:
        best_points = init_fib
    
    return best_points
