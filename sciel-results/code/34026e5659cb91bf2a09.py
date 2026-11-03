import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def compute_facet_data(points):
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
    if not np.all(valid):
        pass
        
    # Normalize normals
    u = np.zeros_like(n)
    u[valid] = n[valid] / norms[valid, np.newaxis]
    u[~valid] = [0, 0, 1]
    
    # Orient normals outwards (away from origin)
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius
    # The circumcenter of a spherical triangle (facet of convex hull) is the unit vector u.
    # The angular distance from the circumcenter to any vertex is the circumradius.
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    # For degenerate facets, set radius to pi
    radii[~valid] = np.pi
    
    return radii, u, simplices

def greedy_init(n_points, seed=42):
    """
    Greedy initialization: Farthest Point Sampling.
    """
    np.random.seed(seed)
    points = [np.random.randn(3)]
    points[0] /= np.linalg.norm(points[0])
    
    for _ in range(n_points - 1):
        # Generate candidates
        candidates = np.random.randn(500, 3)
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # Find min distance to existing points
        min_dists = np.full(len(candidates), np.inf)
        # Vectorized distance calculation
        # dists shape: (500, M)
        # We can do this in chunks if M is large, but M=130 is small.
        dists = np.linalg.norm(candidates[:, np.newaxis, :] - np.array(points)[np.newaxis, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        
        best_idx = np.argmax(min_dists)
        points.append(candidates[best_idx])
        
    return np.array(points)

def lloyd_step(points, alpha=0.5):
    """
    Perform one step of Lloyd's algorithm.
    Move each point towards the centroid of its Voronoi cell.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return points

    # For each point, find the facets that contain it.
    # Sum the facet centroids (or circumcenters, weighted by area?)
    # A better approximation for the Voronoi centroid on the sphere is to average
    # the circumcenters of the adjacent facets, weighted by the area of the facet.
    # However, simple average of circumcenters is often sufficient for convergence.
    # Let's use the circumcenters of the adjacent triangles.
    
    # Get circumcenters for all facets
    radii, centers, simplices = compute_facet_data(points)
    
    if len(radii) == 0:
        return points

    # Accumulate updates for each point
    updates = np.zeros_like(points)
    counts = np.zeros(N)
    
    # For each facet, add its circumcenter to the update of its vertices
    for m in range(len(centers)):
        idx = simplices[m]
        i_idx, j_idx, k_idx = idx
        # Weight by 1/radius? No, just simple average.
        # Or weight by sin(radius) which is proportional to area?
        # Let's stick to simple average first.
        updates[i_idx] += centers[m]
        updates[j_idx] += centers[m]
        updates[k_idx] += centers[m]
        counts[i_idx] += 1
        counts[j_idx] += 1
        counts[k_idx] += 1
    
    # Average
    valid = counts > 0
    updates[valid] /= counts[valid, np.newaxis]
    
    # Move points
    new_points = points.copy()
    new_points[valid] = points[valid] + alpha * (updates[valid] - points[valid])
    
    # Normalize
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    new_points /= norms
    
    return new_points

def facet_push_step(points, radii, centers, simplices, T=0.1):
    """
    Push vertices of large facets towards the circumcenter.
    """
    if len(radii) == 0:
        return points
        
    # Identify the top 20% largest facets
    threshold = np.percentile(radii, 80)
    mask = radii > threshold
    
    if not np.any(mask):
        return points
        
    updates = np.zeros_like(points)
    
    # Vectorized update accumulation is tricky with duplicate indices.
    # Loop is fast enough for ~260 facets.
    for m in np.where(mask)[0]:
        idx = simplices[m]
        i_idx, j_idx, k_idx = idx
        center = centers[m]
        radius = radii[m]
        
        # Step size
        step = T * radius
        
        updates[i_idx] += step * (center - points[i_idx])
        updates[j_idx] += step * (center - points[j_idx])
        updates[k_idx] += step * (center - points[k_idx])
        
    new_points = points + updates
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    new_points /= norms
    
    return new_points

def construct():
    start_time = time.time()
    time_limit = 115.0 # Safety margin
    
    best_points = None
    best_radius = np.pi
    
    # Initial configurations
    # 1. Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Greedy Initialization
    init_greedy = greedy_init(N, seed=10)
    
    # 3. Random
    np.random.seed(42)
    init_rand = np.random.randn(N, 3)
    init_rand /= np.linalg.norm(init_rand, axis=1, keepdims=True)
    
    initializations = [init_greedy, init_fib, init_rand]
    
    # We will run optimization on these.
    # Strategy: Alternate between Lloyd (global balance) and Facet Push (local refinement).
    
    max_restarts = 10
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if restart < len(initializations):
            points = initializations[restart].copy()
        else:
            # Perturb the best found so far
            if best_points is not None:
                points = best_points + np.random.randn(N, 3) * 0.05
            else:
                points = init_greedy + np.random.randn(N, 3) * 0.2
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Optimization loop
        T = 0.05 # Step size for facet push
        alpha_lloyd = 0.5 # Step size for Lloyd
        no_improve_count = 0
        current_radius = np.pi
        
        # Inner loop
        while True:
            if time.time() - start_time > time_limit:
                break
                
            radii, centers, simplices = compute_facet_data(points)
            
            if len(radii) == 0:
                break
                
            max_radius = np.max(radii)
            
            if max_radius < best_radius:
                best_radius = max_radius
                best_points = points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 50:
                    T *= 0.9
                    alpha_lloyd *= 0.95
                if no_improve_count > 200:
                    break
                if T < 1e-6 and alpha_lloyd < 1e-6:
                    break
            
            # Alternate steps
            # Every 10 steps, do a Lloyd step to redistribute
            # Otherwise, do Facet Push to refine the worst caps
            
            if no_improve_count % 10 == 0:
                # Lloyd Step
                points = lloyd_step(points, alpha=alpha_lloyd)
            else:
                # Facet Push Step
                points = facet_push_step(points, radii, centers, simplices, T=T)
            
            # Normalize just in case (lloyd_step and facet_push_step already normalize, 
            # but it's cheap)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            points /= norms
            
            # Cool down
            T *= 0.995
            alpha_lloyd *= 0.995

    if best_points is None:
        best_points = init_fib
        
    return best_points
