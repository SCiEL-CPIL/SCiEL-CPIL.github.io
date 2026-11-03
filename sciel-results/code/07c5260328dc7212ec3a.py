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
        return np.pi, np.array([]), np.array([])
    
    if len(simplices) == 0:
        return np.pi, np.array([]), np.array([])
        
    # Extract vertices for all facets
    # simplices is shape (M, 3)
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    # Compute normals: cross(b-a, c-a)
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    # Avoid division by zero
    valid = norms > 1e-10
    if not np.all(valid):
        # Handle degenerate cases (though unlikely for convex hull)
        norms[~valid] = 1.0
        n[~valid] = [0, 0, 1]
        
    u = n / norms[:, np.newaxis]
    
    # Orient normals to point outwards (away from origin)
    # Centroid of each triangle
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius is angle between u and a
    # cos_theta = dot(u, a)
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    max_radius = np.max(radii)
    return max_radius, radii, simplices

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    best_points = None
    best_radius = np.pi
    
    # Initial configurations
    # 1. Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Random points
    np.random.seed(42)
    init_rand = np.random.randn(N, 3)
    init_rand /= np.linalg.norm(init_rand, axis=1, keepdims=True)
    
    # 3. Fibonacci + small noise
    np.random.seed(123)
    init_fib_noise = init_fib + np.random.randn(N, 3) * 0.1
    init_fib_noise /= np.linalg.norm(init_fib_noise, axis=1, keepdims=True)
    
    initializations = [init_fib, init_rand, init_fib_noise]
    
    # Add more randomized restarts based on time
    max_restarts = 25
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if restart < len(initializations):
            points = initializations[restart].copy()
        else:
            # Random perturbation of Fibonacci
            np.random.seed(restart * 100 + 7)
            points = init_fib + np.random.randn(N, 3) * 0.3
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Local Optimization with top-k facet updates
        alpha = 0.04
        no_improve = 0
        current_radius = np.pi
        top_k_fraction = 0.2  # Move vertices of top 20% largest facets
        
        for step in range(4000):
            if time.time() - start_time > time_limit:
                break
                
            max_radius, radii, simplices = compute_covering_radius_and_facets(points)
            
            if len(simplices) == 0:
                break
                
            if max_radius < best_radius:
                best_radius = max_radius
                best_points = points.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 30:
                    alpha *= 0.92
                    if alpha < 1e-5:
                        break
                if no_improve > 150:
                    break
            
            # Find top-k facets to move
            n_facets = len(radii)
            k = max(1, int(n_facets * top_k_fraction))
            top_indices = np.argsort(radii)[-k:]
            
            # Accumulate updates for each point
            updates = np.zeros((N, 3))
            update_counts = np.zeros(N)
            
            for idx in top_indices:
                simplex = simplices[idx]
                i_idx, j_idx, k_idx = simplex
                
                a = points[i_idx]
                b = points[j_idx]
                c = points[k_idx]
                
                n = np.cross(b - a, c - a)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                u = n / norm_n
                centroid = (a + b + c) / 3.0
                if np.dot(u, centroid) < 0:
                    u = -u
                
                # Weight by radius (larger facets get more movement)
                weight = radii[idx] / max_radius
                
                step_size = alpha * weight
                updates[i_idx] += step_size * (u - a)
                updates[j_idx] += step_size * (u - b)
                updates[k_idx] += step_size * (u - c)
                update_counts[i_idx] += 1
                update_counts[j_idx] += 1
                update_counts[k_idx] += 1
            
            # Apply accumulated updates
            mask = update_counts > 0
            if np.any(mask):
                # Average the updates for points that appear in multiple facets
                for idx in np.where(mask)[0]:
                    if update_counts[idx] > 0:
                        points[idx] += updates[idx] / update_counts[idx]
                points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Final polish: single-facet with smaller steps
        if time.time() - start_time < time_limit - 5.0:
            alpha_polish = 0.01
            for step in range(500):
                if time.time() - start_time > time_limit - 2.0:
                    break
                
                max_radius, radii, simplices = compute_covering_radius_and_facets(points)
                if len(simplices) == 0:
                    break
                
                if max_radius < best_radius:
                    best_radius = max_radius
                    best_points = points.copy()
                
                max_idx = np.argmax(radii)
                simplex = simplices[max_idx]
                i_idx, j_idx, k_idx = simplex
                
                a = points[i_idx]
                b = points[j_idx]
                c = points[k_idx]
                
                n = np.cross(b - a, c - a)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    break
                u = n / norm_n
                centroid = (a + b + c) / 3.0
                if np.dot(u, centroid) < 0:
                    u = -u
                
                step_size = alpha_polish * radii[max_idx]
                
                points[i_idx] += step_size * (u - a)
                points[j_idx] += step_size * (u - b)
                points[k_idx] += step_size * (u - c)
                
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                
                alpha_polish *= 0.995

    if best_points is None:
        best_points = init_fib
        
    return best_points
