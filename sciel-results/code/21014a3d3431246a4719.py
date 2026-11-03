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
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if restart < len(initializations):
            points = initializations[restart].copy()
        else:
            # Random perturbation of Fibonacci
            np.random.seed(restart * 100)
            points = init_fib + np.random.randn(N, 3) * 0.3
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Local Optimization
        alpha = 0.05
        no_improve = 0
        current_radius = np.pi
        
        # Dynamic step limit based on remaining time
        # Estimate time per step ~ 2ms. 
        # We will just check time in the loop.
        
        for step in range(3000):
            if time.time() - start_time > time_limit:
                break
                
            max_radius, radii, simplices, centers = compute_covering_radius_and_facets(points)
            
            if len(simplices) == 0:
                break
                
            if max_radius < best_radius:
                best_radius = max_radius
                best_points = points.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 50:
                    alpha *= 0.9
                    if alpha < 1e-4:
                        break
                if no_improve > 200:
                    break
            
            # Identify facets to move
            # Move vertices of the largest facets
            # Use top 20% as in inspiration 1, but keep the step logic from current
            # Current program moves only the single largest facet.
            # Inspiration 1 moves top 20%.
            # Let's try moving top 20% with accumulated updates.
            
            threshold = np.percentile(radii, 80)
            mask = radii > threshold
            
            if not np.any(mask):
                # If no facets are above threshold, just pick the largest
                max_idx = np.argmax(radii)
                mask = np.zeros(len(radii), dtype=bool)
                mask[max_idx] = True
            
            # Accumulate moves for vertices to avoid conflicting updates
            updates = np.zeros_like(points)
            
            # Vectorized accumulation is hard because simplices are indices.
            # We can loop over the masked indices.
            # len(radii) is usually ~2N. So loop is small.
            
            for m in np.where(mask)[0]:
                simplex = simplices[m]
                i_idx, j_idx, k_idx = simplex
                
                a = points[i_idx]
                b = points[j_idx]
                c = points[k_idx]
                
                center = centers[m]
                
                # Move towards center
                # Step size proportional to radius
                step_size = alpha * radii[m]
                
                updates[i_idx] += step_size * (center - a)
                updates[j_idx] += step_size * (center - b)
                updates[k_idx] += step_size * (center - c)
            
            points += updates
            points /= np.linalg.norm(points, axis=1, keepdims=True)

    if best_points is None:
        best_points = init_fib
        
    return best_points
