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
        
    # Extract vertices for all facets
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    # Compute normals (which point to the circumcenter of the spherical triangle)
    # The circumcenter u is parallel to (b-a) x (c-a)
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
    
    return radii, u, simplices

def spherical_voronoi_centroids(points):
    """
    Compute the centroid of each point's spherical Voronoi cell.
    Uses the fact that the Voronoi cell of point i is bounded by the great-circle
    bisectors with its nearest neighbors. We approximate by finding, for each point,
    the centroid of the region closer to it than to any other point.
    
    For efficiency, we use a sampling-based approach: sample many points on the sphere,
    assign each to its nearest input point, and compute the weighted centroid.
    """
    # Use a moderately dense sampling of the sphere
    n_samples = 50000
    # Fibonacci sphere for uniform sampling
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Assign each sample to its nearest point
    # Compute distances: ||s - p||^2 = 2 - 2 s.p
    dots = samples @ points.T  # shape (n_samples, N)
    nearest = np.argmin(-dots, axis=1)  # maximize dot product = minimize angle
    
    # Compute centroid for each point
    centroids = np.zeros_like(points)
    counts = np.zeros(N)
    for i in range(N):
        mask = nearest == i
        if np.any(mask):
            centroids[i] = np.sum(samples[mask], axis=0)
            counts[i] = np.sum(mask)
    
    # Normalize centroids
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    centroids = centroids / norms
    
    return centroids

def lloyd_relaxation(points, n_iter=20, step_size=0.3):
    """
    Lloyd's algorithm for spherical Voronoi: move each point toward the centroid
    of its Voronoi cell. This equalizes cell areas and tends to find better
    covering configurations.
    """
    current = points.copy()
    for _ in range(n_iter):
        centroids = spherical_voronoi_centroids(current)
        # Move toward centroid
        current = current + step_size * (centroids - current)
        current /= np.linalg.norm(current, axis=1, keepdims=True)
    return current

def soft_gradient_step(points, radii, centers, simplices, beta, step_size):
    """
    Performs a gradient step using a soft-max weighting of the facet radii.
    """
    if len(radii) == 0:
        return points
        
    # Calculate weights: w_i = exp(beta * (r_i - r_max))
    r_max = np.max(radii)
    exponents = beta * (radii - r_max)
    weights = np.exp(exponents)
    
    updates = np.zeros_like(points)
    
    for i in range(len(simplices)):
        w = weights[i]
        if w < 1e-10:
            continue
        simplex = simplices[i]
        center = centers[i]
        
        for v_idx in simplex:
            updates[v_idx] += w * (center - points[v_idx])
    
    points = points + step_size * updates
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    return points

def polish(points, time_limit, max_iters=5000):
    """
    Final high-precision polish: alternate between Lloyd relaxation and
    fine-grained facet pushing with small steps.
    """
    current = points.copy()
    best_radius = np.pi
    best_points = current.copy()
    start = time.time()
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
            
        # Compute current covering radius
        radii, centers, simplices = compute_facet_data(current)
        if len(radii) == 0:
            break
        cur_r = np.max(radii)
        
        if cur_r < best_radius:
            best_radius = cur_r
            best_points = current.copy()
        
        # Alternate: every 5 iterations do a Lloyd step, otherwise do fine facet push
        if it % 5 == 0:
            # Small Lloyd relaxation
            centroids = spherical_voronoi_centroids(current)
            current = current + 0.05 * (centroids - current)
            current /= np.linalg.norm(current, axis=1, keepdims=True)
        else:
            # Fine facet pushing with adaptive beta
            beta = 20.0 + 10.0 * (it / max_iters)
            step = 0.005 * (1.0 - 0.5 * it / max_iters)
            current = soft_gradient_step(current, radii, centers, simplices, beta, step)
    
    return best_points

def greedy_farthest_init(n_points):
    """
    Greedy farthest-point sampling for a good initial configuration.
    """
    np.random.seed(123)
    points = [np.random.randn(3)]
    points[0] /= np.linalg.norm(points[0])
    
    for _ in range(n_points - 1):
        # Find the point on a grid that is farthest from all existing points
        # Use a moderate grid
        n_grid = 200
        i = np.arange(n_grid) + 0.5
        z = 1.0 - 2.0 * i / n_grid
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # For each grid point, compute min distance to existing points
        min_dists = np.inf
        for p in points:
            dists = np.arccos(np.clip(grid @ p, -1, 1))
            min_dists = np.minimum(min_dists, dists)
        
        idx = np.argmax(min_dists)
        points.append(grid[idx].copy())
    
    return np.array(points)

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    best_points = None
    best_radius = np.pi
    
    def evaluate(points):
        radii, _, _ = compute_facet_data(points)
        if len(radii) == 0:
            return np.pi
        return np.max(radii)
    
    def update_best(points):
        nonlocal best_points, best_radius
        r = evaluate(points)
        if r < best_radius:
            best_radius = r
            best_points = points.copy()
        return r
    
    # Initializations
    inits = []
    
    # 1. Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(init_fib)
    
    # 2. Random
    np.random.seed(42)
    init_rand = np.random.randn(N, 3)
    init_rand /= np.linalg.norm(init_rand, axis=1, keepdims=True)
    inits.append(init_rand)
    
    # 3. Greedy farthest-point
    try:
        init_greedy = greedy_farthest_init(N)
        inits.append(init_greedy)
    except Exception:
        pass
    
    # 4. Perturbed Fibonacci
    for noise_scale in [0.1, 0.2]:
        for seed in [1, 2]:
            np.random.seed(seed)
            p = init_fib + np.random.randn(N, 3) * noise_scale
            p /= np.linalg.norm(p, axis=1, keepdims=True)
            inits.append(p)
    
    # Phase 1: Lloyd relaxation on each init
    for restart_idx, init_points in enumerate(inits):
        if time.time() - start_time > time_limit:
            break
        
        # Apply Lloyd relaxation
        lloyd_points = lloyd_relaxation(init_points, n_iter=15, step_size=0.3)
        update_best(lloyd_points)
        
        # Phase 2: Facet pushing from Lloyd-relaxed points
        current = lloyd_points.copy()
        current_r = evaluate(current)
        
        for beta_val in [10.0, 20.0, 40.0]:
            if time.time() - start_time > time_limit:
                break
            
            step_size = 0.03
            no_improve = 0
            
            for step in range(1500):
                if time.time() - start_time > time_limit:
                    break
                
                radii, centers, simplices = compute_facet_data(current)
                if len(radii) == 0:
                    break
                    
                cur_r = np.max(radii)
                
                if cur_r < current_r:
                    current_r = cur_r
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > 80:
                        step_size *= 0.9
                        if step_size < 1e-4:
                            break
                    if no_improve > 250:
                        break
                        
                current = soft_gradient_step(current, radii, centers, simplices, beta_val, step_size)
            
            update_best(current)
            
            # Use best for next beta
            if current_r < evaluate(lloyd_points):
                lloyd_points = current.copy()
    
    # Phase 3: Final polish on the best configuration
    if best_points is not None and time.time() - start_time < time_limit - 10:
        polish_time = min(30.0, time_limit - (time.time() - start_time))
        polished = polish(best_points, polish_time, max_iters=3000)
        update_best(polished)
    
    if best_points is None:
        best_points = inits[0]
        
    return best_points
