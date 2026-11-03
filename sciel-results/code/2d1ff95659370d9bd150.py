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

def compute_gradient(points, radii, centers, simplices, beta):
    """
    Compute the gradient of the soft-max objective:
    f = (1/beta) * log(sum(exp(beta * r_i)))
    
    The gradient of f w.r.t. point p_j is:
    sum_i w_i * d(r_i)/d(p_j)
    
    where w_i = exp(beta * (r_i - r_max)) / sum(exp(beta * (r_k - r_max)))
    
    For a facet i with circumcenter u_i and radius r_i, the gradient of r_i w.r.t. 
    a vertex p is approximately:
    d(r_i)/d(p) = -u_i * (I - u_i u_i^T) / sin(r_i)  (projected gradient)
    
    But a simpler approximation: move each vertex toward the circumcenter.
    The gradient direction for vertex p in facet i is proportional to:
    -u_i (the circumcenter direction), projected to the tangent plane at p.
    """
    if len(radii) == 0:
        return np.zeros_like(points)
        
    # Soft-max weights
    r_max = np.max(radii)
    exponents = beta * (radii - r_max)
    weights = np.exp(exponents)
    weights /= np.sum(weights)
    
    grad = np.zeros_like(points)
    
    for i in range(len(simplices)):
        w = weights[i]
        if w < 1e-12:
            continue
        
        center = centers[i]
        radius = radii[i]
        
        # Gradient of arccos(u.p) w.r.t. p is -u/sqrt(1-(u.p)^2) = -u/sin(r)
        # But we want to minimize r, so we move p in direction that decreases r.
        # d(r)/d(p) = -u / sin(r) (unprojected)
        # Project to tangent plane at p: grad_tangent = grad - (grad.p)*p
        sin_r = np.sin(radius)
        if sin_r < 1e-10:
            continue
            
        for v_idx in simplices[i]:
            p = points[v_idx]
            # Unprojected gradient of r_i w.r.t. p
            grad_unproj = -center / sin_r
            # Project to tangent plane at p
            dot = np.dot(grad_unproj, p)
            grad_tangent = grad_unproj - dot * p
            grad[v_idx] += w * grad_tangent
    
    return grad

def smooth_polish(points, time_limit, max_iters=2000, beta_start=5.0, beta_end=100.0):
    """
    Smooth polish using soft-max objective and gradient descent.
    """
    current = points.copy()
    best_radius = np.pi
    best_points = current.copy()
    start = time.time()
    
    step_size = 0.02
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        radii, centers, simplices = compute_facet_data(current)
        if len(radii) == 0:
            break
        
        cur_r = np.max(radii)
        
        if cur_r < best_radius:
            best_radius = cur_r
            best_points = current.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 50:
                step_size *= 0.8
                if step_size < 1e-5:
                    break
                no_improve_count = 0
        
        # Adaptive beta: start small (smooth), increase (sharp)
        progress = it / max_iters
        beta = beta_start + (beta_end - beta_start) * progress
        
        # Compute gradient
        grad = compute_gradient(current, radii, centers, simplices, beta)
        
        # Gradient descent step
        current = current - step_size * grad
        current /= np.linalg.norm(current, axis=1, keepdims=True)
    
    return best_points

def lloyd_relaxation(points, n_iter=20, step_size=0.3):
    """
    Lloyd's algorithm for spherical Voronoi: move each point toward the centroid
    of its Voronoi cell.
    """
    current = points.copy()
    for _ in range(n_iter):
        centroids = spherical_voronoi_centroids(current)
        current = current + step_size * (centroids - current)
        current /= np.linalg.norm(current, axis=1, keepdims=True)
    return current

def spherical_voronoi_centroids(points):
    """
    Compute the centroid of each point's spherical Voronoi cell via sampling.
    """
    n_samples = 50000
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = samples @ points.T
    nearest = np.argmin(-dots, axis=1)
    
    centroids = np.zeros_like(points)
    for i in range(N):
        mask = nearest == i
        if np.any(mask):
            centroids[i] = np.sum(samples[mask], axis=0)
    
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    centroids = centroids / norms
    
    return centroids

def greedy_farthest_init(n_points):
    """
    Greedy farthest-point sampling for a good initial configuration.
    """
    np.random.seed(123)
    points = [np.random.randn(3)]
    points[0] /= np.linalg.norm(points[0])
    
    for _ in range(n_points - 1):
        n_grid = 200
        i = np.arange(n_grid) + 0.5
        z = 1.0 - 2.0 * i / n_grid
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
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
    
    # Phase 1: Lloyd relaxation on each init, then smooth polish
    for restart_idx, init_points in enumerate(inits):
        if time.time() - start_time > time_limit - 20:
            break
        
        # Apply Lloyd relaxation
        lloyd_points = lloyd_relaxation(init_points, n_iter=10, step_size=0.3)
        update_best(lloyd_points)
        
        # Phase 2: Smooth polish with soft-max gradient
        remaining_time = time_limit - (time.time() - start_time)
        polish_time = min(20.0, remaining_time * 0.4)
        
        if polish_time > 5:
            polished = smooth_polish(lloyd_points, polish_time, max_iters=1500)
            update_best(polished)
            lloyd_points = polished
    
    # Phase 3: Final high-precision polish on the best configuration
    if best_points is not None and time.time() - start_time < time_limit - 10:
        polish_time = min(30.0, time_limit - (time.time() - start_time))
        polished = smooth_polish(best_points, polish_time, max_iters=2000, beta_start=20.0, beta_end=200.0)
        update_best(polished)
    
    if best_points is None:
        best_points = inits[0]
        
    return best_points
