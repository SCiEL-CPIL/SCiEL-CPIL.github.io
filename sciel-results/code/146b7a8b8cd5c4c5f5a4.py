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
    
    # Normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    valid = norms > 1e-10
    if not np.all(valid):
        # Degenerate facets, handle gracefully
        radii = np.full(len(simplices), np.pi)
        centers = np.zeros((len(simplices), 3))
        centers[~valid] = [0, 0, 1]
        return radii, centers, simplices
        
    u = n / norms[:, np.newaxis]
    
    # Orient normals outwards
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    return radii, u, simplices

def get_objective_and_gradient(points, tau=0.1):
    """
    Computes the log-sum-exp objective and its gradient.
    """
    radii, centers, simplices = compute_facet_data(points)
    if len(radii) == 0:
        return np.pi, np.zeros_like(points)
        
    # Log-sum-exp for numerical stability
    r_scaled = radii / tau
    r_max = np.max(r_scaled)
    exp_terms = np.exp(r_scaled - r_max)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    objective = np.log(sum_exp) + r_max
    
    # Gradient computation
    updates = np.zeros_like(points)
    sin_r = np.sin(radii)
    sin_r[sin_r < 1e-6] = 1e-6
    
    for m in range(len(radii)):
        simplex = simplices[m]
        i_idx, j_idx, k_idx = simplex
        
        center = centers[m]
        w = weights[m]
        factor = w / sin_r[m]
        
        updates[i_idx] += factor * center
        updates[j_idx] += factor * center
        updates[k_idx] += factor * center
        
    # Project updates to tangent plane
    for i in range(N):
        proj = np.dot(updates[i], points[i])
        updates[i] -= proj * points[i]
        
    return objective, updates

def greedy_init(n_points, seed=42):
    """
    Greedy initialization: start with a random point, then repeatedly add the point
    that maximizes the minimum distance to existing points.
    """
    np.random.seed(seed)
    points = []
    
    # First point: random
    p = np.random.randn(3)
    p /= np.linalg.norm(p)
    points.append(p)
    
    for _ in range(n_points - 1):
        # Generate many candidate points
        candidates = np.random.randn(2000, 3)
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # For each candidate, compute min distance to existing points
        best_score = -1
        best_candidate = None
        for c in candidates:
            dists = 1.0 - points[-1] @ c  # This is wrong, need all points
            break
        
        # More efficient: compute all distances at once
        pts_arr = np.array(points)
        cos_dists = candidates @ pts_arr.T  # (2000, n_points)
        min_cos = np.min(cos_dists, axis=1)
        max_min_cos = np.max(min_cos)
        best_idx = np.argmax(min_cos)
        best_candidate = candidates[best_idx]
        
        points.append(best_candidate)
    
    return np.array(points)

def force_init(n_points, iterations=100, seed=42):
    """
    Initialize points using a repulsive force model to spread them out.
    """
    np.random.seed(seed)
    points = np.random.randn(n_points, 3)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    for _ in range(iterations):
        forces = np.zeros_like(points)
        for i in range(n_points):
            f = 0
            for j in range(n_points):
                if i == j: continue
                diff = points[i] - points[j]
                d2 = np.dot(diff, diff)
                if d2 < 1e-10:
                    f += np.random.randn(3)
                else:
                    f += diff / (d2 * np.sqrt(d2))
            forces[i] = f
            
        points += 0.01 * forces
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
    return points

def construct():
    start_time = time.time()
    time_limit = 148.0  # Safety margin
    
    best_points = None
    best_radius = np.pi
    best_obj = np.pi
    
    # Initial configurations
    initializations = []
    
    # 1. Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(init_fib)
    
    # 2. Greedy initialization
    init_greedy = greedy_init(N, seed=42)
    initializations.append(init_greedy)
    
    # 3. Force-based initialization
    init_force = force_init(N, iterations=50, seed=42)
    initializations.append(init_force)
    
    # 4. Force-based with different seed
    init_force2 = force_init(N, iterations=50, seed=123)
    initializations.append(init_force2)
    
    # 5. Fibonacci + noise
    np.random.seed(123)
    init_fib_noise = init_fib + np.random.randn(N, 3) * 0.2
    init_fib_noise /= np.linalg.norm(init_fib_noise, axis=1, keepdims=True)
    initializations.append(init_fib_noise)
    
    # 6. More randomized restarts
    for s in range(8):
        np.random.seed(1000 + s * 7)
        init = init_fib + np.random.randn(N, 3) * (0.1 + 0.1 * s / 8)
        init /= np.linalg.norm(init, axis=1, keepdims=True)
        initializations.append(init)
        
    max_restarts = len(initializations)
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        points = initializations[restart].copy()
        
        # Phase 1: Gradient Descent with Line Search
        tau = 0.1
        alpha = 0.1
        no_improve = 0
        
        for step in range(2000):
            if time.time() - start_time > time_limit:
                break
                
            obj, grad = get_objective_and_gradient(points, tau)
            
            # Check actual covering radius for best tracking
            radii, centers, simplices = compute_facet_data(points)
            if len(radii) > 0:
                max_radius = np.max(radii)
                if max_radius < best_radius:
                    best_radius = max_radius
                    best_points = points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > 150:
                        break
            else:
                break
            
            direction = grad # The function returns the descent direction
            
            # Normalize direction to have unit norm for consistent step size
            norm_dir = np.linalg.norm(direction, axis=1)
            norm_dir[norm_dir < 1e-10] = 1e-10
            direction = direction / norm_dir[:, np.newaxis]
            
            # Backtracking line search with fewer iterations
            step_size = alpha
            improved = False
            for _ in range(6):
                new_points = points + step_size * direction
                new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
                
                new_obj, _ = get_objective_and_gradient(new_points, tau)
                
                if new_obj < obj:
                    points = new_points
                    improved = True
                    break
                else:
                    step_size *= 0.5
                    
            if not improved:
                alpha *= 0.5
                if alpha < 1e-4:
                    break
                    
            # Periodic random perturbation to escape local minima
            if step % 200 == 199:
                np.random.seed(int(time.time() * 1000) % 10000)
                perturb = np.random.randn(N, 3) * 0.02
                points += perturb
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                    
        # Phase 2: High-precision polish with Nelder-Mead refinement on worst facets
        if time.time() - start_time < time_limit - 5.0:
            alpha_polish = 0.01
            for step in range(800):
                if time.time() - start_time > time_limit - 2.0:
                    break
                
                radii, centers, simplices = compute_facet_data(points)
                if len(radii) == 0:
                    break
                
                max_radius = np.max(radii)
                if max_radius < best_radius:
                    best_radius = max_radius
                    best_points = points.copy()
                
                # Focus on worst facets - top 10%
                if len(radii) > 5:
                    threshold = np.percentile(radii, 90)
                    mask = radii > threshold
                else:
                    max_idx = np.argmax(radii)
                    mask = np.zeros(len(radii), dtype=bool)
                    mask[max_idx] = True
                    
                if not np.any(mask):
                    break
                    
                updates = np.zeros_like(points)
                
                for m in np.where(mask)[0]:
                    simplex = simplices[m]
                    i_idx, j_idx, k_idx = simplex
                    
                    center = centers[m]
                    # Move vertices towards the circumcenter
                    step_size = alpha_polish * radii[m]
                    
                    updates[i_idx] += step_size * (center - points[i_idx])
                    updates[j_idx] += step_size * (center - points[j_idx])
                    updates[k_idx] += step_size * (center - points[k_idx])
                    
                # Project to tangent
                for i in range(N):
                    proj = np.dot(updates[i], points[i])
                    updates[i] -= proj * points[i]
                
                points += updates
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                
                alpha_polish *= 0.985

    if best_points is None:
        best_points = initializations[0]
        
    return best_points
