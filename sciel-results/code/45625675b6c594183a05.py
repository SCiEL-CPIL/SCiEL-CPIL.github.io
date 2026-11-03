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

def greedy_init(n_points, seed=0):
    """
    Greedy initialization: start with a random point, then repeatedly add the point
    that maximizes the minimum distance to existing points.
    """
    np.random.seed(seed)
    points = [np.random.randn(3)]
    points[0] /= np.linalg.norm(points[0])
    
    for _ in range(n_points - 1):
        # Generate many candidates
        candidates = np.random.randn(500, 3)
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # For each candidate, compute min distance to existing points
        existing = np.array(points)
        # Dot products
        dots = candidates @ existing.T
        min_cos = np.min(dots, axis=1)
        min_cos = np.clip(min_cos, -1.0, 1.0)
        max_dist = np.arccos(min_cos)
        
        best_idx = np.argmax(max_dist)
        points.append(candidates[best_idx])
    
    return np.array(points)

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    best_points = None
    best_radius = np.pi
    
    # Initial configurations
    initializations = []
    
    # 1. Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(init_fib)
    
    # 2. Random points
    np.random.seed(42)
    init_rand = np.random.randn(N, 3)
    init_rand /= np.linalg.norm(init_rand, axis=1, keepdims=True)
    initializations.append(init_rand)
    
    # 3. Greedy initialization
    init_greedy = greedy_init(N, seed=42)
    initializations.append(init_greedy)
    
    # 4. Another greedy with different seed
    init_greedy2 = greedy_init(N, seed=123)
    initializations.append(init_greedy2)
    
    # 5. Fibonacci + noise
    np.random.seed(123)
    init_fib_noise = init_fib + np.random.randn(N, 3) * 0.2
    init_fib_noise /= np.linalg.norm(init_fib_noise, axis=1, keepdims=True)
    initializations.append(init_fib_noise)
    
    # More randomized restarts
    for s in range(15):
        np.random.seed(1000 + s * 7)
        init = init_fib + np.random.randn(N, 3) * (0.1 + 0.1 * s / 15)
        init /= np.linalg.norm(init, axis=1, keepdims=True)
        initializations.append(init)
        
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if restart < len(initializations):
            points = initializations[restart].copy()
        else:
            np.random.seed(restart * 137)
            points = init_fib + np.random.randn(N, 3) * 0.3
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Phase 1: Smooth Gradient Descent using Log-Sum-Exp surrogate
        # This helps balance all facets rather than just the worst one.
        tau = 0.1  # Temperature for soft-max
        alpha = 0.05
        no_improve = 0
        
        for step in range(3000):
            if time.time() - start_time > time_limit:
                break
                
            radii, centers, simplices = compute_facet_data(points)
            if len(radii) == 0:
                break
                
            max_radius = np.max(radii)
            
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
                if no_improve > 150:
                    break
            
            # Compute soft-max weights
            # To avoid overflow, subtract max radius
            r_scaled = radii / tau
            r_max = np.max(r_scaled)
            exp_terms = np.exp(r_scaled - r_max)
            sum_exp = np.sum(exp_terms)
            weights = exp_terms / sum_exp
            
            # Compute gradients
            # grad_r_f wrt p_i = -u_f / sin(r_f)
            # We want to move p_i in direction -grad = u_f / sin(r_f)
            # Update = sum_f w_f * (u_f / sin(r_f))
            
            updates = np.zeros_like(points)
            
            sin_r = np.sin(radii)
            # Avoid division by zero
            sin_r[sin_r < 1e-6] = 1e-6
            
            for m in range(len(radii)):
                simplex = simplices[m]
                i_idx, j_idx, k_idx = simplex
                
                center = centers[m]
                w = weights[m]
                factor = w / sin_r[m]
                
                # Gradient direction is towards the center u
                # We add factor * u to the update (since we are doing descent on -log-sum-exp? 
                # Wait. Objective is J = log(sum(exp(r/tau))). 
                # grad J = sum w_f grad r_f.
                # grad r_f wrt p_i = -u_f / sin(r_f).
                # So grad J wrt p_i = sum_f w_f (-u_f / sin(r_f)).
                # Descent: p = p - alpha * grad J = p + alpha * sum_f w_f (u_f / sin(r_f)).
                # So we add factor * center.
                
                updates[i_idx] += factor * center
                updates[j_idx] += factor * center
                updates[k_idx] += factor * center
            
            # Project updates to tangent plane
            # v_tan = v - (v . p) p
            for i in range(N):
                proj = np.dot(updates[i], points[i])
                updates[i] -= proj * points[i]
            
            points += alpha * updates
            points /= np.linalg.norm(points, axis=1, keepdims=True)

        # Phase 2: High-precision polish with discrete max-facet logic
        # This refines the solution found by the smooth phase.
        if time.time() - start_time < time_limit - 5.0:
            alpha_polish = 0.01
            for step in range(1000):
                if time.time() - start_time > time_limit - 2.0:
                    break
                
                radii, centers, simplices = compute_facet_data(points)
                if len(radii) == 0:
                    break
                
                max_radius = np.max(radii)
                
                if max_radius < best_radius:
                    best_radius = max_radius
                    best_points = points.copy()
                
                # Move only the top 10% largest facets
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
                
                alpha_polish *= 0.995

    if best_points is None:
        best_points = init_fib
        
    return best_points
