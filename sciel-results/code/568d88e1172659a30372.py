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
    
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    valid = norms > 1e-10
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
    
    u = n / norms[:, np.newaxis]
    
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    return radii, u, simplices

def fibonacci_sphere(n):
    """Generate n points on the unit sphere using Fibonacci lattice."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def greedy_init(n_points, seed=42, n_candidates=2000):
    """
    Greedy farthest-point sampling initialization.
    """
    np.random.seed(seed)
    points = [np.random.randn(3)]
    points[0] /= np.linalg.norm(points[0])
    
    for _ in range(n_points - 1):
        candidates = np.random.randn(n_candidates, 3)
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        existing = np.array(points)
        dots = candidates @ existing.T
        min_cos = np.min(dots, axis=1)
        min_cos = np.clip(min_cos, -1.0, 1.0)
        max_dist = np.arccos(min_cos)
        
        best_idx = np.argmax(max_dist)
        points.append(candidates[best_idx])
        
    return np.array(points)

def lloyd_refine(points, alpha=0.3, n_steps=50, time_limit=None, start_time=None):
    """
    Lloyd-style refinement: move each point toward the centroid of its Voronoi cell
    (approximated by averaging circumcenters of adjacent facets).
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 10.0
    
    best = points.copy()
    best_radius = np.pi
    
    for step in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        
        radii, centers, simplices = compute_facet_data(points)
        if len(radii) == 0:
            break
        
        max_r = np.max(radii)
        if max_r < best_radius:
            best_radius = max_r
            best = points.copy()
        
        # Compute Voronoi centroid for each point using bincount for efficiency
        updates = np.zeros((N, 3))
        counts = np.zeros(N)
        
        # For each facet, add its circumcenter to each of its 3 vertices
        for m in range(len(centers)):
            idx = simplices[m]
            updates[idx[0]] += centers[m]
            updates[idx[1]] += centers[m]
            updates[idx[2]] += centers[m]
            counts[idx[0]] += 1
            counts[idx[1]] += 1
            counts[idx[2]] += 1
        
        valid = counts > 0
        updates[valid] /= counts[valid, np.newaxis]
        
        # Move toward centroid
        new_points = points.copy()
        new_points[valid] = points[valid] + alpha * (updates[valid] - points[valid])
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        # Check if improvement
        radii_new, _, _ = compute_facet_data(new_points)
        if len(radii_new) > 0 and np.max(radii_new) < max_r:
            points = new_points
        else:
            # Try smaller step
            new_points = points.copy()
            new_points[valid] = points[valid] + alpha * 0.5 * (updates[valid] - points[valid])
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            radii_new, _, _ = compute_facet_data(new_points)
            if len(radii_new) > 0 and np.max(radii_new) < max_r:
                points = new_points
                alpha *= 0.9
            else:
                alpha *= 0.8
                if alpha < 0.01:
                    break
        
        if alpha < 0.01:
            break
    
    return best, best_radius

def smooth_optimize(points, time_budget, start_time):
    """
    Gradient-based optimization using a smooth log-sum-exp surrogate for max radius.
    Coarse-to-fine: start with large temperature, decrease over time.
    """
    best = points.copy()
    best_radius = np.pi
    
    # Initial radii to set temperature scale
    radii, centers, simplices = compute_facet_data(points)
    if len(radii) == 0:
        return points, np.pi
    
    max_r = np.max(radii)
    min_r = np.min(radii)
    spread = max(max_r - min_r, 0.01)
    
    # Temperature schedule: start broad, tighten
    t_start = time.time()
    
    # Phase 1: Coarse optimization (large temperature)
    alpha = 0.03
    for phase in range(3):
        phase_budget = time_budget * [0.4, 0.35, 0.25][phase]
        phase_start = time.time()
        
        # Temperature for this phase
        temp = [0.15, 0.08, 0.04][phase]
        
        while time.time() - phase_start < phase_budget:
            if time.time() - start_time > time_budget:
                break
            
            radii, centers, simplices = compute_facet_data(points)
            if len(radii) == 0:
                break
            
            max_r = np.max(radii)
            if max_r < best_radius:
                best_radius = max_r
                best = points.copy()
            
            # Log-sum-exp for smooth max
            scaled = radii / temp
            max_scaled = np.max(scaled)
            exp_terms = np.exp(scaled - max_scaled)
            sum_exp = np.sum(exp_terms)
            weights = exp_terms / sum_exp
            
            # Gradient computation using bincount-style accumulation
            updates = np.zeros((N, 3))
            
            sin_r = np.sin(radii)
            sin_r[sin_r < 1e-8] = 1e-8
            
            for m in range(len(radii)):
                if weights[m] < 1e-10:
                    continue
                
                idx = simplices[m]
                center = centers[m]
                r = radii[m]
                
                # dr/dp_i = -(u · dp_i) / sin(r) for tangent displacement
                # We want to minimize max(r), so move points to reduce large radii
                # Moving a vertex toward the circumcenter reduces the circumradius
                coeff = weights[m] / sin_r[m]
                
                for v in idx:
                    # Project center onto tangent plane at point v
                    p = points[v]
                    tangent_center = center - np.dot(center, p) * p
                    updates[v] += coeff * tangent_center
            
            # Normalize gradient
            gnorm = np.linalg.norm(updates)
            if gnorm > 1e-10:
                updates /= gnorm
            
            # Apply update
            points = points + alpha * updates
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
            # Adaptive learning rate
            if phase == 0:
                alpha *= 0.999
            elif phase == 1:
                alpha *= 0.998
            else:
                alpha *= 0.997
        
        # After each phase, do a quick Lloyd refinement
        if time.time() - start_time < time_budget - 5:
            points, _ = lloyd_refine(points, alpha=0.2, n_steps=10, 
                                      time_limit=time_budget - (time.time() - start_time) - 2,
                                      start_time=start_time)
    
    return best, best_radius

def facet_push_optimize(points, time_budget, start_time):
    """
    Direct facet-push optimization: identify large facets and push their vertices
    toward the circumcenter.
    """
    best = points.copy()
    best_radius = np.pi
    
    alpha = 0.05
    
    while time.time() - start_time < time_budget:
        radii, centers, simplices = compute_facet_data(points)
        if len(radii) == 0:
            break
        
        max_r = np.max(radii)
        if max_r < best_radius:
            best_radius = max_r
            best = points.copy()
        
        # Identify facets to push (top 20% largest)
        if len(radii) > 5:
            threshold = np.percentile(radii, 80)
            mask = radii > threshold
        else:
            max_idx = np.argmax(radii)
            mask = np.zeros(len(radii), dtype=bool)
            mask[max_idx] = True
        
        if not np.any(mask):
            break
        
        updates = np.zeros((N, 3))
        
        for m in np.where(mask)[0]:
            idx = simplices[m]
            center = centers[m]
            r = radii[m]
            
            step = alpha * r
            
            for v in idx:
                p = points[v]
                # Move toward circumcenter
                updates[v] += step * (center - p)
        
        # Project to tangent plane
        for i in range(N):
            proj = np.dot(updates[i], points[i])
            updates[i] -= proj * points[i]
        
        new_points = points + updates
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        # Check improvement
        radii_new, _, _ = compute_facet_data(new_points)
        if len(radii_new) > 0:
            if np.max(radii_new) < max_r:
                points = new_points
                alpha *= 0.995
            else:
                alpha *= 0.95
                if alpha < 0.005:
                    break
        else:
            alpha *= 0.9
            if alpha < 0.005:
                break
    
    return best, best_radius

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    best_points = None
    best_radius = np.pi
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci sphere
    init_fib = fibonacci_sphere(N)
    initializations.append(init_fib)
    
    # 2. Random
    np.random.seed(42)
    init_rand = np.random.randn(N, 3)
    init_rand /= np.linalg.norm(init_rand, axis=1, keepdims=True)
    initializations.append(init_rand)
    
    # 3. Greedy farthest-point
    init_greedy = greedy_init(N, seed=42)
    initializations.append(init_greedy)
    
    # 4. Greedy with different seed
    init_greedy2 = greedy_init(N, seed=123)
    initializations.append(init_greedy2)
    
    # 5. Fibonacci + noise
    np.random.seed(123)
    init_fib_noise = init_fib + np.random.randn(N, 3) * 0.15
    init_fib_noise /= np.linalg.norm(init_fib_noise, axis=1, keepdims=True)
    initializations.append(init_fib_noise)
    
    # 6-10. More randomized perturbations of Fibonacci
    for s in range(5):
        np.random.seed(500 + s * 37)
        noise_scale = 0.05 + 0.05 * s
        init = init_fib + np.random.randn(N, 3) * noise_scale
        init /= np.linalg.norm(init, axis=1, keepdims=True)
        initializations.append(init)
    
    n_restarts = len(initializations)
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit - 5:
            break
        
        points = initializations[restart].copy()
        
        # Phase 1: Quick Lloyd refinement to get a decent starting point
        if time.time() - start_time < time_limit - 30:
            points, r = lloyd_refine(points, alpha=0.5, n_steps=30,
                                      time_limit=min(15, time_limit - (time.time() - start_time) - 25),
                                      start_time=start_time)
            if r < best_radius:
                best_radius = r
                best_points = points.copy()
        
        # Phase 2: Smooth gradient optimization
        if time.time() - start_time < time_limit - 15:
            remaining = time_limit - (time.time() - start_time) - 10
            points, r = smooth_optimize(points, remaining, start_time)
            if r < best_radius:
                best_radius = r
                best_points = points.copy()
        
        # Phase 3: Facet push refinement
        if time.time() - start_time < time_limit - 5:
            remaining = time_limit - (time.time() - start_time) - 3
            points, r = facet_push_optimize(points, remaining, start_time)
            if r < best_radius:
                best_radius = r
                best_points = points.copy()
        
        # Phase 4: Final Lloyd polish on best
        if time.time() - start_time < time_limit - 2 and best_points is not None:
            remaining = time_limit - (time.time() - start_time) - 1
            if remaining > 3:
                points, r = lloyd_refine(best_points, alpha=0.1, n_steps=20,
                                          time_limit=remaining,
                                          start_time=start_time)
                if r < best_radius:
                    best_radius = r
                    best_points = points.copy()
    
    if best_points is None:
        best_points = init_fib
    
    # Final normalization check
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
