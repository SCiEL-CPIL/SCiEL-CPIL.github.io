import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import warnings
warnings.filterwarnings('ignore')

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

def get_covering_radius(points):
    radii, _, _ = compute_facet_data(points)
    if len(radii) == 0:
        return np.pi
    return np.max(radii)

def smooth_covering_radius(points, temperature=0.01):
    """
    Smooth approximation of the covering radius using log-sum-exp.
    """
    radii, _, _ = compute_facet_data(points)
    if len(radii) == 0:
        return np.pi
    
    # Log-sum-exp approximation to max
    max_r = np.max(radii)
    shifted = (radii - max_r) / temperature
    # Numerical stability: clip to prevent overflow
    shifted = np.clip(shifted, -50, 50)
    log_sum = np.log(np.sum(np.exp(shifted)))
    smooth_max = max_r + temperature * log_sum
    return smooth_max

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

def nelder_mead_refine(points, indices, time_budget):
    """
    Refines a subset of points using Nelder-Mead to minimize the global covering radius.
    Uses smooth objective for better convergence.
    """
    start = time.time()
    initial_radius = get_covering_radius(points)
    
    # Flatten the subset of points
    x0 = points[indices].flatten()
    
    def objective(x_flat):
        if time.time() - start > time_budget:
            return initial_radius
        
        new_points = points.copy()
        new_points[indices] = x_flat.reshape(-1, 3)
        # Normalize
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points /= norms
        
        # Check for invalid states (coincident points etc)
        if np.any(norms < 1e-6):
            return np.pi
            
        # Use smooth objective for better gradient information
        return smooth_covering_radius(new_points, temperature=0.005)
        
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        
        # Verify improvement with exact radius
        new_points = points.copy()
        new_points[indices] = res.x.reshape(-1, 3)
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        exact_radius = get_covering_radius(new_points)
        
        if exact_radius < initial_radius:
            return new_points
    except Exception:
        pass
        
    return points

def polish(points, time_budget):
    """
    High-precision polish pass.
    """
    start = time.time()
    alpha = 0.005
    best_local = points.copy()
    best_local_radius = get_covering_radius(points)
    
    for step in range(3000):
        if time.time() - start > time_budget:
            break
            
        radii, centers, simplices = compute_facet_data(points)
        if len(radii) == 0:
            break
            
        max_radius = np.max(radii)
        
        if max_radius < best_local_radius:
            best_local_radius = max_radius
            best_local = points.copy()
            
        # Identify worst facets
        n_worst = max(3, len(radii) // 10)
        worst_indices = np.argsort(radii)[-n_worst:]
        mask = np.zeros(len(radii), dtype=bool)
        mask[worst_indices] = True
            
        if not np.any(mask):
            break
            
        updates = np.zeros_like(points)
        
        for m in np.where(mask)[0]:
            simplex = simplices[m]
            i_idx, j_idx, k_idx = simplex
            
            a = points[i_idx]
            b = points[j_idx]
            c = points[k_idx]
            
            center = centers[m]
            
            # Scale step by how much this facet exceeds the median
            median_radius = np.median(radii)
            excess = max(0, radii[m] - median_radius)
            step_size = alpha * (radii[m] + 0.1 * excess)
            
            updates[i_idx] += step_size * (center - a)
            updates[j_idx] += step_size * (center - b)
            updates[k_idx] += step_size * (center - c)
        
        points += updates
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Decay step size
        alpha *= 0.998
        
        if alpha < 1e-6:
            break
            
    # Nelder-Mead refinement on worst points
    if time.time() - start < time_budget - 3.0:
        radii, _, simplices = compute_facet_data(best_local)
        if len(radii) > 0:
            n_worst = max(5, len(radii) // 5)
            worst_facets = np.argsort(radii)[-n_worst:]
            worst_indices = set()
            for f in worst_facets:
                worst_indices.update(simplices[f].tolist())
            worst_indices = list(worst_indices)
            
            if worst_indices:
                remaining = time_budget - (time.time() - start) - 2.0
                if remaining > 1.0:
                    refined = nelder_mead_refine(best_local, worst_indices, remaining)
                    ref_radius = get_covering_radius(refined)
                    if ref_radius < best_local_radius:
                        best_local = refined
                        best_local_radius = ref_radius
    
    return best_local

def construct():
    start_time = time.time()
    time_limit = 145.0
    
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
    
    # More randomized restarts with varied noise levels
    for s in range(20):
        np.random.seed(1000 + s * 7)
        noise_scale = 0.05 + 0.15 * (s / 20)
        init = init_fib + np.random.randn(N, 3) * noise_scale
        init /= np.linalg.norm(init, axis=1, keepdims=True)
        initializations.append(init)
    
    max_restarts = 25
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if restart < len(initializations):
            points = initializations[restart].copy()
        else:
            np.random.seed(restart * 137)
            points = init_fib + np.random.randn(N, 3) * 0.3
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Local Optimization
        alpha = 0.03
        no_improve = 0
        stuck_counter = 0
        
        for step in range(5000):
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
                stuck_counter = 0
            else:
                no_improve += 1
                if no_improve > 30:
                    alpha *= 0.92
                    if alpha < 1e-5:
                        break
                if no_improve > 150:
                    # Try a small random perturbation to escape local minimum
                    np.random.seed(step * 31 + restart * 17)
                    perturbation = np.random.randn(N, 3) * 0.01
                    points = points + perturbation
                    points /= np.linalg.norm(points, axis=1, keepdims=True)
                    no_improve = 0
                    stuck_counter += 1
                    if stuck_counter > 5:
                        break
            
            # Identify facets to move - top 15% or at least 5 facets
            n_worst = max(5, len(radii) // 7)
            worst_indices = np.argsort(radii)[-n_worst:]
            mask = np.zeros(len(radii), dtype=bool)
            mask[worst_indices] = True
            
            if not np.any(mask):
                max_idx = np.argmax(radii)
                mask = np.zeros(len(radii), dtype=bool)
                mask[max_idx] = True
            
            # Accumulate moves for vertices with weighted steps
            updates = np.zeros_like(points)
            
            for m in np.where(mask)[0]:
                simplex = simplices[m]
                i_idx, j_idx, k_idx = simplex
                
                a = points[i_idx]
                b = points[j_idx]
                c = points[k_idx]
                
                center = centers[m]
                
                # Weight by how much this facet is above average
                avg_radius = np.mean(radii)
                weight = 1.0 + 0.5 * max(0, (radii[m] - avg_radius) / avg_radius)
                step_size = alpha * radii[m] * weight
                
                updates[i_idx] += step_size * (center - a)
                updates[j_idx] += step_size * (center - b)
                updates[k_idx] += step_size * (center - c)
            
            points += updates
            points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Final polish pass on the best configuration found
    if best_points is not None and time.time() - start_time < time_limit - 3.0:
        remaining_time = time_limit - 3.0 - (time.time() - start_time)
        if remaining_time > 3.0:
            polished = polish(best_points, remaining_time)
            # Check if polished is better
            radii, _, _ = compute_facet_data(polished)
            if len(radii) > 0:
                pol_radius = np.max(radii)
                if pol_radius < best_radius:
                    best_radius = pol_radius
                    best_points = polished
    
    if best_points is None:
        best_points = init_fib
        
    return best_points
