import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def spectral_refinement(points, n_steps=50, learning_rate=0.1):
    """
    Spectral refinement: use Laplacian eigenmaps on the Delaunay triangulation
    to smooth point positions while preserving local structure.
    This helps escape local minima by applying a global smoothing operation.
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return points
    
    # Build adjacency from Delaunay triangulation
    adj = np.zeros((N, N))
    for tri in simplices:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adj[tri[i], tri[j]] = 1.0
    
    # Compute Laplacian
    degree = adj.sum(axis=1)
    L = np.diag(degree) - adj
    
    # Solve (L + lambda*I) x = L * points for smoothing
    # Use iterative approach with multiple eigenmodes
    smoothed = points.copy()
    for step in range(n_steps):
        # Compute Laplacian smoothing
        Lp = L @ smoothed  # shape (N, 3)
        # Update: move towards smoothed version
        smoothed = smoothed - learning_rate * Lp
        # Project back to unit sphere
        norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        smoothed /= norms
    
    return smoothed

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

    # All valid
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
        candidates = np.random.randn(1000, 3)
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

def lloyd_step(points, alpha=0.5):
    """
    Perform one step of Lloyd's algorithm.
    Move each point towards the centroid of its Voronoi cell (average of adjacent facet centers).
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return points

    radii, centers, simplices = compute_facet_data(points)
    
    if len(radii) == 0:
        return points

    updates = np.zeros_like(points)
    counts = np.zeros(N)
    
    for m in range(len(centers)):
        idx = simplices[m]
        i_idx, j_idx, k_idx = idx
        updates[i_idx] += centers[m]
        updates[j_idx] += centers[m]
        updates[k_idx] += centers[m]
        counts[i_idx] += 1
        counts[j_idx] += 1
        counts[k_idx] += 1
    
    valid = counts > 0
    updates[valid] /= counts[valid, np.newaxis]
    
    new_points = points.copy()
    new_points[valid] = points[valid] + alpha * (updates[valid] - points[valid])
    
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    new_points /= norms
    
    return new_points

def smooth_objective_and_grad(points):
    """
    Compute a smooth surrogate for the max facet radius and its gradient.
    Uses log-sum-exp of radii combined with an equiangularity variance term.
    Returns:
        value: float (approximation of max radius + variance penalty)
        grad: array of shape (N, 3)
    """
    radii, centers, simplices = compute_facet_data(points)
    if len(radii) == 0:
        return 0.0, np.zeros_like(points)
    
    # Adaptive beta: larger beta makes it sharper (closer to max)
    beta = 50.0
    
    # J = (1/beta) * log(sum(exp(beta * r_i)))
    max_r = np.max(radii)
    scaled = beta * (radii - max_r)
    exp_terms = np.exp(scaled)
    sum_exp = np.sum(exp_terms)
    
    value_lse = max_r + np.log(sum_exp) / beta
    
    # Add equiangularity variance penalty
    # Variance of radii encourages uniformity
    mean_r = np.mean(radii)
    var_r = np.var(radii)
    value = value_lse + 0.1 * var_r  # Small penalty for non-uniformity
    
    # Gradient of log-sum-exp part
    weights = exp_terms / sum_exp
    
    grad_points = np.zeros_like(points)
    
    for m in range(len(radii)):
        if weights[m] < 1e-12:
            continue
            
        simplex = simplices[m]
        i_idx, j_idx, k_idx = simplex
        
        center = centers[m]
        r = radii[m]
        
        sin_r = np.sin(r)
        if sin_r < 1e-6:
            continue
            
        # dr/dp = -u / sin(r)
        coeff = (weights[m] / beta) * (-1.0 / sin_r)
        
        grad_points[i_idx] += coeff * center
        grad_points[j_idx] += coeff * center
        grad_points[k_idx] += coeff * center
    
    # Gradient of variance term: d(var)/dp_i = 2*(r_i - mean) * dr_i/dp_i / N
    # Approximate by adding small correction for each point's contribution
    # For each facet, the variance gradient contribution is proportional to (r_m - mean)
    var_grad_scale = 0.2 / N  # 2 * 0.1 / N
    
    for m in range(len(radii)):
        r = radii[m]
        dr = r - mean_r
        if abs(dr) < 1e-8:
            continue
            
        simplex = simplices[m]
        i_idx, j_idx, k_idx = simplex
        center = centers[m]
        
        sin_r = np.sin(r)
        if sin_r < 1e-6:
            continue
            
        # d(var)/dp = 2*(r-m)/N * dr/dp = 2*(r-m)/N * (-u/sin(r))
        coeff_v = var_grad_scale * dr * (-1.0 / sin_r)
        
        grad_points[i_idx] += coeff_v * center
        grad_points[j_idx] += coeff_v * center
        grad_points[k_idx] += coeff_v * center
    
    return value, grad_points

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
        
        # Phase 1: Lloyd's algorithm for quick coarse optimization
        # This is fast and helps escape bad local minima
        if time.time() - start_time < time_limit - 10.0:
            for lloyd_iter in range(50):
                if time.time() - start_time > time_limit - 10.0:
                    break
                points = lloyd_step(points, alpha=0.5)
                # Check improvement
                radii, _, _ = compute_facet_data(points)
                if len(radii) > 0:
                    max_r = np.max(radii)
                    if max_r < best_radius:
                        best_radius = max_r
                        best_points = points.copy()

        # Phase 2: Smooth Gradient Descent using Log-Sum-Exp surrogate
        # This helps balance all facets rather than just the worst one.
        if time.time() - start_time < time_limit - 5.0:
            alpha = 0.05
            no_improve = 0
            
            for step in range(3000):
                if time.time() - start_time > time_limit - 5.0:
                    break
                
                val, grad = smooth_objective_and_grad(points)
                
                # Also compute exact max radius for tracking
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
                
                # Project gradient to tangent plane
                grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
                
                # Adaptive step size based on gradient magnitude
                grad_norm = np.linalg.norm(grad_tangent)
                if grad_norm > 1e-10:
                    step_size = alpha * min(1.0, 0.1 / grad_norm)
                else:
                    step_size = 0.0
                
                points += step_size * grad_tangent
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                
                # Periodic spectral refinement to escape local minima
                if step % 200 == 199 and time.time() - start_time < time_limit - 10.0:
                    points = spectral_refinement(points, n_steps=20, learning_rate=0.05)
                    radii_check, _, _ = compute_facet_data(points)
                    if len(radii_check) > 0:
                        max_r_check = np.max(radii_check)
                        if max_r_check < best_radius:
                            best_radius = max_r_check
                            best_points = points.copy()
                
                alpha *= 0.998

        # Phase 3: High-precision polish with discrete max-facet logic
        # This refines the solution found by the smooth phase.
        if time.time() - start_time < time_limit - 2.0:
            alpha_polish = 0.01
            no_improve_polish = 0
            for step in range(2000):
                if time.time() - start_time > time_limit - 1.0:
                    break
                
                radii, centers, simplices = compute_facet_data(points)
                if len(radii) == 0:
                    break
                
                max_radius = np.max(radii)
                
                if max_radius < best_radius:
                    best_radius = max_radius
                    best_points = points.copy()
                    no_improve_polish = 0
                else:
                    no_improve_polish += 1
                    if no_improve_polish > 100:
                        alpha_polish *= 0.9
                        if alpha_polish < 1e-5:
                            break
                    if no_improve_polish > 300:
                        break
                
                # Use weighted approach: weight by (radius - median) for facets above median
                med = np.median(radii)
                above = radii > med
                if not np.any(above):
                    break
                
                # Weight proportional to how much above median
                weights_p = np.zeros(len(radii))
                weights_p[above] = (radii[above] - med)
                if np.sum(weights_p) < 1e-10:
                    break
                weights_p /= np.sum(weights_p)
                
                updates = np.zeros_like(points)
                
                for m in np.where(above)[0]:
                    simplex = simplices[m]
                    i_idx, j_idx, k_idx = simplex
                    
                    center = centers[m]
                    w = weights_p[m]
                    step_size = alpha_polish * w
                    
                    # Move towards center but only tangentially
                    # The direction from point to center projected onto tangent plane
                    for idx in [i_idx, j_idx, k_idx]:
                        p = points[idx]
                        # Tangential component of (center - p)
                        radial = np.dot(center - p, p)
                        tangent_dir = (center - p) - radial * p
                        norm_td = np.linalg.norm(tangent_dir)
                        if norm_td > 1e-10:
                            tangent_dir /= norm_td
                        updates[idx] += step_size * tangent_dir
                    
                # Project to tangent (should already be mostly tangential)
                for i in range(N):
                    proj = np.dot(updates[i], points[i])
                    updates[i] -= proj * points[i]
                
                points += updates
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                
                alpha_polish *= 0.998

    if best_points is None:
        best_points = init_fib
    
    # Final high-precision polish: focus only on the single worst facet
    # Use a more sophisticated approach: optimize the 3 vertices of the worst
    # facet to minimize the circumradius directly
    if time.time() - start_time < time_limit - 0.5:
        final_points = best_points.copy()
        
        # First, do a broader polish moving all points slightly towards their
        # Voronoi centroids to improve overall uniformity
        for step in range(100):
            if time.time() - start_time > time_limit - 0.5:
                break
            
            radii, centers, simplices = compute_facet_data(final_points)
            if len(radii) == 0:
                break
            
            # Move each point towards the average of its adjacent facet centers
            updates = np.zeros_like(final_points)
            counts = np.zeros(N)
            
            for m in range(len(centers)):
                idx = simplices[m]
                i_idx, j_idx, k_idx = idx
                updates[i_idx] += centers[m]
                updates[j_idx] += centers[m]
                updates[k_idx] += centers[m]
                counts[i_idx] += 1
                counts[j_idx] += 1
                counts[k_idx] += 1
            
            valid = counts > 0
            updates[valid] /= counts[valid, np.newaxis]
            
            # Project to tangent plane
            for i in range(N):
                proj = np.dot(updates[i], final_points[i])
                updates[i] -= proj * final_points[i]
            
            final_points += 0.01 * updates
            final_points /= np.linalg.norm(final_points, axis=1, keepdims=True)
            
            radii_check, _, _ = compute_facet_data(final_points)
            if len(radii_check) > 0:
                max_r_check = np.max(radii_check)
                if max_r_check < best_radius:
                    best_radius = max_r_check
                    best_points = final_points.copy()
        
        # Then focus on the worst facet with finer steps
        for step in range(300):
            if time.time() - start_time > time_limit - 0.1:
                break
            
            radii, centers, simplices = compute_facet_data(final_points)
            if len(radii) == 0:
                break
            
            max_idx = np.argmax(radii)
            max_r = radii[max_idx]
            
            if max_r < best_radius:
                best_radius = max_r
                best_points = final_points.copy()
            
            # Only move vertices of the worst facet
            simplex = simplices[max_idx]
            i_idx, j_idx, k_idx = simplex
            center = centers[max_idx]
            
            updates = np.zeros_like(final_points)
            step_size = 0.002
            
            for idx in [i_idx, j_idx, k_idx]:
                p = final_points[idx]
                radial = np.dot(center - p, p)
                tangent_dir = (center - p) - radial * p
                norm_td = np.linalg.norm(tangent_dir)
                if norm_td > 1e-10:
                    tangent_dir /= norm_td
                updates[idx] += step_size * tangent_dir
            
            final_points += updates
            final_points /= np.linalg.norm(final_points, axis=1, keepdims=True)
    
    return best_points
