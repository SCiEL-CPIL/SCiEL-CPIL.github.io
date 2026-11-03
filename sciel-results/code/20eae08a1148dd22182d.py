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

def get_covering_radius(points):
    """Compute the covering radius of the configuration."""
    radii, centers, simplices = compute_facet_data(points)
    if len(radii) == 0:
        return np.pi
    return np.max(radii)

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

def lloyd_step(points, alpha=0.1):
    """
    Lloyd-like step: for each facet, move its vertices towards the facet's circumcenter.
    """
    radii, centers, simplices = compute_facet_data(points)
    if len(radii) == 0:
        return points
    
    updates = np.zeros_like(points)
    
    for m in range(len(radii)):
        simplex = simplices[m]
        i_idx, j_idx, k_idx = simplex
        
        center = centers[m]
        # Weight by how large this facet is relative to others
        weight = radii[m]
        
        updates[i_idx] += weight * (center - points[i_idx])
        updates[j_idx] += weight * (center - points[j_idx])
        updates[k_idx] += weight * (center - points[k_idx])
    
    # Project to tangent plane
    for i in range(N):
        proj = np.dot(updates[i], points[i])
        updates[i] -= proj * points[i]
    
    points = points + alpha * updates
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    return points

def facet_push_step(points, alpha=0.05, top_fraction=0.1):
    """
    Push only the largest facets' vertices towards their circumcenters.
    """
    radii, centers, simplices = compute_facet_data(points)
    if len(radii) == 0:
        return points
    
    n_facets = len(radii)
    n_top = max(1, int(n_facets * top_fraction))
    
    # Get indices of top facets
    top_indices = np.argsort(radii)[-n_top:]
    
    updates = np.zeros_like(points)
    
    for m in top_indices:
        simplex = simplices[m]
        i_idx, j_idx, k_idx = simplex
        
        center = centers[m]
        weight = radii[m]
        
        updates[i_idx] += weight * (center - points[i_idx])
        updates[j_idx] += weight * (center - points[j_idx])
        updates[k_idx] += weight * (center - points[k_idx])
    
    # Project to tangent plane
    for i in range(N):
        proj = np.dot(updates[i], points[i])
        updates[i] -= proj * points[i]
    
    points = points + alpha * updates
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    return points

def smooth_objective_and_grad(points, tau=0.1):
    """
    Compute the log-sum-exp surrogate objective and its gradient.
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
    
    objective = r_max + np.log(sum_exp)
    
    # Gradient: sum_f w_f * grad(r_f)
    # grad r_f wrt p_i = -u_f / sin(r_f)
    updates = np.zeros_like(points)
    
    sin_r = np.sin(radii)
    sin_r[sin_r < 1e-6] = 1e-6
    
    for m in range(len(radii)):
        simplex = simplices[m]
        i_idx, j_idx, k_idx = simplex
        
        center = centers[m]
        w = weights[m]
        factor = w / sin_r[m]
        
        # Gradient direction is -u / sin(r), so descent adds +u/sin(r)
        updates[i_idx] += factor * center
        updates[j_idx] += factor * center
        updates[k_idx] += factor * center
    
    # Project to tangent plane
    for i in range(N):
        proj = np.dot(updates[i], points[i])
        updates[i] -= proj * points[i]
    
    return objective, updates

def polish(points, time_limit, start_time):
    """
    High-precision polish using adaptive step sizes and multiple strategies.
    """
    best_points = points.copy()
    best_radius = get_covering_radius(points)
    
    alpha = 0.05
    tau = 0.05
    
    for step in range(2000):
        if time.time() - start_time > time_limit:
            break
        
        # Alternate between smooth gradient and facet push
        if step % 3 == 0:
            # Smooth gradient descent
            obj, grad = smooth_objective_and_grad(points, tau)
            
            # Adaptive step size
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-8:
                alpha *= 0.9
                continue
            
            points_new = points + alpha * grad
            points_new /= np.linalg.norm(points_new, axis=1, keepdims=True)
            
            radius_new = get_covering_radius(points_new)
            
            if radius_new < best_radius:
                best_radius = radius_new
                best_points = points_new.copy()
                points = points_new
                alpha = min(alpha * 1.1, 0.2)
            else:
                alpha *= 0.8
                if alpha < 1e-6:
                    break
        else:
            # Facet push step
            points_new = facet_push_step(points, alpha=alpha * 0.5, top_fraction=0.05)
            radius_new = get_covering_radius(points_new)
            
            if radius_new < best_radius:
                best_radius = radius_new
                best_points = points_new.copy()
                points = points_new
            else:
                alpha *= 0.9
    
    return best_points, best_radius

def nelder_mead_refine(points, time_limit, start_time, max_iter=500):
    """
    Nelder-Mead style refinement on the sphere using the covering radius as objective.
    This works on the full configuration but uses a simplified approach.
    """
    best_points = points.copy()
    best_radius = get_covering_radius(points)
    
    # Create initial simplex by perturbing points
    simplex = [points.copy()]
    step_size = 0.01
    for _ in range(3):
        perturbation = np.random.randn(N, 3) * step_size
        p = points + perturbation
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        simplex.append(p)
    
    # Evaluate initial simplex
    radii_list = [get_covering_radius(p) for p in simplex]
    
    for iter in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Sort by objective
        order = np.argsort(radii_list)
        simplex = [simplex[i] for i in order]
        radii_list = [radii_list[i] for i in order]
        
        # Check convergence
        if radii_list[-1] - radii_list[0] < 1e-10:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha_ref = 1.0
        reflected = centroid + alpha_ref * (centroid - simplex[-1])
        reflected /= np.linalg.norm(reflected, axis=1, keepdims=True)
        ref_radius = get_covering_radius(reflected)
        
        if ref_radius < radii_list[-2]:
            # Accept reflection
            simplex[-1] = reflected
            radii_list[-1] = ref_radius
            continue
        elif ref_radius < radii_list[0]:
            # Expansion
            gamma_exp = 2.0
            expanded = centroid + gamma_exp * (centroid - simplex[-1])
            expanded /= np.linalg.norm(expanded, axis=1, keepdims=True)
            exp_radius = get_covering_radius(expanded)
            
            if exp_radius < ref_radius:
                simplex[-1] = expanded
                radii_list[-1] = exp_radius
            else:
                simplex[-1] = reflected
                radii_list[-1] = ref_radius
            continue
        else:
            # Contraction
            beta_con = 0.5
            contracted = centroid + beta_con * (simplex[-1] - centroid)
            contracted /= np.linalg.norm(contracted, axis=1, keepdims=True)
            con_radius = get_covering_radius(contracted)
            
            if con_radius < radii_list[-1]:
                simplex[-1] = contracted
                radii_list[-1] = con_radius
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    simplex[i] /= np.linalg.norm(simplex[i], axis=1, keepdims=True)
                    radii_list[i] = get_covering_radius(simplex[i])
    
    # Return best point found
    best_idx = np.argmin(radii_list)
    best_points = simplex[best_idx]
    best_radius = radii_list[best_idx]
    
    return best_points, best_radius

def spectral_refinement(points, time_limit, start_time):
    """
    Use eigendecomposition to find optimal directions for point movement.
    This is a heuristic that may help escape local minima.
    """
    best_points = points.copy()
    best_radius = get_covering_radius(points)
    
    # Compute covariance of facet centers weighted by radius
    radii, centers, simplices = compute_facet_data(points)
    if len(radii) == 0:
        return best_points, best_radius
    
    # Weighted average direction
    weights = radii / np.sum(radii)
    avg_center = np.sum(centers * weights[:, np.newaxis], axis=0)
    avg_center /= np.linalg.norm(avg_center)
    
    # Try moving all points slightly towards the average center direction
    for trial_alpha in [0.001, 0.005, 0.01, 0.02]:
        if time.time() - start_time > time_limit:
            break
        
        # Project average center to tangent plane for each point
        updates = np.zeros_like(points)
        for i in range(N):
            proj = np.dot(avg_center, points[i])
            updates[i] = avg_center - proj * points[i]
        
        points_trial = points + trial_alpha * updates
        points_trial /= np.linalg.norm(points_trial, axis=1, keepdims=True)
        
        radius_trial = get_covering_radius(points_trial)
        if radius_trial < best_radius:
            best_radius = radius_trial
            best_points = points_trial.copy()
    
    return best_points, best_radius

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
    for s in range(10):
        np.random.seed(1000 + s * 7)
        init = init_fib + np.random.randn(N, 3) * (0.1 + 0.1 * s / 10)
        init /= np.linalg.norm(init, axis=1, keepdims=True)
        initializations.append(init)
    
    max_restarts = len(initializations)
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        points = initializations[restart].copy()
        
        # Phase 1: Smooth gradient descent using Log-Sum-Exp surrogate
        tau = 0.1
        alpha = 0.05
        no_improve = 0
        
        for step in range(1500):
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
                if no_improve > 30:
                    alpha *= 0.9
                    if alpha < 1e-4:
                        break
                if no_improve > 100:
                    break
            
            # Compute soft-max weights
            r_scaled = radii / tau
            r_max = np.max(r_scaled)
            exp_terms = np.exp(r_scaled - r_max)
            sum_exp = np.sum(exp_terms)
            weights = exp_terms / sum_exp
            
            # Compute gradients
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
            
            points += alpha * updates
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Phase 2: Lloyd-like steps
        if time.time() - start_time < time_limit - 10.0:
            alpha_lloyd = 0.02
            for step in range(500):
                if time.time() - start_time > time_limit - 5.0:
                    break
                
                points = lloyd_step(points, alpha=alpha_lloyd)
                
                radius = get_covering_radius(points)
                if radius < best_radius:
                    best_radius = radius
                    best_points = points.copy()
                
                alpha_lloyd *= 0.995
        
        # Phase 3: Facet push refinement
        if time.time() - start_time < time_limit - 5.0:
            alpha_push = 0.01
            for step in range(500):
                if time.time() - start_time > time_limit - 2.0:
                    break
                
                points = facet_push_step(points, alpha=alpha_push, top_fraction=0.05)
                
                radius = get_covering_radius(points)
                if radius < best_radius:
                    best_radius = radius
                    best_points = points.copy()
                
                alpha_push *= 0.995
        
        # Phase 4: High-precision polish
        if time.time() - start_time < time_limit - 2.0:
            points, radius = polish(points, time_limit - 2.0, start_time)
            if radius < best_radius:
                best_radius = radius
                best_points = points
        
        # Phase 5: Spectral refinement
        if time.time() - start_time < time_limit - 1.0:
            points, radius = spectral_refinement(points, time_limit - 1.0, start_time)
            if radius < best_radius:
                best_radius = radius
                best_points = points
    
    if best_points is None:
        best_points = init_fib
        
    return best_points
