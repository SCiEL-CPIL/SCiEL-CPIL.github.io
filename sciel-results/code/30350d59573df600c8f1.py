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
    
    # Compute normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    valid = norms > 1e-10
    
    # Initialize outputs
    M = len(simplices)
    radii = np.full(M, np.pi)
    centers = np.zeros((M, 3))
    centers[~valid] = [0, 0, 1]
    
    if not np.all(valid):
        # For valid facets, compute normally, for invalid keep defaults
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
    
    # Orient normals outwards (away from origin)
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    return radii, u, simplices

def greedy_init(n_points, seed=42):
    """
    Greedy initialization: Farthest Point Sampling.
    """
    np.random.seed(seed)
    points = [np.random.randn(3)]
    points[0] /= np.linalg.norm(points[0])
    
    for _ in range(n_points - 1):
        # Generate candidates
        candidates = np.random.randn(1000, 3)
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # Find min distance to existing points
        existing = np.array(points)
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
    Move each point towards the centroid of its Voronoi cell.
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

def facet_push_step(points, radii, centers, simplices, T=0.1, percentile=80):
    """
    Push vertices of large facets towards the circumcenter.
    """
    if len(radii) == 0:
        return points
        
    threshold = np.percentile(radii, percentile)
    mask = radii > threshold
    
    if not np.any(mask):
        # If no facets are above threshold, pick the single largest
        max_idx = np.argmax(radii)
        mask = np.zeros(len(radii), dtype=bool)
        mask[max_idx] = True
        
    updates = np.zeros_like(points)
    
    for m in np.where(mask)[0]:
        idx = simplices[m]
        i_idx, j_idx, k_idx = idx
        center = centers[m]
        radius = radii[m]
        
        step = T * radius
        
        updates[i_idx] += step * (center - points[i_idx])
        updates[j_idx] += step * (center - points[j_idx])
        updates[k_idx] += step * (center - points[k_idx])
        
    new_points = points + updates
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    new_points /= norms
    
    return new_points

def smooth_objective_and_grad(points):
    """
    Compute a smooth surrogate for the max facet radius and its gradient.
    Uses log-sum-exp of squared radii.
    Returns:
        value: float
        grad: array of shape (N, 3)
    """
    radii, centers, simplices = compute_facet_data(points)
    if len(radii) == 0:
        return 0.0, np.zeros_like(points)
    
    beta = 100.0
    
    max_r2 = np.max(radii**2)
    scaled = beta * (radii**2 - max_r2)
    
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp
    
    log_sum_exp_val = beta * max_r2 + max_scaled + np.log(sum_exp)
    
    value = np.sqrt(log_sum_exp_val / beta)
    
    grad_points = np.zeros_like(points)
    
    for m in range(len(radii)):
        if weights[m] < 1e-12:
            continue
            
        i_idx, j_idx, k_idx = simplices[m]
        a = points[i_idx]
        b = points[j_idx]
        c = points[k_idx]
        center = centers[m]
        r = radii[m]
        
        sin_r = np.sin(r)
        if sin_r < 1e-6:
            continue
        
        # r = arccos(center · a) where center is the outward unit normal
        # dr/da = -(center · da) / sin(r)  [if center were fixed]
        # But center also depends on a, b, c.
        # For a triangle on the unit sphere, the circumcenter direction is:
        #   u = (a × b + b × c + c × a) / |a × b + b × c + c × a|  (for equilateral-ish)
        # More precisely, u is the normalized cross product of (b-a) and (c-a).
        # 
        # Let's compute du/dp_i for each vertex.
        # u = n/|n| where n = (b-a)×(c-a)
        # du = (dn/|n|) - (n·dn)n/|n|^3
        # 
        # dn/da = -((b-a)×(c-a))_da... this gets complex.
        # 
        # Simpler: use finite-difference-like approach or just the dominant term.
        # The dominant contribution to dr is from the change in the dot product,
        # and the change in u is second-order for small perturbations.
        # 
        # Let's use the first-order approximation: dr ≈ -(center · da) / sin(r)
        # This is valid when the facet doesn't change topology.
        
        coeff = -weights[m] * 2.0 * beta * r / sin_r
        
        grad_points[i_idx] += coeff * center
        grad_points[j_idx] += coeff * center
        grad_points[k_idx] += coeff * center
    
    if value > 1e-12:
        grad_points /= (2.0 * value * beta)
    else:
        grad_points = np.zeros_like(points)
        
    return value, grad_points

def polish(points, time_budget, start_alpha=0.05):
    """
    High-precision polish pass with adaptive step sizes and multiple phases.
    """
    start = time.time()
    best_local = points.copy()
    best_local_radius = np.pi
    
    def check_improve(pts):
        nonlocal best_local, best_local_radius
        radii, _, _ = compute_facet_data(pts)
        if len(radii) > 0:
            max_r = np.max(radii)
            if max_r < best_local_radius:
                best_local_radius = max_r
                best_local = pts.copy()
        return best_local_radius
    
    # Phase 1: Gradient Descent on Smooth Surrogate (40% of time)
    alpha = start_alpha
    phase1_budget = time_budget * 0.4
    for step in range(2000):
        if time.time() - start > phase1_budget:
            break
            
        val, grad = smooth_objective_and_grad(points)
        check_improve(points)
                
        grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        
        points = points + alpha * grad_tangent
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        alpha *= 0.995
        
        if alpha < 1e-6:
            break
            
    points = best_local
    
    # Phase 2: Facet Push with adaptive step (30% of time)
    phase2_budget = time_budget * 0.3
    alpha_push = 0.03
    for step in range(3000):
        if time.time() - start > phase1_budget + phase2_budget:
            break
            
        radii, centers, simplices = compute_facet_data(points)
        if len(radii) == 0:
            break
            
        check_improve(points)
        
        # Adaptive percentile: focus more on worst facets as we converge
        percentile = 95 if best_local_radius < 0.2 else 90
        threshold = np.percentile(radii, percentile)
        mask = radii > threshold
        
        if not np.any(mask):
            max_idx = np.argmax(radii)
            mask = np.zeros(len(radii), dtype=bool)
            mask[max_idx] = True
            
        updates = np.zeros_like(points)
        
        for m in np.where(mask)[0]:
            idx = simplices[m]
            i_idx, j_idx, k_idx = idx
            center = centers[m]
            r = radii[m]
            
            # Scale step by how much this facet exceeds the threshold
            excess = max(0, r - threshold)
            step_size = alpha_push * (r + 0.1 * excess)
            
            updates[i_idx] += step_size * (center - points[i_idx])
            updates[j_idx] += step_size * (center - points[j_idx])
            updates[k_idx] += step_size * (center - points[k_idx])
            
        points = points + updates
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        alpha_push *= 0.997
        
        if alpha_push < 1e-6:
            break
            
    points = best_local
    
    # Phase 3: Nelder-Mead refinement on worst facets (30% of time)
    nelder_mead_refine(points, time_budget - (time.time() - start) - 0.5)
    check_improve(points)
    
    # Phase 4: Final fine-tuning
    remaining = time_budget - (time.time() - start)
    if remaining > 1.0:
        fine_alpha = 0.003
        for step in range(3000):
            if time.time() - start > time_budget - 0.3:
                break
                
            radii, centers, simplices = compute_facet_data(points)
            if len(radii) == 0:
                break
                
            check_improve(points)
            
            # Only work on the top 3 worst facets
            top_k = 3
            worst_indices = np.argsort(radii)[-top_k:]
            
            updates = np.zeros_like(points)
            
            for m in worst_indices:
                idx = simplices[m]
                i_idx, j_idx, k_idx = idx
                center = centers[m]
                r = radii[m]
                
                step_size = fine_alpha * r
                
                updates[i_idx] += step_size * (center - points[i_idx])
                updates[j_idx] += step_size * (center - points[j_idx])
                updates[k_idx] += step_size * (center - points[k_idx])
                
            points = points + updates
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
            fine_alpha *= 0.998
            
            if fine_alpha < 1e-7:
                break
    
    return best_local

def nelder_mead_refine(points, time_budget):
    """
    Nelder-Mead style local refinement focusing on the worst facets.
    Tries small perturbations of individual points and keeps improvements.
    """
    start = time.time()
    best_local = points.copy()
    best_local_radius = np.pi
    
    def get_radius(pts):
        radii, _, _ = compute_facet_data(pts)
        if len(radii) == 0:
            return np.pi
        return np.max(radii)
    
    best_local_radius = get_radius(points)
    
    # Focus on worst facets, try moving their vertices
    for iteration in range(5000):
        if time.time() - start > time_budget:
            break
        
        radii, centers, simplices = compute_facet_data(points)
        if len(radii) == 0:
            break
        
        current_radius = np.max(radii)
        if current_radius < best_local_radius:
            best_local_radius = current_radius
            best_local = points.copy()
        
        # Get the worst facet
        worst_idx = np.argmax(radii)
        i_idx, j_idx, k_idx = simplices[worst_idx]
        center = centers[worst_idx]
        r = radii[worst_idx]
        
        # Try small perturbations of each vertex towards the circumcenter
        for vertex_idx in [i_idx, j_idx, k_idx]:
            for scale in [0.01, 0.005, 0.001]:
                trial = points.copy()
                trial[vertex_idx] = points[vertex_idx] + scale * (center - points[vertex_idx])
                trial[vertex_idx] /= np.linalg.norm(trial[vertex_idx])
                
                trial_radius = get_radius(trial)
                if trial_radius < current_radius - 1e-12:
                    points = trial
                    current_radius = trial_radius
                    break
    
    return best_local

def spectral_refinement(points, time_budget):
    """
    Structural change: Spherical Linear Programming (SLP) dual-inspired refinement.
    
    The covering radius is determined by the largest empty cap. In the dual of the SLP formulation,
    we seek a non-negative measure (weights on the points) such that its spherical harmonic
    expansion dominates a constant function. This ensures that the "potential" generated by the
    points is at least 1 everywhere on the sphere.
    
    We approximate this by:
    1. Assigning weights to points based on their Voronoi cell area (or simply 1/N).
    2. Computing the potential field on a fine grid using a kernel (e.g., Gaussian or inverse distance).
    3. Identifying regions where the potential is below a threshold (indicating poor coverage).
    4. Moving points towards the "holes" (low potential regions) in a manner that balances the potential.
    
    This is a different optimization landscape than minimizing facet circumradii directly,
    which may escape local minima.
    """
    start = time.time()
    best_local = points.copy()
    best_local_radius = np.pi
    
    # Pre-compute a fine grid for potential evaluation
    # Use a Fibonacci sphere for the grid to ensure uniform distribution
    n_grid = 2000
    i_grid = np.arange(n_grid) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / n_grid
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    grid_points = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    # Kernel bandwidth: related to the target covering radius
    # We want the potential to be smooth but responsive to local gaps
    sigma = 0.1  # Initial bandwidth
    
    for step in range(500):
        if time.time() - start > time_budget:
            break
        
        # Compute potential at grid points
        # Potential = sum_j w_j * exp(-||x_i - x_j||^2 / (2*sigma^2))
        # ||x_i - x_j||^2 = 2 - 2*x_i.x_j
        dots = grid_points @ points.T  # (n_grid, N)
        dist_sq = 2.0 - 2.0 * dots
        dist_sq = np.maximum(dist_sq, 0.0)
        
        # Gaussian kernel
        weights = np.exp(-dist_sq / (2.0 * sigma**2))
        potential = np.sum(weights, axis=1)  # (n_grid,)
        
        # Identify low potential regions (holes)
        # We want to move points towards the locations of minimum potential
        # But we also need to consider the current geometry
        
        # Update best local if improved
        radii, _, _ = compute_facet_data(points)
        if len(radii) > 0:
            max_r = np.max(radii)
            if max_r < best_local_radius:
                best_local_radius = max_r
                best_local = points.copy()
        
        # Find the grid points with lowest potential
        # We want to distribute points more evenly in terms of potential
        # Simple heuristic: move each point towards the centroid of the low-potential region nearest to it
        
        # Threshold for "low" potential
        threshold = np.percentile(potential, 20)
        low_mask = potential < threshold
        
        if not np.any(low_mask):
            break
            
        low_points = grid_points[low_mask]
        
        # For each point, find the nearest low-potential region
        # And move it slightly towards that region
        updates = np.zeros_like(points)
        
        for i in range(N):
            # Distance from point i to low-potential grid points
            dists = np.linalg.norm(low_points - points[i], axis=1)
            nearest_idx = np.argmin(dists)
            target = low_points[nearest_idx]
            
            # Move towards target
            # Step size depends on how far the point is from the target and the current radius
            step_size = 0.01
            updates[i] = step_size * (target - points[i])
        
        points = points + updates
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Adapt sigma
        sigma *= 0.99
        
        if sigma < 0.01:
            break
    
    return best_local

def construct():
    start_time = time.time()
    time_limit = 148.0 # Safety margin (evaluator kills at 150s)
    
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
    
    # 2. Greedy Initialization
    init_greedy = greedy_init(N, seed=10)
    initializations.append(init_greedy)
    
    # 3. Random
    np.random.seed(42)
    init_rand = np.random.randn(N, 3)
    init_rand /= np.linalg.norm(init_rand, axis=1, keepdims=True)
    initializations.append(init_rand)
    
    # 4. Greedy 2
    init_greedy2 = greedy_init(N, seed=123)
    initializations.append(init_greedy2)
    
    # 5. Fibonacci + noise
    np.random.seed(123)
    init_fib_noise = init_fib + np.random.randn(N, 3) * 0.2
    init_fib_noise /= np.linalg.norm(init_fib_noise, axis=1, keepdims=True)
    initializations.append(init_fib_noise)
    
    # 6. Greedy 3
    init_greedy3 = greedy_init(N, seed=777)
    initializations.append(init_greedy3)
    
    # 7. Fibonacci + larger noise
    np.random.seed(456)
    init_fib_noise2 = init_fib + np.random.randn(N, 3) * 0.4
    init_fib_noise2 /= np.linalg.norm(init_fib_noise2, axis=1, keepdims=True)
    initializations.append(init_fib_noise2)
    
    max_restarts = len(initializations) + 5
    
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 15.0:  # Leave time for final polish
            break
            
        if restart < len(initializations):
            points = initializations[restart].copy()
        else:
            # Perturb the best found so far
            if best_points is not None:
                np.random.seed(restart * 137 + 7)
                noise_scale = 0.03 + 0.02 * np.exp(-restart * 0.3)
                points = best_points + np.random.randn(N, 3) * noise_scale
            else:
                points = init_greedy + np.random.randn(N, 3) * 0.2
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
        # Optimization loop
        T = 0.05
        alpha_lloyd = 0.5
        no_improve_count = 0
        restart_start = time.time()
        restart_budget = min(20.0, time_limit - elapsed - 15.0)
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > time_limit - 15.0:
                break
            if time.time() - restart_start > restart_budget:
                break
                
            radii, centers, simplices = compute_facet_data(points)
            
            if len(radii) == 0:
                break
                
            max_radius = np.max(radii)
            
            if max_radius < best_radius:
                best_radius = max_radius
                best_points = points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 80:
                    T *= 0.9
                    alpha_lloyd *= 0.95
                if no_improve_count > 150:
                    break
                if T < 1e-7 and alpha_lloyd < 1e-7:
                    break
            
            if no_improve_count % 10 == 0:
                points = lloyd_step(points, alpha=alpha_lloyd)
            else:
                # Adaptive percentile: focus on worst facets
                adaptive_percentile = 85 if max_radius > 0.25 else 75
                points = facet_push_step(points, radii, centers, simplices, T=T, percentile=adaptive_percentile)
                
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            points /= norms
            
            T *= 0.996
            alpha_lloyd *= 0.996
            
        # Apply spectral refinement if time allows
        elapsed = time.time() - start_time
        if elapsed < time_limit - 20.0:
            spec_budget = min(8.0, time_limit - elapsed - 15.0)
            if spec_budget > 3.0:
                polished = spectral_refinement(points, time_budget=spec_budget)
                radii, _, _ = compute_facet_data(polished)
                if len(radii) > 0:
                    max_r = np.max(radii)
                    if max_r < best_radius:
                        best_radius = max_r
                        best_points = polished
    
    # Final Polish - use remaining time
    if best_points is not None:
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 3.0:
            polished = polish(best_points, remaining_time - 0.5)
            radii, _, _ = compute_facet_data(polished)
            if len(radii) > 0:
                max_r = np.max(radii)
                if max_r < best_radius:
                    best_radius = max_r
                    best_points = polished
    else:
        best_points = init_fib
        
    return best_points
