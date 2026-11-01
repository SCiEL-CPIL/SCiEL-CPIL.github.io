import numpy as np
import scipy.spatial
import time

N = 50

def _compute_facet_data(points):
    """
    Compute convex hull and facet circumcenters/radii.
    Returns: hull, centers (array), radii (array)
    """
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms < 1e-10] = 1.0
    points = points / norms
    
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        return None, None, None
    
    n_facets = len(hull.simplices)
    centers = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    for i, simplex in enumerate(hull.simplices):
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        ab = np.dot(a, b)
        ac = np.dot(a, c)
        bc = np.dot(b, c)
        
        Gram = np.array([
            [1.0, ab, ac],
            [ab, 1.0, bc],
            [ac, bc, 1.0]
        ])
        
        try:
            m = np.linalg.solve(Gram, np.ones(3))
            c_dir = m[0]*a + m[1]*b + m[2]*c
            norm_c_dir = np.linalg.norm(c_dir)
            if norm_c_dir < 1e-10:
                # Degenerate case, use centroid direction as fallback
                c_dir = a + b + c
                norm_c_dir = np.linalg.norm(c_dir)
                if norm_c_dir < 1e-10:
                    continue
            centers[i] = c_dir / norm_c_dir
            # Radius is angle between center and vertex
            cos_r = np.dot(centers[i], a)
            radii[i] = np.arccos(np.clip(cos_r, -1.0, 1.0))
        except np.linalg.LinAlgError:
            continue
    
    return hull, centers, radii

def _compute_objective_and_grad(points, beta=50.0, lambda_var=0.1):
    """
    Computes a smooth objective function and its gradient.
    Objective: Soft-max of radii + variance penalty.
    This encourages reducing the max radius while also balancing the radii of critical facets.
    """
    hull, centers, radii = _compute_facet_data(points)
    
    if hull is None:
        return np.pi, np.zeros_like(points)
    
    if len(radii) == 0:
        return np.pi, np.zeros_like(points)
    
    max_r = np.max(radii)
    # Soft-max for stability
    exp_terms = np.exp(beta * (radii - max_r))
    sum_exp = np.sum(exp_terms)
    if sum_exp < 1e-10:
        sum_exp = 1e-10
    soft_radius = max_r + np.log(sum_exp) / beta
    
    # Weights for softmax
    weights = exp_terms / sum_exp
    
    # Variance term
    mean_r = np.mean(radii)
    var = np.sum((radii - mean_r) ** 2) / len(radii)
    
    objective = soft_radius + lambda_var * var
    
    # Gradient calculation
    grads = np.zeros_like(points)
    
    # Gradient of soft-max: sum(w_i * grad(r_i))
    # Gradient of variance: 2 * (r_i - mean_r) / N * grad(r_i)
    # Combined weight for facet i: w_i + (2 * lambda_var / N) * (r_i - mean_r)
    
    n_facets = len(radii)
    for i in range(n_facets):
        r = radii[i]
        if r < 1e-8:
            continue
        
        # Base weight from soft-max
        w_sm = weights[i]
        # Weight from variance penalty
        w_var = (2.0 * lambda_var / n_facets) * (r - mean_r)
        
        total_w = w_sm + w_var
        
        if abs(total_w) < 1e-12:
            continue
            
        C = centers[i]
        sin_r = np.sin(r)
        if sin_r < 1e-10:
            continue
            
        # Gradient of radius r wrt point p is approx -C / sin(r) (tangent component)
        # Note: This gradient is in R^3. We will project onto tangent space later or just normalize.
        grad_contrib = -C / sin_r
        
        simplex = hull.simplices[i]
        for k in range(3):
            p_idx = simplex[k]
            grads[p_idx] += total_w * grad_contrib
            
    return objective, grads

def _local_search(points, time_budget, beta=50.0, lambda_var=0.1):
    """
    Local search using Nesterov Accelerated Gradient Descent with adaptive step size.
    """
    current_points = points.copy()
    velocity = np.zeros_like(current_points)
    
    # Initial evaluation
    current_obj, grads = _compute_objective_and_grad(current_points, beta, lambda_var)
    best_obj = current_obj
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.05
    min_step = 1e-8
    decay = 0.9
    momentum = 0.9
    
    # Nesterov parameters
    t = 1.0
    prev_t = 1.0
    
    while time.time() - start_time < time_budget:
        # Check time
        if time.time() - start_time > time_budget:
            break
            
        # Nesterov lookahead
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = current_points + (t - 1) / t_new * velocity
        
        # Compute gradient at y
        _, grads_y = _compute_objective_and_grad(y, beta, lambda_var)
        
        # Update velocity
        velocity = (t / t_new) * velocity + (step_size / t_new) * grads_y
        
        # Update position
        new_points = current_points - velocity
        
        # Retract to sphere
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        new_points = new_points / norms
        
        # Evaluate new point
        new_obj, _ = _compute_objective_and_grad(new_points, beta, lambda_var)
        
        if new_obj < current_obj:
            current_points = new_points
            current_obj = new_obj
            t = t_new
            # Increase step size slightly if improvement
            step_size = min(step_size * 1.05, 0.2)
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_points = current_points.copy()
        else:
            # Decrease step size
            step_size *= decay
            if step_size < min_step:
                # Try a small random perturbation to escape local minimum
                if np.random.rand() < 0.1:
                    noise = np.random.normal(scale=0.01, size=current_points.shape)
                    perturbed = current_points + noise
                    perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
                    p_obj, _ = _compute_objective_and_grad(perturbed, beta, lambda_var)
                    if p_obj < current_obj:
                        current_points = perturbed
                        current_obj = p_obj
                        velocity = np.zeros_like(current_points)
                        step_size = 0.01
                    else:
                        break
            
    return best_points, best_obj

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_points = None
    best_radius = np.inf
    
    # Seeds
    seeds = []
    
    # 1. Fibonacci Sphere (good initial distribution)
    j = np.arange(N) + 0.5
    z = 1.0 - 2.0 * j / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * j
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    seeds.append(fib)
    
    # 2. Random starts
    for seed_val in [42, 1, 2, 3, 7, 13, 99, 123]:
        rng = np.random.default_rng(seed_val)
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        seeds.append(pts)
    
    # Phase 1: Parallel-ish search on seeds with limited time
    # We don't have parallelism, so we allocate short time to each
    n_seeds = len(seeds)
    
    # Allocate 5 seconds to each seed initially
    time_per_seed = 5.0
    
    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining < 10:
            break
            
        # Adjust time per seed if we are running out of time
        current_time_budget = min(time_per_seed, remaining / (n_seeds - i))
        
        if current_time_budget < 1.0:
            break
            
        # Use lower beta and lower variance weight for initial search to find basin
        pts, obj = _local_search(seed_pts, time_budget=current_time_budget, beta=30.0, lambda_var=0.05)
        
        # Get actual radius for comparison
        _, _, radii = _compute_facet_data(pts)
        if len(radii) > 0:
            rad = np.max(radii)
            if rad < best_radius:
                best_radius = rad
                best_points = pts.copy()
    
    # Phase 2: Refine the best candidate found
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining > 10:
            # Use higher beta and higher variance weight for precision
            # This sharpens the objective towards the true max and balances facets
            time_budget = remaining - 5.0
            pts, obj = _local_search(best_points, time_budget=time_budget, beta=100.0, lambda_var=0.2)
            
            _, _, radii = _compute_facet_data(pts)
            if len(radii) > 0:
                rad = np.max(radii)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
                    
    # Fallback
    if best_points is None:
        best_points = seeds[0]
        
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
