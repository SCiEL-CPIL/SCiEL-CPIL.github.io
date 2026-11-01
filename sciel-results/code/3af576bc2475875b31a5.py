import numpy as np
import scipy.spatial
import time

N = 50

def _compute_facet_data(points):
    """
    Compute convex hull and facet circumcenters/radii in one pass.
    Returns: hull, centers (array), radii (array), worst_idx
    """
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        return None, None, None, None
    
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
                continue
            centers[i] = c_dir / norm_c_dir
            radii[i] = np.arccos(np.clip(np.dot(centers[i], a), -1.0, 1.0))
        except np.linalg.LinAlgError:
            continue
    
    if n_facets > 0 and np.any(radii > 0):
        worst_idx = np.argmax(radii)
    else:
        worst_idx = None
    
    return hull, centers, radii, worst_idx

def _compute_soft_max_radius_and_grads(points, beta=100.0):
    """
    Compute soft-max of facet radii and gradients.
    Soft-max: R_soft = (1/beta) * log(sum(exp(beta * r_i)))
    This is smooth and differentiable.
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    
    if hull is None or worst_idx is None:
        return np.pi, np.zeros_like(points)
    
    # Soft-max radius
    max_r = np.max(radii)
    # Use log-sum-exp for numerical stability
    exp_terms = np.exp(beta * (radii - max_r))
    log_sum = np.log(np.sum(exp_terms))
    soft_radius = max_r + np.log(np.sum(exp_terms)) / beta
    
    # Weights for each facet (softmax of radii)
    weights = exp_terms / np.sum(exp_terms)
    
    # Gradients: d(soft_radius)/d(p_j) = sum_i w_i * d(r_i)/d(p_j)
    # For each facet i with circumcenter C_i and radius r_i:
    # d(r_i)/d(p_j) = -C_i / sin(r_i) for j in facet i (if r_i > 0)
    # This is approximate - ignores that C_i depends on p_j
    
    grads = np.zeros_like(points)
    
    for i in range(len(hull.simplices)):
        if radii[i] < 1e-10 or weights[i] < 1e-12:
            continue
        
        w = weights[i]
        C = centers[i]
        r = radii[i]
        sin_r = np.sin(r)
        if sin_r < 1e-10:
            continue
        
        # Gradient contribution from this facet
        # d(r_i)/d(p_j) ≈ -C / sin(r_i) for vertices j of facet i
        grad_contrib = -C / sin_r
        
        simplex = hull.simplices[i]
        for k in range(3):
            p_idx = simplex[k]
            grads[p_idx] += w * grad_contrib
    
    return soft_radius, grads

def _local_search(points, time_budget, use_soft_max=True, beta=100.0):
    """
    Local search using gradient descent with adaptive step size.
    """
    current_points = points.copy()
    
    if use_soft_max:
        current_radius, _ = _compute_soft_max_radius_and_grads(current_points, beta)
    else:
        _, centers, radii, worst_idx = _compute_facet_data(current_points)
        current_radius = np.max(radii) if worst_idx is not None else np.pi
    
    best_radius = current_radius
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.03
    min_step = 1e-6
    decay = 0.95
    
    while time.time() - start_time < time_budget:
        if use_soft_max:
            radius, grads = _compute_soft_max_radius_and_grads(current_points, beta)
        else:
            # Fallback to hard max with simple gradient
            hull, centers, radii, worst_idx = _compute_facet_data(current_points)
            if hull is None or worst_idx is None:
                break
            radius = radii[worst_idx]
            grads = np.zeros_like(current_points)
            C = centers[worst_idx]
            r = radii[worst_idx]
            if r > 1e-10:
                sin_r = np.sin(r)
                if sin_r > 1e-10:
                    grad_contrib = -C / sin_r
                    simplex = hull.simplices[worst_idx]
                    for k in range(3):
                        p_idx = simplex[k]
                        grads[p_idx] += grad_contrib
        
        if radius < best_radius - 1e-12:
            best_radius = radius
            best_points = current_points.copy()
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-12:
            break
        
        grads = grads / grad_norm
        
        # Try step
        new_points = current_points + step_size * grads
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        # Check if improvement
        if use_soft_max:
            new_radius, _ = _compute_soft_max_radius_and_grads(new_points, beta)
        else:
            _, _, new_radii, new_worst = _compute_facet_data(new_points)
            new_radius = np.max(new_radii) if new_worst is not None else np.pi
        
        if new_radius < radius:
            current_points = new_points
            step_size = min(step_size * 1.1, 0.1)  # Increase step slightly
        else:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_points = None
    best_radius = np.inf
    
    # Seeds
    seeds = []
    
    # 1. Fibonacci Sphere
    j = np.arange(N) + 0.5
    z = 1.0 - 2.0 * j / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * j
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    seeds.append(fib)
    
    # 2. Random starts with different seeds
    for seed_val in [42, 1, 2, 3, 7, 13]:
        rng = np.random.default_rng(seed_val)
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        seeds.append(pts)
    
    # Process seeds with adaptive time allocation
    n_seeds = len(seeds)
    base_time_per_seed = 8.0  # 8 seconds per seed initially
    
    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining < 10:
            break
        
        # Allocate time: more time for later seeds if we're doing well
        if i < n_seeds - 1:
            time_budget = min(base_time_per_seed, remaining / (n_seeds - i))
        else:
            time_budget = min(30.0, remaining)  # Last seed gets more time
        
        if time_budget < 2.0:
            break
        
        t_start = time.time()
        pts, rad = _local_search(seed_pts, time_budget=time_budget, use_soft_max=True, beta=100.0)
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
    
    # Final high-precision polish pass
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining > 5:
            # Polish with higher beta for sharper max
            budget = min(20.0, remaining - 2)
            if budget > 2:
                t_start = time.time()
                pts, rad = _local_search(best_points, time_budget=budget, use_soft_max=True, beta=200.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
    
    if best_points is None:
        best_points = seeds[0]
    
    # Ensure output is normalized
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
