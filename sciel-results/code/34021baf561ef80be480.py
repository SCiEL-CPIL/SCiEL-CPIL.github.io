import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    Returns radius in degrees.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_radius = 0.0
        
        for tri in facets:
            p1 = points[tri[0]]
            p2 = points[tri[1]]
            p3 = points[tri[2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                continue
                
            c = cross / norm_cross
            
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            dist = np.arccos(dot_val)
            
            if dist > np.pi / 2:
                dist = np.pi - dist
                
            if dist > max_radius:
                max_radius = dist
                
    except Exception:
        return 180.0
        
    return np.degrees(max_radius)

def _get_facet_circumradii(points):
    """
    Compute circumradii (in radians) for all facets of the convex hull.
    Returns array of circumradii and the facet indices.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
    except Exception:
        return np.array([np.pi]), np.array([], dtype=int)
    
    radii = []
    for tri in facets:
        p1 = points[tri[0]]
        p2 = points[tri[1]]
        p3 = points[tri[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            radii.append(np.pi)
            continue
            
        c = cross / norm_cross
        dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        
        if dist > np.pi / 2:
            dist = np.pi - dist
            
        radii.append(dist)
        
    return np.array(radii), facets

def _soft_max_objective(x, beta=50.0):
    """
    Smooth approximation of max(circumradii) using log-sum-exp.
    x is a flat array of 3*N point coordinates.
    """
    points = x.reshape(-1, 3)
    # Normalize points
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    points = points / norms
    
    radii, _ = _get_facet_circumradii(points)
    if len(radii) == 0:
        return np.pi, np.zeros_like(x)
    
    # Soft max: (1/beta) * log(sum(exp(beta * radii)))
    max_r = np.max(radii)
    shifted = beta * (radii - max_r)
    log_sum = max_r + np.log(np.sum(np.exp(shifted))) / beta
    
    # Compute gradient via finite differences on the soft max
    # Use analytic gradient: grad = (1/beta) * sum(exp(beta*(r_i - max_r)) * grad_r_i) / sum(exp(beta*(r_i - max_r)))
    # For now, use numerical gradient with central differences
    grad = np.zeros_like(x)
    h = 1e-5
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += h
        f_plus = _soft_max_objective_no_grad(x_plus, beta)[0]
        
        x_minus = x.copy()
        x_minus[i] -= h
        f_minus = _soft_max_objective_no_grad(x_minus, beta)[0]
        
        grad[i] = (f_plus - f_minus) / (2 * h)
    
    return log_sum, grad

def _soft_max_objective_no_grad(x, beta=50.0):
    """
    Compute soft max objective without gradient (for finite difference use).
    """
    points = x.reshape(-1, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    points = points / norms
    
    radii, _ = _get_facet_circumradii(points)
    if len(radii) == 0:
        return np.pi, np.zeros_like(x)
    
    max_r = np.max(radii)
    shifted = beta * (radii - max_r)
    log_sum = max_r + np.log(np.sum(np.exp(shifted))) / beta
    
    return log_sum, np.zeros_like(x)

def _refine_with_l_bfgs(points, max_iter=200, time_limit=30.0):
    """
    Refine points using L-BFGS-B on the soft-max objective.
    """
    start_time = time.time()
    x0 = points.flatten().copy()
    
    beta = 50.0
    
    def objective_and_grad(x):
        if time.time() - start_time > time_limit:
            raise Exception("Time limit exceeded")
        val, grad = _soft_max_objective(x, beta)
        return val, grad
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxfun': 5000}
        )
        
        if result.success or result.status == 0:
            new_points = result.x.reshape(-1, 3)
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            new_points = new_points / np.where(norms < 1e-9, 1.0, norms)
            return new_points
    except Exception:
        pass
    
    return points

def _simulated_annealing(points, rng, time_limit=30.0):
    """
    Simulated annealing to escape local minima.
    """
    start_time = time.time()
    current_points = points.copy()
    current_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    best_radius = current_radius
    
    T = 0.5
    T_min = 0.01
    cooling = 0.999
    noise_scale = 0.03
    
    while time.time() - start_time < time_limit:
        idx = rng.integers(0, N)
        noise = rng.normal(0, noise_scale, 3)
        new_p = current_points[idx] + noise
        norm_new_p = np.linalg.norm(new_p)
        if norm_new_p < 1e-9:
            continue
        new_p /= norm_new_p
        
        old_p = current_points[idx].copy()
        current_points[idx] = new_p
        
        new_radius = _compute_covering_radius(current_points)
        delta_R = new_radius - current_radius
        
        if delta_R < 0:
            accept = True
        else:
            r = rng.random()
            if r < 1e-15:
                r = 1e-15
            accept = np.log(r) < -delta_R / T
        
        if accept:
            current_radius = new_radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = current_points.copy()
        else:
            current_points[idx] = old_p
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_points

def construct():
    rng = np.random.default_rng(42)
    
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # Generate multiple initial configurations
    initializations = []
    
    # Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    fib_points /= np.linalg.norm(fib_points, axis=1, keepdims=True)
    initializations.append(fib_points)
    
    # Random configurations
    for _ in range(3):
        rand_points = rng.normal(0, 1, (N, 3))
        rand_points /= np.linalg.norm(rand_points, axis=1, keepdims=True)
        initializations.append(rand_points)
    
    best_points = initializations[0].copy()
    best_radius = _compute_covering_radius(best_points)
    
    # For each initialization, run: anneal → L-BFGS refine → verify
    for init_idx, init_points in enumerate(initializations):
        if time.time() > deadline:
            break
        
        # Phase 1: Simulated annealing to find a good local minimum
        anneal_time = min(20.0, (deadline - time.time()) * 0.4)
        if anneal_time < 5.0:
            break
        
        annealed_points = _simulated_annealing(init_points, rng, time_limit=anneal_time)
        anneal_radius = _compute_covering_radius(annealed_points)
        
        # Phase 2: L-BFGS refinement
        refine_time = min(25.0, (deadline - time.time()) * 0.5)
        if refine_time < 10.0:
            break
        
        refined_points = _refine_with_l_bfgs(annealed_points, max_iter=100, time_limit=refine_time)
        refine_radius = _compute_covering_radius(refined_points)
        
        # Use the better of annealed and refined
        if refine_radius < anneal_radius:
            candidate_points = refined_points
            candidate_radius = refine_radius
        else:
            candidate_points = annealed_points
            candidate_radius = anneal_radius
        
        if candidate_radius < best_radius:
            best_radius = candidate_radius
            best_points = candidate_points.copy()
    
    return best_points
