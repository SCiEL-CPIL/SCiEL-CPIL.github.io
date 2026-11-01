import numpy as np
import scipy.spatial
import time
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 115.0  # safety margin before 120s kill

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _compute_facet_data(points):
    """
    Compute convex hull and facet circumcenters/radii (vectorized).
    Returns: facets, centers (array), radii (array), worst_idx
    """
    points = _normalize(points)
    
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        return None, None, None, None
    
    facets = hull.simplices
    n_facets = len(facets)
    if n_facets == 0:
        return facets, np.zeros((0,3)), np.zeros(0), None
        
    a = points[facets[:, 0]]  # (F, 3)
    b = points[facets[:, 1]]
    c = points[facets[:, 2]]
    
    # Vectorized circumcenter computation
    v1 = a - b
    v2 = b - c
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1)
    
    # Handle degenerate facets (collinear points)
    valid = norm_n > 1e-12
    
    centers = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    if np.any(valid):
        cc = n[valid] / norm_n[valid, np.newaxis]
        
        # Ensure circumcenter faces outward (toward 'a')
        dots = np.sum(cc * a[valid], axis=1)
        cc[dots < 0] *= -1
        
        centers[valid] = cc
        
        # Calculate radii (angular distance from center to vertex a)
        cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
        radii[valid] = np.arccos(cos_r)
    
    if n_facets > 0 and np.any(radii > 0):
        worst_idx = np.argmax(radii)
    else:
        worst_idx = None
        
    return facets, centers, radii, worst_idx

def _compute_covering_radius(points):
    """
    Compute the actual covering radius (max facet circumradius).
    """
    facets, centers, radii, worst_idx = _compute_facet_data(points)
    if facets is None or worst_idx is None:
        return np.pi
    return radii[worst_idx]

def _soft_max_radius_and_grad(points, beta=50.0):
    """
    Compute soft-max radius and its gradient w.r.t. points.
    Uses log-sum-exp for numerical stability.
    """
    facets, centers, radii, worst_idx = _compute_facet_data(points)
    if facets is None or worst_idx is None:
        return np.pi, np.zeros_like(points)
    
    max_r = np.max(radii)
    # Use log-sum-exp trick for numerical stability
    shifted = beta * (radii - max_r)
    max_shifted = np.max(shifted)
    exp_terms = np.exp(shifted - max_shifted)
    log_sum = max_shifted + np.log(np.sum(exp_terms))
    soft_radius = max_r + log_sum / beta
    weights = exp_terms / np.sum(exp_terms)
    
    grads = np.zeros_like(points)
    n_facets = len(facets)
    
    for i in range(n_facets):
        w = weights[i]
        if w < 1e-15:
            continue
        r = radii[i]
        if r < 1e-12:
            continue
        
        C = centers[i]
        sin_r = np.sin(r)
        if sin_r < 1e-10:
            continue
        
        simplex = facets[i]
        p_indices = simplex
        p_pts = points[p_indices]
        
        # Gradient of arccos(dot(C, p)) w.r.t. p is -C/sin(r)
        grad_contrib = -C / sin_r
        
        # Project to tangent space at each point in the facet
        dots = np.sum(p_pts * grad_contrib, axis=1)
        radial_comps = dots[:, np.newaxis] * p_pts
        tangential_grads = grad_contrib - radial_comps
        
        for k in range(3):
            grads[p_indices[k]] += w * tangential_grads[k]
    
    return soft_radius, grads

def _repulsion_grad(points, weight=0.05):
    """
    Compute gradient of pairwise repulsion energy.
    Energy = weight * sum_{i<j} 1/d(i,j)^2
    """
    n = len(points)
    grads = np.zeros_like(points)
    
    for i in range(n):
        for j in range(i+1, n):
            diff = points[i] - points[j]
            d2 = np.dot(diff, diff)
            if d2 < 1e-15:
                d2 = 1e-15
            # Gradient of 1/d^2 = 1/(p_i-p_j)^2
            # dE/dp_i = -2/d^4 * (p_i - p_j)
            grad_factor = -2.0 * weight / (d2 ** 2)
            grads[i] += grad_factor * diff
            grads[j] -= grad_factor * diff
    
    return grads

def _combined_objective_and_grad(x, beta=50.0, rep_weight=0.05):
    """
    Combined objective: soft-max radius + repulsion penalty.
    Returns (value, gradient).
    """
    points = _normalize(x.reshape(N, 3))
    soft_r, r_grad = _soft_max_radius_and_grad(points, beta=beta)
    rep_grad = _repulsion_grad(points, weight=rep_weight)
    
    total_grad = r_grad + rep_grad
    return soft_r, total_grad.reshape(-1)

def _lbfgs_optimize(points, time_budget, beta_schedule, rep_weight=0.05):
    """
    Use L-BFGS-B to minimize the combined objective with beta schedule.
    """
    start_time = time.time()
    current_pts = points.copy()
    current_radius = _compute_covering_radius(current_pts)
    best_pts = current_pts.copy()
    best_radius = current_radius
    
    for beta in beta_schedule:
        if time.time() - start_time > time_budget:
            break
            
        def objective_and_grad(x, b=beta):
            if time.time() - start_time > time_budget:
                raise Exception("timeout")
            return _combined_objective_and_grad(x, beta=b, rep_weight=rep_weight)
        
        try:
            x0 = current_pts.reshape(-1)
            res = minimize(objective_and_grad, x0, method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200})
            
            if res.fun < 1e10:  # valid result
                new_pts = _normalize(res.x.reshape(N, 3))
                new_radius = _compute_covering_radius(new_pts)
                
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_pts = new_pts.copy()
                
                current_pts = new_pts
        except Exception:
            pass
    
    return best_pts, best_radius

def _gradient_descent_polish(points, time_budget, beta=100.0, rep_weight=0.03):
    """
    Fine polish with gradient descent using combined objective.
    """
    start_time = time.time()
    current_pts = points.copy()
    current_radius = _compute_covering_radius(current_pts)
    best_pts = current_pts.copy()
    best_radius = current_radius
    
    eta = 0.005
    
    while time.time() - start_time < time_budget:
        val, grad = _combined_objective_and_grad(current_pts.reshape(-1), beta=beta, rep_weight=rep_weight)
        grad = grad.reshape(N, 3)
        
        # Project gradient to tangent space
        grad = grad - np.sum(grad * current_pts, axis=1, keepdims=True) * current_pts
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        
        direction = grad / grad_norm
        new_pts = _normalize(current_pts - eta * direction)
        
        new_radius = _compute_covering_radius(new_pts)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
        
        if new_radius < current_radius - 1e-14:
            current_pts = new_pts
            eta = min(eta * 1.05, 0.02)
        else:
            eta *= 0.9
            if eta < 1e-8:
                break
    
    return best_pts, best_radius

def _optimize_from_start(pts_init, time_budget):
    """
    Run the full optimization pipeline from a starting configuration.
    """
    pts = pts_init.copy()
    best_pts_local = pts.copy()
    best_radius_local = _compute_covering_radius(pts)
    
    # Phase 1: L-BFGS with increasing beta
    if time.time() < time_budget - 5.0:
        beta_schedule = [20.0, 40.0, 80.0, 150.0]
        pts_res, rad_res = _lbfgs_optimize(pts, time_budget - 10.0, beta_schedule, rep_weight=0.05)
        if rad_res < best_radius_local:
            best_radius_local = rad_res
            best_pts_local = pts_res.copy()
    
    # Phase 2: Gradient descent polish
    if time.time() < time_budget - 2.0:
        remaining = time_budget - 2.0 - (time.time() - (time_budget - 10.0))
        pts_res, rad_res = _gradient_descent_polish(best_pts_local, max(remaining, 1.0), beta=150.0, rep_weight=0.03)
        if rad_res < best_radius_local:
            best_radius_local = rad_res
            best_pts_local = pts_res.copy()
    
    # Phase 3: Final L-BFGS polish with high beta
    if time.time() < time_budget - 1.0:
        beta_schedule = [200.0, 300.0]
        pts_res, rad_res = _lbfgs_optimize(best_pts_local, time_budget - 1.0 - time.time(), beta_schedule, rep_weight=0.02)
        if rad_res < best_radius_local:
            best_radius_local = rad_res
            best_pts_local = pts_res.copy()
    
    return best_pts_local, best_radius_local

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Generate multiple initial configurations
    # Config 1: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Config 2: Fibonacci with offset
    i2 = np.arange(N) + 0.5
    z2 = 1.0 - 2.0 * i2 / N
    phi2 = np.pi * (1.0 + np.sqrt(5.0)) * (i2 + 0.37)
    r2_ = np.sqrt(1.0 - z2 * z2)
    fib2_pts = np.column_stack([r2_ * np.cos(phi2), r2_ * np.sin(phi2), z2])
    
    # Config 3: Random with good spread (use golden ratio spiral with different phase)
    i3 = np.arange(N) + 0.5
    z3 = 1.0 - 2.0 * i3 / N
    phi3 = np.pi * (1.0 + np.sqrt(5.0)) * (i3 + 0.71)
    r3 = np.sqrt(1.0 - z3 * z3)
    fib3_pts = np.column_stack([r3 * np.cos(phi3), r3 * np.sin(phi3), z3])
    
    # Config 4: Random
    rng = np.random.default_rng(42)
    random_pts = rng.normal(size=(N, 3))
    random_pts = _normalize(random_pts)
    
    initial_configs = [fib_pts, fib2_pts, fib3_pts, random_pts]
    
    best_pts = fib_pts.copy()
    best_radius = float('inf')
    
    # Allocate time budget across restarts
    # First restart gets most time, subsequent ones get less
    n_configs = len(initial_configs)
    
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 2.0:
            break
        
        # Give each restart a portion of remaining time
        if idx == 0:
            tb = deadline - 25.0  # First restart: most time
        elif idx == 1:
            tb = deadline - 15.0  # Second restart
        elif idx == 2:
            tb = deadline - 8.0   # Third restart
        else:
            tb = deadline - 2.0   # Final restart
        
        if time.time() > tb - 1.0:
            continue
        
        pts_res, rad_res = _optimize_from_start(init_pts, tb)
        
        if rad_res < best_radius:
            best_radius = rad_res
            best_pts = pts_res.copy()
    
    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Final L-BFGS polish on the best result
    if time.time() < deadline - 1.0:
        beta_schedule = [200.0, 300.0, 400.0]
        pts_res, rad_res = _lbfgs_optimize(best_pts, deadline - 1.0 - time.time(), beta_schedule, rep_weight=0.02)
        if rad_res < best_radius:
            best_radius = rad_res
            best_pts = pts_res.copy()
    
    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return best_pts
