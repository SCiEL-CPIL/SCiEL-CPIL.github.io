"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _normalize_points(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms


def _covering_radius_radians(points):
    """Compute the true covering radius in radians using hull facets."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    simplices = hull.simplices
    if len(simplices) == 0:
        return np.pi
        
    p1 = points[simplices[:, 0]]
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1, keepdims=True)
    
    mask = norm_n[:, 0] > 1e-10
    if not np.all(mask):
        return np.pi
        
    n = n[mask] / norm_n[mask]
    p1m = p1[mask]
    
    dots = np.sum(n * p1m, axis=1)
    n[dots < 0] *= -1
    
    cos_r = np.sum(n * p1m, axis=1)
    cos_r = np.clip(cos_r, -1.0, 1.0)
    radii = np.arccos(cos_r)
    
    return float(np.max(radii))


def _get_facet_data(points):
    """
    Compute all facet normals, circumradii, and identify valid facets.
    Returns: normals (F,3), radii (F,), simplices (F,3), mask
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None, None
    
    simplices = hull.simplices
    if len(simplices) == 0:
        return None, None, None, None
        
    p1 = points[simplices[:, 0]]
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1, keepdims=True)
    
    mask = norm_n[:, 0] > 1e-10
    if not np.all(mask):
        return None, None, None, None
        
    n = n[mask] / norm_n[mask]
    p1m = p1[mask]
    
    dots = np.sum(n * p1m, axis=1)
    n[dots < 0] *= -1
    
    cos_r = np.sum(n * p1m, axis=1)
    cos_r = np.clip(cos_r, -1.0, 1.0)
    radii = np.arccos(cos_r)
    
    valid_indices = np.where(mask)[0]
    
    return n, radii, simplices[valid_indices], mask


def _smooth_facet_surrogate(params, beta=100.0):
    """
    Smooth differentiable surrogate: soft-max over facets of circumradius.
    This directly approximates the covering radius.
    """
    pts = params.reshape(N, 3)
    pts = _normalize_points(pts)
    
    normals, radii, simplices, mask = _get_facet_data(pts)
    
    if normals is None:
        return np.pi
    
    # Soft-max over facet radii
    beta_radii = beta * radii
    max_val = np.max(beta_radii)
    logsumexp = max_val + np.log(np.sum(np.exp(beta_radii - max_val)))
    soft_max = logsumexp / beta
    
    return soft_max


def _smooth_facet_surrogate_grad(params, beta=100.0):
    """
    Compute value and gradient of smooth facet surrogate via finite differences.
    """
    val = _smooth_facet_surrogate(params, beta=beta)
    
    grad = np.zeros_like(params)
    h = 1e-6
    for i in range(len(params)):
        orig = params[i]
        params[i] = orig + h
        val_plus = _smooth_facet_surrogate(params, beta=beta)
        params[i] = orig - h
        val_minus = _smooth_facet_surrogate(params, beta=beta)
        params[i] = orig
        grad[i] = (val_plus - val_minus) / (2 * h)
    
    return val, grad


def _facet_shrink_polish(points, time_limit=15.0):
    """
    Geometric relaxation: move worst facet vertices toward circumcenter.
    """
    start_time = time.time()
    step = 0.05
    prev_radius = _covering_radius_radians(points)
    
    for it in range(10000):
        if time.time() > start_time + time_limit:
            break
            
        current_radius, worst_facet, worst_center = _get_worst_facet(points)
        
        if worst_facet is None:
            break
        
        # Early stopping if no improvement
        if current_radius >= prev_radius - 1e-10 and it > 10:
            step *= 0.9
            if step < 0.001:
                break
        
        v0 = points[worst_facet[0]]
        v1 = points[worst_facet[1]]
        v2 = points[worst_facet[2]]
        
        def move_vertex(v, c, step):
            # Project circumcenter onto tangent plane at v
            grad = c - np.dot(c, v) * v
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                return v
            grad = grad / grad_norm
            v_new = v + step * grad
            return v_new / np.linalg.norm(v_new)
        
        points[worst_facet[0]] = move_vertex(v0, worst_center, step)
        points[worst_facet[1]] = move_vertex(v1, worst_center, step)
        points[worst_facet[2]] = move_vertex(v2, worst_center, step)
        
        new_radius = _covering_radius_radians(points)
        if new_radius < prev_radius:
            prev_radius = new_radius
        else:
            step *= 0.8
            if step < 0.001:
                break
        
        if it % 100 == 0:
            step *= 0.95
            if step < 0.002:
                step = 0.002

    return points


def _get_worst_facet(points):
    """Find the worst facet and its circumcenter."""
    normals, radii, simplices, mask = _get_facet_data(points)
    
    if normals is None:
        return np.pi, None, None
    
    max_idx = np.argmax(radii)
    max_r = radii[max_idx]
    worst_facet = simplices[max_idx]
    worst_center = normals[max_idx]
    
    return max_r, worst_facet, worst_center


def _repulsive_energy(params, s=20.0):
    """
    Riesz s-energy with high exponent to maximize minimum pairwise separation.
    """
    pts = params.reshape(N, 3)
    pts = _normalize_points(pts)
    
    dots = np.dot(pts, pts.T)
    dots = np.clip(dots, -1.0, 1.0)
    
    dist_sq = 2.0 - 2.0 * dots
    
    mask = ~np.eye(N, dtype=bool)
    dist_sq = dist_sq[mask]
    dist_sq = np.maximum(dist_sq, 1e-10)
    
    return np.sum(dist_sq ** (-s / 2.0))


def _repulsive_energy_grad(params, s=20.0):
    """
    Compute value and gradient of Riesz s-energy via finite differences.
    """
    val = _repulsive_energy(params, s=s)
    
    grad = np.zeros_like(params)
    h = 1e-5
    for i in range(len(params)):
        orig = params[i]
        params[i] = orig + h
        e_plus = _repulsive_energy(params, s=s)
        params[i] = orig - h
        e_minus = _repulsive_energy(params, s=s)
        params[i] = orig
        grad[i] = (e_plus - e_minus) / (2 * h)
    
    return val, grad


def _gradient_descent_phase(points, time_limit=20.0, beta=100.0, use_repulsive=False, s=20.0, use_grad=True):
    """
    Perform gradient descent using L-BFGS-B on the smooth facet surrogate or Riesz energy.
    """
    start_time = time.time()
    
    if use_repulsive:
        if use_grad:
            def objective(params):
                if time.time() - start_time > time_limit:
                    return 1e6
                return _repulsive_energy(params, s=s)
            def gradient(params):
                if time.time() - start_time > time_limit:
                    return np.zeros_like(params)
                _, g = _repulsive_energy_grad(params, s=s)
                return g
        else:
            def objective(params):
                if time.time() - start_time > time_limit:
                    return 1e6
                return _repulsive_energy(params, s=s)
            gradient = None
    else:
        if use_grad:
            def objective(params):
                if time.time() - start_time > time_limit:
                    return 1e6
                return _smooth_facet_surrogate(params, beta=beta)
            def gradient(params):
                if time.time() - start_time > time_limit:
                    return np.zeros_like(params)
                _, g = _smooth_facet_surrogate_grad(params, beta=beta)
                return g
        else:
            def objective(params):
                if time.time() - start_time > time_limit:
                    return 1e6
                return _smooth_facet_surrogate(params, beta=beta)
            gradient = None
    
    x0 = points.flatten()
    
    try:
        if gradient is not None:
            result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                              options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 300})
        else:
            result = minimize(objective, x0, method='L-BFGS-B', 
                              options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 300})
        best_points = _normalize_points(result.x.reshape(N, 3))
        return best_points
    except Exception:
        return points


def _multi_restart_optimize(points, deadline, num_restarts=3):
    """
    Try multiple perturbations and keep the best result.
    """
    best_points = points.copy()
    best_radius = _covering_radius_radians(points)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
        
        perturbation_scale = 0.03 * (restart + 1)
        noise = np.random.randn(N, 3) * perturbation_scale
        perturbed = _normalize_points(points + noise)
        
        optimized = _gradient_descent_phase(perturbed, time_limit=15.0, beta=80.0, use_grad=False)
        
        polished = _facet_shrink_polish(optimized, time_limit=8.0)
        
        radius = _covering_radius_radians(polished)
        if radius < best_radius:
            best_radius = radius
            best_points = polished.copy()
    
    return best_points, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius_radians(points)
    
    # Phase 1: Coarse optimization with low beta
    if time.time() < deadline - 70.0:
        points = _gradient_descent_phase(points, time_limit=20.0, beta=30.0, use_grad=False)
        current_radius = _covering_radius_radians(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    # Phase 2: Medium beta
    if time.time() < deadline - 45.0:
        points = _gradient_descent_phase(points, time_limit=18.0, beta=100.0, use_grad=False)
        current_radius = _covering_radius_radians(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    # Phase 3: High beta polishing
    if time.time() < deadline - 25.0:
        points = _gradient_descent_phase(points, time_limit=15.0, beta=300.0, use_grad=False)
        current_radius = _covering_radius_radians(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    # Phase 4: Repulsive energy refinement
    if time.time() < deadline - 15.0:
        repulsive_points = best_points.copy()
        repulsive_points = _gradient_descent_phase(repulsive_points, time_limit=10.0, use_repulsive=True, s=40.0, use_grad=False)
        repulsive_radius = _covering_radius_radians(repulsive_points)
        if repulsive_radius < best_radius:
            best_radius = repulsive_radius
            best_points = repulsive_points.copy()
            if time.time() < deadline - 5.0:
                polished = _facet_shrink_polish(best_points, time_limit=5.0)
                pol_radius = _covering_radius_radians(polished)
                if pol_radius < best_radius:
                    best_radius = pol_radius
                    best_points = polished.copy()
    
    # Phase 5: Multi-restart exploration
    if time.time() < deadline - 25.0:
        best_points, best_radius = _multi_restart_optimize(best_points, deadline, num_restarts=3)
    
    # Phase 6: Final facet shrink polish
    if time.time() < deadline - 5.0:
        polished = _facet_shrink_polish(best_points, time_limit=12.0)
        pol_radius = _covering_radius_radians(polished)
        if pol_radius < best_radius:
            best_radius = pol_radius
            best_points = polished.copy()
    
    # Final high-beta gradient polish if time allows
    if time.time() < deadline - 3.0:
        final_points = _gradient_descent_phase(best_points, time_limit=5.0, beta=500.0, use_grad=False)
        final_radius = _covering_radius_radians(final_points)
        if final_radius < best_radius:
            best_radius = final_radius
            best_points = final_points.copy()

    # Final check
    if np.all(np.isfinite(best_points)) and np.all(np.linalg.norm(best_points, axis=1) > 1e-9):
        return best_points
    else:
        return points
# EVOLVE-BLOCK-END
