# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

# Deadline guard: must finish within 630s, leave safety margin
_DEADLINE = 630.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - 10.0 - (time.time() - _START_TIME)

def _metric(points):
    """Compute (d_min / d_max)^2. Return 0 if invalid."""
    try:
        if not np.all(np.isfinite(points)):
            return 0.0
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0 or d_max < 1e-15:
            return 0.0
        if d_min < 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio
    except Exception:
        return 0.0

def _neg_metric_flat(x):
    """Objective for optimizers: minimize -metric."""
    points = x.reshape(-1, 3)
    return -_metric(points)

def _fibonacci_sphere(n):
    """Generate n points on a sphere using Fibonacci lattice."""
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # Golden angle
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def _icosahedron_plus_poles(n):
    """Generate icosahedron vertices + poles, then add more points to reach n."""
    phi_g = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi_g], [0, -1, phi_g], [0, 1, -phi_g], [0, -1, -phi_g],
        [1, phi_g, 0], [-1, phi_g, 0], [1, -phi_g, 0], [-1, -phi_g, 0],
        [phi_g, 0, 1], [phi_g, 0, -1], [-phi_g, 0, 1], [-phi_g, 0, -1]
    ])
    norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico = ico / norms
    extra = np.array([[0, 1, 0], [0, -1, 0]])
    pts = np.vstack([ico, extra])
    if len(pts) < n:
        # Add more points via Fibonacci
        fib = _fibonacci_sphere(n - len(pts))
        pts = np.vstack([pts, fib])
    return pts[:n]

def _soft_objective(x, beta=50.0):
    """
    Soft-min objective using log-sum-exp for smooth optimization.
    We want to maximize (d_min/d_max)^2 = d_min^2 / d_max^2
    Minimize -log(sum(exp(-beta * dists_sq/d_max_sq))) / beta
    """
    p = x.reshape(-1, 3)
    if not np.all(np.isfinite(p)):
        return 1.0
    dists_sq = pdist(p) ** 2
    d_max_sq = np.max(dists_sq)
    if d_max_sq < 1e-12:
        return 1.0
    normalized = dists_sq / d_max_sq
    min_norm = np.min(normalized)
    # Avoid overflow in exp
    log_sum = -beta * min_norm + np.log(np.sum(np.exp(-beta * (normalized - min_norm))))
    soft_min_norm = log_sum / beta
    return -soft_min_norm

def _refine_powell(points, max_iter=500):
    """Powell refinement on the exact objective."""
    try:
        res = minimize(
            _neg_metric_flat,
            points.flatten(),
            method='Powell',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        refined = res.x.reshape(-1, 3)
        if _metric(refined) > _metric(points):
            return refined
    except Exception:
        pass
    return points

def _refine_lbfgs_soft(points, beta=50.0, max_iter=500):
    """L-BFGS-B refinement on the soft objective."""
    try:
        res = minimize(
            lambda x: _soft_objective(x, beta=beta),
            points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        refined = res.x.reshape(-1, 3)
        if _metric(refined) > _metric(points):
            return refined
    except Exception:
        pass
    return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    
    Returns
        points: np.ndarray of shape (14,3)
    """
    n = 14
    d = 3
    rng = np.random.default_rng(42)
    
    # Initial candidates
    init_fib = _fibonacci_sphere(n)
    init_ico = _icosahedron_plus_poles(n)
    
    best_points = init_fib.copy()
    best_score = _metric(init_fib)
    
    if _metric(init_ico) > best_score:
        best_points = init_ico.copy()
        best_score = _metric(init_ico)
    
    # Phase 1: Global search with soft objective and multiple starts
    soft_restarts = 0
    max_soft_restarts = 40
    
    while _time_left() > 60 and soft_restarts < max_soft_restarts:
        if soft_restarts == 0:
            x0 = best_points.ravel()
        elif soft_restarts == 1:
            x0 = init_ico.ravel()
        elif soft_restarts == 2:
            x0 = init_fib.ravel()
        else:
            # Random restarts with perturbation around best
            x0 = best_points.ravel() + rng.normal(0, 0.3, size=(n, d)).ravel()
        
        try:
            res = minimize(
                lambda x: _soft_objective(x, beta=50.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            soft_restarts += 1
            
            candidate = res.x.reshape(n, d)
            cand_score = _metric(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate
        except Exception:
            pass
    
    # Phase 2: Local refinement with exact objective using Powell
    local_restarts = 0
    max_local_restarts = 50
    
    while _time_left() > 10 and local_restarts < max_local_restarts:
        if local_restarts == 0:
            x0 = best_points.ravel()
        else:
            # Small perturbations around best point
            x0 = best_points.ravel() + rng.normal(0, 0.05, size=(n, d)).ravel()
        
        try:
            refined = _refine_powell(x0.reshape(n, d), max_iter=300)
            local_restarts += 1
            
            cand_score = _metric(refined)
            if cand_score > best_score:
                best_score = cand_score
                best_points = refined
        except Exception:
            pass
    
    # Phase 3: Final polish with tighter Powell
    if _time_left() > 5:
        try:
            refined = _refine_powell(best_points, max_iter=500)
            if _metric(refined) > best_score:
                best_points = refined
                best_score = _metric(refined)
        except Exception:
            pass
    
    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        best_points = init_fib
    
    return best_points

# EVOLVE-BLOCK-END
