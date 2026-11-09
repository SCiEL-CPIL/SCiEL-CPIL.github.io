"""Direct geometric approach: optimize 130 points on unit sphere to minimize covering radius."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _facet_circumcenters(points):
    """Compute circumcenter and cos value for each facet. Returns (facets, circumcenters, cos_values)."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return [], [], []
    
    facets = hull.simplices
    circumcenters = []
    cos_values = []
    
    for facet in facets:
        a = points[facet[0]]
        b = points[facet[1]]
        c = points[facet[2]]
        
        v1 = a - b
        v2 = b - c
        p = np.cross(v1, v2)
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-12:
            circumcenters.append(np.zeros(3))
            cos_values.append(1.0)
            continue
        p = p / p_norm
        
        if p @ a < 0:
            p = -p
        
        cos_val = p @ a
        circumcenters.append(p.copy())
        cos_values.append(cos_val)
    
    return facets, np.array(circumcenters), np.array(cos_values)


def _covering_radius(points):
    """Compute covering radius in radians via convex hull facets."""
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 3:
        return np.pi
    
    facets, circumcenters, cos_values = _facet_circumcenters(points)
    if len(facets) == 0:
        return np.pi
    
    min_cos = np.min(cos_values)
    return np.arccos(np.clip(min_cos, -1.0, 1.0))


def _soft_covering_radius_and_grad(points, beta=100.0):
    """
    Compute soft covering radius and gradient using log-sum-exp over facet circumradii.
    Uses arccos(cos_val) for the actual angular distance to get a more accurate gradient.
    Returns (soft_radius, gradient of soft_radius w.r.t. points).
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 3:
        return np.pi, np.zeros_like(points)
    
    facets, circumcenters, cos_values = _facet_circumcenters(points)
    if len(facets) == 0:
        return np.pi, np.zeros_like(points)
    
    # Angular distance for each facet: theta_j = arccos(cos_j)
    cos_clipped = np.clip(cos_values, -1.0, 1.0)
    angles = np.arccos(cos_clipped)
    
    # Soft max of angles via log-sum-exp
    max_angle = np.max(angles)
    exps = np.exp(beta * (angles - max_angle))
    weights = exps / np.sum(exps)
    
    soft_radius = max_angle + np.log(np.sum(exps)) / beta
    
    # Gradient: d(soft_radius)/d(points[i])
    # d(soft_radius)/d(angle_j) = weights[j]
    # d(angle_j)/d(cos_j) = -1/sqrt(1 - cos_j^2)  (for cos_j in (-1,1))
    # d(cos_j)/d(points[idx]) ≈ p_j for idx in facet j (fixed normal approximation)
    # So d(soft_radius)/d(points[idx]) = sum_j weights[j] * (-1/sqrt(1-cos_j^2)) * p_j (for idx in facet j)
    
    # Compute d(angle)/d(cos) = -1/sqrt(1-cos^2), clip to avoid division by zero
    sin_angles = np.sqrt(np.maximum(1e-30, 1.0 - cos_clipped * cos_clipped))
    d_angle_d_cos = -1.0 / sin_angles
    
    grad = np.zeros_like(points)
    for j, facet in enumerate(facets):
        w = weights[j] * d_angle_d_cos[j]
        if abs(w) < 1e-15:
            continue
        p_j = circumcenters[j]
        for idx in facet:
            # d(soft_radius)/d(points[idx]) += w * p_j (approx, fixed normal)
            # Project to tangent space at points[idx]
            pt = points[idx]
            tangent = p_j - np.dot(p_j, pt) * pt
            grad[idx] += w * tangent
    
    return soft_radius, grad


def _gradient_descent(points, deadline, max_iters=3000, base_step=0.05):
    """Optimize points using gradient descent toward circumcenters of worst facets."""
    points = points.copy()
    best_radius = _covering_radius(points)
    step = base_step
    patience = 0
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        radius = _covering_radius(points)
        if radius < best_radius - 1e-10:
            best_radius = radius
            patience = 0
            step = min(step * 1.02, 0.10)
        else:
            patience += 1
            step *= 0.92
            if patience > 80:
                break
        
        if step < 1e-13:
            break
        
        worst = _worst_facets(points, k=8)
        if not worst:
            break
        
        for facet, circ in worst:
            for idx in facet:
                pt = points[idx]
                direction = circ - pt
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir
                    tangent = direction - np.dot(direction, pt) * pt
                    norm_t = np.linalg.norm(tangent)
                    if norm_t > 1e-10:
                        tangent = tangent / norm_t
                        new_pt = pt + step * tangent
                        new_pt = new_pt / np.linalg.norm(new_pt)
                        points[idx] = new_pt
    
    return points


def _worst_facets(points, k=5):
    """Return the k worst facets (by circumradius) as list of (indices, circumcenter)."""
    points = np.asarray(points, dtype=np.float64)
    facets, circumcenters, cos_values = _facet_circumcenters(points)
    if len(facets) == 0:
        return []
    
    order = np.argsort(cos_values)
    return [(facets[i], circumcenters[i]) for i in order[:k]]


def _lbfgs_smooth_optimize(points, deadline, max_iter=500):
    """Optimize using L-BFGS-B on the soft covering radius surrogate."""
    points_flat = points.flatten().copy()
    beta = 80.0
    
    def fun(flat_pts):
        pts = flat_pts.reshape(N, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
        
        soft_r, _ = _soft_covering_radius_and_grad(pts, beta=beta)
        return soft_r

    def jac(flat_pts):
        pts = flat_pts.reshape(N, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
        
        _, grad = _soft_covering_radius_and_grad(pts, beta=beta)
        
        # Project gradient to tangent space
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        return grad_tangent.flatten()
    
    result = minimize(fun, points_flat, jac=jac, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    
    optimized = result.x.reshape(N, 3)
    norms = np.linalg.norm(optimized, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    optimized = optimized / norms
    
    return optimized


def _polish(points, deadline):
    """Final high-precision polish with multiple phases."""
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Phase 1: Gradient descent with small steps
    if time.time() < deadline - 12.0:
        polished = _gradient_descent(points, deadline - 12.0, max_iters=600, base_step=0.008)
        r = _covering_radius(polished)
        if r < best_radius:
            best_points = polished
            best_radius = r
    
    # Phase 2: L-BFGS on smooth surrogate
    if time.time() < deadline - 8.0:
        polished = _lbfgs_smooth_optimize(best_points, deadline - 8.0, max_iter=300)
        r = _covering_radius(polished)
        if r < best_radius:
            best_points = polished
            best_radius = r
    
    # Phase 3: Very fine gradient descent
    if time.time() < deadline - 4.0:
        polished = _gradient_descent(best_points, deadline - 4.0, max_iters=400, base_step=0.003)
        r = _covering_radius(polished)
        if r < best_radius:
            best_points = polished
            best_radius = r
    
    # Phase 4: One more L-BFGS pass if time allows
    if time.time() < deadline - 2.0:
        polished = _lbfgs_smooth_optimize(best_points, deadline - 2.0, max_iter=100)
        r = _covering_radius(polished)
        if r < best_radius:
            best_points = polished
            best_radius = r
    
    return best_points


def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.pi
    
    # Start with Fibonacci spiral as baseline
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    fib_points = fib_points / np.linalg.norm(fib_points, axis=1, keepdims=True)
    
    # Initial optimization with gradient descent
    best_points = _gradient_descent(fib_points, deadline, max_iters=1200, base_step=0.04)
    best_radius = _covering_radius(best_points)
    
    # Multiple restarts with varied strategies
    num_restarts = 7
    for restart in range(num_restarts):
        if time.time() > deadline - 50.0:
            break
        
        remaining = deadline - time.time()
        if remaining < 25.0:
            break
        
        # Vary perturbation size based on restart index
        if best_points is not None and restart % 3 != 2:
            pert_size = 0.08 + 0.03 * (restart % 3)
            perturbation = pert_size * rng.standard_normal((N, 3))
            points = best_points + perturbation
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            # One fully random restart
            points = rng.standard_normal((N, 3))
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        else:
            # Moderate perturbation
            perturbation = 0.15 * rng.standard_normal((N, 3))
            points = best_points + perturbation
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        max_iters = min(1200, int(remaining * 12))
        optimized = _gradient_descent(points, deadline, max_iters=max_iters, base_step=0.05)
        
        radius = _covering_radius(optimized)
        if radius < best_radius:
            best_radius = radius
            best_points = optimized.copy()
    
    # Final polish
    if best_points is not None and time.time() < deadline - 15.0:
        best_points = _polish(best_points, deadline)
        best_radius = _covering_radius(best_points)
    
    # Ensure valid output
    if best_points is None:
        best_points = fib_points
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_points = best_points / norms
    
    return best_points
