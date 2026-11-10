"""Gradient-based optimization using log-sum-exp surrogate for sphere covering."""
N = 50
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _cover_radius(pts):
    """Compute covering radius (in radians) as max circumradius over hull facets."""
    if len(pts) < 3:
        return np.pi
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi

    max_angle = 0.0
    for simplex in hull.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        cross_bc = np.cross(b, c)
        cross_ca = np.cross(c, a)
        cross_ab = np.cross(a, b)
        w_a = np.linalg.norm(cross_bc)
        w_b = np.linalg.norm(cross_ca)
        w_c = np.linalg.norm(cross_ab)
        total = w_a + w_b + w_c
        if total < 1e-15:
            continue
        circum = a * w_a + b * w_b + c * w_c
        cnorm = np.linalg.norm(circum)
        if cnorm < 1e-15:
            continue
        circum /= cnorm
        facet_normal = np.cross(b - a, c - a)
        fnorm = np.linalg.norm(facet_normal)
        if fnorm > 1e-15:
            facet_normal /= fnorm
            if np.dot(circum, facet_normal) < 0:
                circum = -circum
        dots = np.array([np.dot(circum, a), np.dot(circum, b), np.dot(circum, c)])
        min_dot = np.clip(min(dots), -1.0, 1.0)
        angle = np.arccos(min_dot)
        if angle > max_angle:
            max_angle = angle

    return max_angle


def _logsumexp_surrogate(pts, beta=50.0):
    """
    Smooth surrogate for covering radius using log-sum-exp over facet circumradii.
    Returns (value, gradient) in flat 3N-vector space.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, np.zeros(3 * N)

    # Compute circumradii and circumcenters for each facet
    n_facets = len(hull.simplices)
    if n_facets == 0:
        return np.pi, np.zeros(3 * N)

    angles = np.zeros(n_facets)
    circumcenters = np.zeros((n_facets, 3))

    for i, simplex in enumerate(hull.simplices):
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        cross_bc = np.cross(b, c)
        cross_ca = np.cross(c, a)
        cross_ab = np.cross(a, b)
        w_a = np.linalg.norm(cross_bc)
        w_b = np.linalg.norm(cross_ca)
        w_c = np.linalg.norm(cross_ab)
        total = w_a + w_b + w_c
        if total < 1e-15:
            angles[i] = 0.0
            continue
        circum = a * w_a + b * w_b + c * w_c
        cnorm = np.linalg.norm(circum)
        if cnorm < 1e-15:
            angles[i] = 0.0
            continue
        circum /= cnorm
        facet_normal = np.cross(b - a, c - a)
        fnorm = np.linalg.norm(facet_normal)
        if fnorm > 1e-15:
            facet_normal /= fnorm
            if np.dot(circum, facet_normal) < 0:
                circum = -circum
        dots = np.array([np.dot(circum, a), np.dot(circum, b), np.dot(circum, c)])
        min_dot = np.clip(min(dots), -1.0, 1.0)
        angle = np.arccos(min_dot)
        angles[i] = angle
        circumcenters[i] = circum

    # Log-sum-exp surrogate: (1/beta) * log(sum(exp(beta * angles)))
    max_angle = np.max(angles)
    if max_angle < 1e-15:
        return 0.0, np.zeros(3 * N)
    
    shifted = beta * (angles - max_angle)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -500, 500)
    log_sum = max_angle + (1.0 / beta) * np.log(np.sum(np.exp(shifted)))
    
    # Weights for gradient computation
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    # Gradient with respect to each point position
    # d(logsumexp)/d(point_k) = sum_i weights[i] * d(angles[i])/d(point_k)
    # d(angles[i])/d(point_k) involves the circumcenter and the point's position
    # For point k in facet i: angle = arccos(min_dot), min_dot = dot(circum, point_k)
    # d(angle)/d(point_k) = -circum / sqrt(1 - dot^2) for the min_dot component
    
    grad = np.zeros((N, 3))
    
    for i in range(n_facets):
        if weights[i] < 1e-12:
            continue
        simplex = hull.simplices[i]
        circum = circumcenters[i]
        angle_i = angles[i]
        
        for k in [simplex[0], simplex[1], simplex[2]]:
            p_k = pts[k]
            dot_val = np.dot(circum, p_k)
            dot_val = np.clip(dot_val, -1.0 + 1e-10, 1.0 - 1e-10)
            # d(arccos(x))/dx = -1/sqrt(1-x^2)
            # But we need d(angle)/d(point_k) where angle = arccos(dot(circum, p_k))
            # = -circum / sqrt(1 - dot^2)
            denom = np.sqrt(1.0 - dot_val * dot_val)
            if denom > 1e-10:
                # Gradient of angle w.r.t. p_k
                d_angle_dp = -circum / denom
                # Project to tangent plane (since p_k is constrained to sphere)
                d_angle_dp = d_angle_dp - np.dot(d_angle_dp, p_k) * p_k
                grad[k] += weights[i] * d_angle_dp
    
    # Scale by 1/beta for the log-sum-exp
    grad *= (1.0 / beta)
    
    return log_sum, grad.flatten()


def _optimize_lgbfgs(pts, deadline=None):
    """Optimize using L-BFGS-B with log-sum-exp surrogate."""
    x0 = pts.flatten()
    
    def objective_and_grad(x):
        if deadline is not None and time.time() > deadline:
            return np.pi, np.zeros_like(x)
        pts_reshaped = x.reshape(N, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(pts_reshaped, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts_reshaped = pts_reshaped / norms
        val, grad = _logsumexp_surrogate(pts_reshaped, beta=50.0)
        return val, grad
    
    result = minimize(
        objective_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500}
    )
    
    opt_pts = result.x.reshape(N, 3)
    norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    opt_pts = opt_pts / norms
    
    return opt_pts


def _fibonacci_spiral(n=N):
    """Generate Fibonacci spiral points on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _local_refine(pts, deadline=None):
    """Local refinement using small random perturbations and acceptance."""
    pts = pts.copy()
    best_radius = _cover_radius(pts)
    best_pts = pts.copy()
    rng = np.random.RandomState(999)
    step = 0.01
    
    for it in range(200):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            random_dir = rng.randn(3)
            tangent = random_dir - np.dot(random_dir, pts[i]) * pts[i]
            tnorm = np.linalg.norm(tangent)
            if tnorm < 1e-10:
                continue
            tangent /= tnorm
            
            new_pts = pts.copy()
            new_pts[i] = pts[i] + step * tangent
            new_pts[i] /= np.linalg.norm(new_pts[i])
            
            new_radius = _cover_radius(new_pts)
            if new_radius < best_radius - 1e-12:
                pts = new_pts
                best_radius = new_radius
                best_pts = pts.copy()
                improved = True
        
        if not improved:
            step *= 0.85
            if step < 1e-7:
                break
    
    return best_pts


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    rng = np.random.RandomState(42)

    best_pts = None
    best_radius = np.pi

    # Strategy 1: Fibonacci spiral + L-BFGS-B optimization
    fib_pts = _fibonacci_spiral(N)
    if time.time() < deadline - 20:
        opt_pts = _optimize_lgbfgs(fib_pts, deadline=deadline - 15)
        opt_radius = _cover_radius(opt_pts)
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_pts = opt_pts.copy()

    # Strategy 2: Multiple random restarts with L-BFGS-B
    for restart in range(8):
        if time.time() > deadline - 15:
            break
        
        random_pts = rng.randn(N, 3)
        random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
        
        opt_pts = _optimize_lgbfgs(random_pts, deadline=deadline - 10)
        opt_radius = _cover_radius(opt_pts)
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_pts = opt_pts.copy()

    # Strategy 3: Local refinement on best result
    if best_pts is not None and time.time() < deadline - 5:
        refined_pts = _local_refine(best_pts, deadline=deadline - 2)
        refined_radius = _cover_radius(refined_pts)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_pts = refined_pts.copy()

    if best_pts is None:
        # Fallback: Fibonacci spiral
        best_pts = _fibonacci_spiral(N)

    # Ensure all points are on the unit sphere
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)

    return best_pts
