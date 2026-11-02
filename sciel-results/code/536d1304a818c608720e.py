"""Improved covering-radius optimizer for 50 points on the unit sphere."""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 50
TIME_LIMIT = 110.0  # safety margin before 120s kill


def _covering_radius(pts):
    """Compute the covering radius in degrees for points on the unit sphere.
    
    The covering radius is the maximum angular distance from any point on the
    sphere to its nearest chosen point. This equals the maximum circumradius
    of the facets of the convex hull (for facets whose circumcenter lies on
    the sphere side, i.e., the cap is empty).
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return _covering_radius_brute(pts)
    
    facets = hull.simplices
    F = len(facets)
    
    # Vectorized circumcenter computation
    a = pts[facets[:, 0]]  # (F, 3)
    b = pts[facets[:, 1]]
    c = pts[facets[:, 2]]
    
    d1 = a - b
    d2 = a - c
    cross = np.cross(d1, d2)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    p = cross / norms  # (F, 3)
    
    # Orient outward: dot with centroid should be positive
    centroids = (a + b + c) / 3.0
    dots = np.einsum('ij,ij->i', p, centroids)
    p[dots < 0] *= -1
    
    # Cos of circumradius for each facet
    cos_vals = np.einsum('ij,ij->i', p, a)
    cos_vals = np.clip(cos_vals, -1.0, 1.0)
    
    min_cos = cos_vals.min()
    return np.degrees(np.arccos(min_cos))


def _covering_radius_brute(pts, n_samples=50000):
    """Brute-force estimate of covering radius by sampling the sphere."""
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = samples @ pts.T
    max_dots = dots.max(axis=1)
    min_cos = max_dots.min()
    min_cos = np.clip(min_cos, -1.0, 1.0)
    return np.degrees(np.arccos(min_cos))


def _params_to_pts(params):
    """Convert spherical coordinate params to Cartesian points on unit sphere."""
    theta = params[0::2]
    phi = params[1::2]
    sin_theta = np.sin(theta)
    pts = np.column_stack([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        np.cos(theta)
    ])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _soft_max_radius(pts, beta=50.0):
    """Compute soft-max covering radius using log-sum-exp over facet circumradii.
    
    Returns (soft_radius_degrees, facet_circumcenters, cos_vals) for gradient use.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use brute force estimate
        return _covering_radius_brute(pts), None, None
    
    facets = hull.simplices
    F = len(facets)
    
    a = pts[facets[:, 0]]
    b = pts[facets[:, 1]]
    c = pts[facets[:, 2]]
    
    d1 = a - b
    d2 = a - c
    cross = np.cross(d1, d2)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    p = cross / norms
    
    centroids = (a + b + c) / 3.0
    dots = np.einsum('ij,ij->i', p, centroids)
    p[dots < 0] *= -1
    
    cos_vals = np.einsum('ij,ij->i', p, a)
    cos_vals = np.clip(cos_vals, -1.0, 1.0)
    
    # Angles in radians
    angles = np.arccos(cos_vals)
    
    # Soft max via log-sum-exp
    max_angle = angles.max()
    exp_vals = np.exp(beta * (angles - max_angle))
    log_sum_exp = max_angle + np.log(exp_vals.sum()) / beta
    
    return np.degrees(log_sum_exp), p, cos_vals


def _soft_max_gradient(params, beta=50.0):
    """Compute gradient of soft-max covering radius w.r.t. spherical params.
    
    Uses central differences on the soft-max objective for stability.
    """
    grad = np.zeros_like(params)
    eps = 1e-6
    
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus, _, _ = _soft_max_radius(_params_to_pts(params_plus), beta)
        
        params_minus = params.copy()
        params_minus[i] -= eps
        f_minus, _, _ = _soft_max_radius(_params_to_pts(params_minus), beta)
        
        grad[i] = (f_plus - f_minus) / (2 * eps)
    
    return grad


def _random_params():
    """Generate random spherical coordinates."""
    theta = np.random.uniform(0, np.pi, N)
    phi = np.random.uniform(0, 2 * np.pi, N)
    params = np.zeros(2 * N)
    params[0::2] = theta
    params[1::2] = phi
    return params


def _repulsion_relax(pts, n_iters=100, time_budget=5.0):
    """Repulsion-based relaxation to spread points evenly on sphere."""
    start = time.time()
    pts = pts.copy()
    
    for it in range(n_iters):
        if time.time() - start > time_budget:
            break
        
        # Compute pairwise repulsion forces (inverse square)
        dots = pts @ pts.T
        np.fill_diagonal(dots, 0)
        # Force = -gradient of sum of squared distances, which pushes apart
        # For points on sphere, project force to tangent plane
        forces = np.zeros_like(pts)
        for i in range(N):
            # Repulsion from all other points
            f = np.zeros(3)
            for j in range(N):
                if i == j:
                    continue
                d = pts[i] - pts[j]
                dist_sq = np.dot(d, d)
                if dist_sq < 1e-12:
                    continue
                f += d / (dist_sq ** 1.5)
            # Project to tangent plane
            f -= np.dot(f, pts[i]) * pts[i]
            forces[i] = f
        
        # Small step
        step = 0.01 / (1.0 + it * 0.01)
        pts += step * forces
        # Renormalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
    
    return pts


def _fibonacci_params():
    """Generate Fibonacci spiral parameters as initial guess."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    z = np.clip(z, -1.0 + 1e-10, 1.0 - 1e-10)
    theta = np.arccos(z)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    params = np.zeros(2 * N)
    params[0::2] = theta
    params[1::2] = phi
    return params


def _facet_targeted_move(pts, max_iters=50, time_budget=3.0):
    """Move vertices of near-worst facets toward their circumcenters."""
    start = time.time()
    pts = pts.copy()
    
    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        facets = hull.simplices
        F = len(facets)
        
        a = pts[facets[:, 0]]
        b = pts[facets[:, 1]]
        c = pts[facets[:, 2]]
        
        d1 = a - b
        d2 = a - c
        cross = np.cross(d1, d2)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        p = cross / norms
        
        centroids = (a + b + c) / 3.0
        dots = np.einsum('ij,ij->i', p, centroids)
        p[dots < 0] *= -1
        
        cos_vals = np.einsum('ij,ij->i', p, a)
        cos_vals = np.clip(cos_vals, -1.0, 1.0)
        
        # Find worst facets (lowest cos = largest angle)
        worst_idx = np.argsort(cos_vals)[:max(1, F // 5)]
        
        # For each worst facet, nudge its vertices slightly toward circumcenter
        step = 0.05 / (1.0 + it * 0.05)
        for fi in worst_idx:
            circum = p[fi]
            for vi in facets[fi]:
                # Move vertex toward circumcenter, projected to tangent
                d = circum - pts[vi]
                d -= np.dot(d, pts[vi]) * pts[vi]
                pts[vi] += step * d
        
        # Renormalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
    
    return pts


def _local_search(params, max_iters=200, tol=1e-8, time_limit=None, beta=50.0):
    """Gradient descent with adaptive step size on spherical coordinates using soft-max."""
    start_time = time.time()
    current = params.copy()
    current_obj, _, _ = _soft_max_radius(_params_to_pts(current), beta)
    
    for it in range(max_iters):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Compute gradient
        grad = _soft_max_gradient(current, beta)
        
        if np.isnan(grad).any() or np.isinf(grad).any():
            break
        
        # Adaptive step size
        step = 0.02
        improved = False
        for trial in range(15):
            new_params = current - step * grad
            new_obj, _, _ = _soft_max_radius(_params_to_pts(new_params), beta)
            
            if new_obj < current_obj - 1e-10:
                current = new_params
                current_obj = new_obj
                improved = True
                break
            else:
                step *= 0.5
        
        if not improved:
            break
        
        if np.abs(grad).max() < tol:
            break
    
    return current, _covering_radius(_params_to_pts(current))


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Start with Fibonacci spiral
    best_params = _fibonacci_params()
    best_pts = _params_to_pts(best_params)
    best_obj = _covering_radius(best_pts)
    
    # Phase 1: Repulsion relaxation for better initial spread
    remaining = deadline - time.time()
    if remaining > 10:
        relaxed_pts = _repulsion_relax(best_pts, n_iters=200, time_budget=min(8, remaining - 10))
        relaxed_obj = _covering_radius(relaxed_pts)
        if relaxed_obj < best_obj:
            best_pts = relaxed_pts
            best_obj = relaxed_obj
            # Convert back to params for further optimization
            theta = np.arccos(np.clip(best_pts[:, 2], -1, 1))
            phi = np.arctan2(best_pts[:, 1], best_pts[:, 0])
            best_params = np.zeros(2 * N)
            best_params[0::2] = theta
            best_params[1::2] = phi
    
    # Phase 2: Soft-max gradient descent
    remaining = deadline - time.time()
    if remaining > 10:
        best_params, best_obj = _local_search(
            best_params, max_iters=150, time_limit=min(30, remaining - 10), beta=50.0
        )
        best_pts = _params_to_pts(best_params)
    
    # Phase 3: Facet-targeted moves for final polish
    remaining = deadline - time.time()
    if remaining > 5:
        polished_pts = _facet_targeted_move(best_pts, max_iters=30, time_budget=min(5, remaining - 5))
        polished_obj = _covering_radius(polished_pts)
        if polished_obj < best_obj:
            best_pts = polished_pts
            best_obj = polished_obj
    
    # Phase 4: Another round of soft-max gradient from improved starting point
    remaining = deadline - time.time()
    if remaining > 8:
        theta = np.arccos(np.clip(best_pts[:, 2], -1, 1))
        phi = np.arctan2(best_pts[:, 1], best_pts[:, 0])
        current_params = np.zeros(2 * N)
        current_params[0::2] = theta
        current_params[1::2] = phi
        current_params, current_obj = _local_search(
            current_params, max_iters=100, time_limit=min(20, remaining - 5), beta=100.0
        )
        current_pts = _params_to_pts(current_params)
        if current_obj < best_obj:
            best_pts = current_pts
            best_obj = current_obj
    
    # Phase 5: Final facet polish
    remaining = deadline - time.time()
    if remaining > 3:
        polished_pts = _facet_targeted_move(best_pts, max_iters=20, time_budget=min(3, remaining - 2))
        polished_obj = _covering_radius(polished_pts)
        if polished_obj < best_obj:
            best_pts = polished_pts
            best_obj = polished_obj
    
    # Ensure valid output
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_pts = best_pts / norms
    
    return best_pts
