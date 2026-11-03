N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-14] = 1.0
    return pts / norms


def _facet_circumradii(pts):
    """
    Compute circumradii (in radians) for all triangular facets of the convex hull.
    Returns an array of circumradii.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.array([np.pi / 2])
    
    simplices = hull.simplices
    if len(simplices) == 0:
        return np.array([np.pi / 2])
    
    # Get all facet vertices
    p1 = pts[simplices[:, 0]]
    p2 = pts[simplices[:, 1]]
    p3 = pts[simplices[:, 2]]
    
    # Compute normals
    v1 = p2 - p1
    v2 = p3 - p1
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    mask = norms > 1e-12
    norms[mask] = 1.0 / norms[mask]
    norms[~mask] = 0.0
    normals = normals * norms
    
    # Ensure normals point outward
    dots_p1 = np.sum(normals * p1, axis=1)
    normals[dots_p1 < 0] *= -1
    
    # Compute circumradius for each facet
    # For a spherical triangle, the circumradius is the max angle from facet center to vertices
    # The facet center (on sphere) is the normalized normal
    dots1 = np.sum(normals * p1, axis=1)
    dots2 = np.sum(normals * p2, axis=1)
    dots3 = np.sum(normals * p3, axis=1)
    dots1 = np.clip(dots1, -1.0, 1.0)
    dots2 = np.clip(dots2, -1.0, 1.0)
    dots3 = np.clip(dots3, -1.0, 1.0)
    
    angles1 = np.arccos(dots1)
    angles2 = np.arccos(dots2)
    angles3 = np.arccos(dots3)
    
    cr = np.max(np.stack([angles1, angles2, angles3], axis=1), axis=1)
    
    # Filter out invalid facets
    valid = mask.flatten()
    cr = cr[valid]
    
    if len(cr) == 0:
        return np.array([np.pi / 2])
    
    return cr


def _covering_radius(pts):
    """Covering radius in radians."""
    cr = _facet_circumradii(pts)
    return np.max(cr) if len(cr) > 0 else np.pi / 2


def _smooth_max_circumradii(pts, beta=50.0):
    """
    Smooth approximation of max squared circumradius using log-sum-exp.
    Using squared values gives a smoother objective for gradient-based optimization.
    """
    cr = _facet_circumradii(pts)
    if len(cr) == 0:
        return np.pi / 2
    
    # Use squared circumradii for smoother gradient
    cr_sq = cr ** 2
    
    # Log-sum-exp for numerical stability
    max_cr_sq = np.max(cr_sq)
    if max_cr_sq < 1e-10:
        return 0.0
    
    shifted = cr_sq - max_cr_sq
    log_sum = max_cr_sq + np.log(np.sum(np.exp(beta * shifted))) / beta
    return log_sum


def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def _lbfgs_optimize(pts, deadline, max_iter=200, beta=50.0):
    """
    Optimize using L-BFGS with smooth max squared circumradius objective.
    """
    def objective(x):
        p = _normalize(x.reshape(N, 3))
        return _smooth_max_circumradii(p, beta)
    
    x0 = pts.flatten()
    
    try:
        res = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': 1e-6, 'maxls': 30, 'ftol': 1e-8}
        )
        return _normalize(res.x.reshape(N, 3))
    except Exception:
        return pts


def _nelder_mead_polish(pts, deadline, max_iter=100):
    """
    Nelder-Mead polish for final refinement.
    """
    def objective(x):
        p = _normalize(x.reshape(N, 3))
        return _smooth_max_circumradii(p, beta=100.0)
    
    x0 = pts.flatten()
    
    try:
        res = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        return _normalize(res.x.reshape(N, 3))
    except Exception:
        return pts


def _local_facet_search(pts, deadline, max_iter=200):
    """
    Local search: find worst facet, move its vertices toward the facet center.
    More targeted than moving all nearby points.
    """
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    step_size = 0.05
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        cr = _facet_circumradii(pts)
        if len(cr) == 0:
            break
        
        # Find the worst facet
        worst_idx = np.argmax(cr)
        worst_cr = cr[worst_idx]
        
        # Get the hull to find which facet this is
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        simplices = hull.simplices
        if worst_idx >= len(simplices):
            break
        
        facet = simplices[worst_idx]
        p1, p2, p3 = pts[facet[0]], pts[facet[1]], pts[facet[2]]
        
        # Compute facet center (normal direction)
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            break
        n = n / nn
        if np.dot(n, p1) < 0:
            n = -n
        
        # Move the three vertices of the worst facet toward the center
        new_pts = pts.copy()
        for idx in facet:
            move = (n - new_pts[idx]) * step_size
            new_pts[idx] = _normalize(new_pts[idx] + move)
        
        new_cr = _covering_radius(new_pts)
        if new_cr < best_cr - 1e-12:
            best_cr = new_cr
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            # Gradually reduce step size
            step_size *= 0.95
        else:
            # Try smaller step
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return best_pts


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Phase 1: Generate diverse starting configurations
    starts = [
        _fibonacci_sphere(N),
        _random_sphere(N, 1),
        _random_sphere(N, 2),
        _random_sphere(N, 3),
        _random_sphere(N, 4),
    ]
    
    best_pts = starts[0].copy()
    best_cr = _covering_radius(best_pts)
    
    # Phase 2: L-BFGS optimization from each start
    for i, s in enumerate(starts):
        if time.time() > deadline - 20:
            break
        try:
            optimized = _lbfgs_optimize(s, deadline - 15, max_iter=150, beta=50.0)
            cr = _covering_radius(optimized)
            if cr < best_cr:
                best_cr = cr
                best_pts = optimized.copy()
        except Exception:
            pass
    
    # Phase 3: Local facet search from best
    if time.time() < deadline - 15:
        try:
            local_best = _local_facet_search(best_pts, deadline - 10, max_iter=200)
            cr = _covering_radius(local_best)
            if cr < best_cr:
                best_cr = cr
                best_pts = local_best.copy()
        except Exception:
            pass
    
    # Phase 4: L-BFGS polish with tighter tolerance
    if time.time() < deadline - 10:
        try:
            polished = _lbfgs_optimize(best_pts, deadline - 5, max_iter=100, beta=100.0)
            cr = _covering_radius(polished)
            if cr < best_cr:
                best_cr = cr
                best_pts = polished.copy()
        except Exception:
            pass
    
    # Phase 5: Nelder-Mead final polish
    if time.time() < deadline - 3:
        try:
            nm_best = _nelder_mead_polish(best_pts, deadline - 1, max_iter=50)
            cr = _covering_radius(nm_best)
            if cr < best_cr:
                best_cr = cr
                best_pts = nm_best.copy()
        except Exception:
            pass
    
    # Final validation
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            min_dist = min(min_dist, d)
    if min_dist < 1e-6:
        rng = np.random.default_rng(42)
        best_pts = _normalize(best_pts + rng.randn(N, 3) * 1e-4)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
