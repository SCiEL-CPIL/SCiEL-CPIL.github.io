"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _circumcenter_robust(a, b, c):
    """Robust circumcenter of triangle on sphere."""
    for e in ((a, b, c), (b, a, c), (a, c, b)):
        cand = np.cross(e[0], e[1])
        cn = np.linalg.norm(cand)
        if np.dot(cand, e[2]) > 0 and cn > 1e-12:
            return cand / cn
    ctr = a + b + c
    cn = np.linalg.norm(ctr)
    if cn < 1e-9:
        return None
    return ctr / cn


def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull facets."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    facets = hull.simplices
    if len(facets) == 0:
        return 180.0

    max_r = 0.0
    for f in facets:
        a, b, c = pts[f[0]], pts[f[1]], pts[f[2]]
        ctr = _circumcenter_robust(a, b, c)
        if ctr is None:
            continue
        if np.dot(ctr, (a + b + c)) < 0:
            ctr = -ctr
        r = max(
            np.arccos(np.clip(np.dot(ctr, a), -1, 1)),
            np.arccos(np.clip(np.dot(ctr, b), -1, 1)),
            np.arccos(np.clip(np.dot(ctr, c), -1, 1)),
        )
        if r > max_r:
            max_r = r

    return np.degrees(max_r)


def _fibonacci_spiral(n, jitter=0.0, rng=None):
    """Generate Fibonacci spiral points on sphere with optional jitter."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if jitter > 0 and rng is not None:
        noise = rng.normal(0, jitter, size=pts.shape)
        pts = pts + noise
        pts = _normalize(pts)
    return pts


def _compute_circumradii(pts):
    """
    Compute circumradius (in radians) for each facet of the convex hull.
    Returns array of circumradii and the facet indices.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.array([np.pi]), np.array([[]])
    
    facets = hull.simplices
    if len(facets) == 0:
        return np.array([np.pi]), np.array([[]])
    
    radii = np.zeros(len(facets))
    for i, f in enumerate(facets):
        a, b, c = pts[f[0]], pts[f[1]], pts[f[2]]
        ctr = _circumcenter_robust(a, b, c)
        if ctr is None:
            radii[i] = np.pi
            continue
        if np.dot(ctr, (a + b + c)) < 0:
            ctr = -ctr
        r = max(
            np.arccos(np.clip(np.dot(ctr, a), -1, 1)),
            np.arccos(np.clip(np.dot(ctr, b), -1, 1)),
            np.arccos(np.clip(np.dot(ctr, c), -1, 1)),
        )
        radii[i] = r
    
    return radii, facets


def _smooth_max_objective_and_grad(pts, theta_target, beta=50.0):
    """
    Objective: sum of (r_i - theta_target)^2 for all facets, where r_i is circumradius.
    Use smooth approximation: we want all circumradii to be close to theta_target.
    Gradient w.r.t. point positions is complex, so we use a simplified approach:
    For each facet, the gradient of r_i w.r.t. vertices points away from circumcenter.
    """
    radii, facets = _compute_circumradii(pts)
    n_facets = len(facets)
    
    if n_facets == 0:
        return 0.0, np.zeros_like(pts)
    
    # Objective: sum of (r_i - theta_target)^2
    residuals = radii - theta_target
    obj = np.sum(residuals**2)
    
    # Gradient: for each facet, d(r_i)/d(point_j) for j in facet
    # The circumcenter direction is the gradient of the angular distance
    grad = np.zeros_like(pts)
    for i, f in enumerate(facets):
        a, b, c = pts[f[0]], pts[f[1]], pts[f[2]]
        ctr = _circumcenter_robust(a, b, c)
        if ctr is None:
            continue
        if np.dot(ctr, (a + b + c)) < 0:
            ctr = -ctr
        
        # Gradient of circumradius w.r.t. each vertex points away from circumcenter
        # (increasing distance from circumcenter increases circumradius)
        # Actually, the gradient of the angular distance from circumcenter to vertex
        # is in the direction from circumcenter to vertex, but we want to minimize
        # |r_i - theta_target|, so:
        # d(r_i - theta_target)/d(vertex) is proportional to (vertex - ctr) normalized
        # but projected onto the sphere tangent plane
        
        dr = residuals[i]
        for idx in f:
            v = pts[idx]
            # Project (v - ctr) onto tangent plane at v
            proj = v - np.dot(v, ctr) * v  # this is not quite right
            # Better: gradient of angular distance from ctr to v w.r.t. v
            # is (v - (v·ctr)ctr) / (|v - (v·ctr)ctr| * sin(angle))
            # Simplified: direction from ctr to v, projected to tangent
            diff = v - np.dot(v, ctr) * ctr
            dn = np.linalg.norm(diff)
            if dn > 1e-12:
                diff = diff / dn
            else:
                diff = np.zeros(3)
            grad[idx] += dr * diff
    
    return obj, grad


def _lm_optimize(pts, deadline, theta_init=None):
    """
    Levenberg-Marquardt style optimization.
    We define residuals as (r_i - theta_target) for each facet.
    But theta_target is also a variable. To handle this, we use a two-stage approach:
    1. Fix theta_target to current max circumradius
    2. Optimize points to make all circumradii equal to theta_target
    3. Update theta_target
    """
    best_pts = pts.copy()
    best_score = _covering_radius_deg(pts)
    
    if theta_init is None:
        radii, _ = _compute_circumradii(pts)
        theta_target = np.max(radii)
    else:
        theta_target = theta_init
    
    t0 = time.time()
    while time.time() < deadline - 2:
        # Compute current circumradii
        radii, facets = _compute_circumradii(pts)
        if len(facets) == 0:
            break
        
        # Update theta_target to current max
        theta_target = np.max(radii)
        
        # Compute residuals
        residuals = radii - theta_target
        
        # Compute gradient
        _, grad = _smooth_max_objective_and_grad(pts, theta_target)
        
        # LM step: try to reduce the residuals
        # Use a simple gradient descent with adaptive step
        step = 0.002
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            new_pts = pts - step * grad
            new_pts = _normalize(new_pts)
            
            new_radii, _ = _compute_circumradii(new_pts)
            if len(new_radii) == 0:
                break
            new_max = np.max(new_radii)
            
            if new_max < np.max(radii) - 1e-10:
                pts = new_pts
                radii = new_radii
                # Compute new gradient
                _, grad = _smooth_max_objectise_and_grad(pts, theta_target) if False else _smooth_max_objective_and_grad(pts, theta_target)
                step = min(step * 1.1, 0.01)
            else:
                step *= 0.7
                if step < 1e-8:
                    break
        
        # Check improvement
        current_score = _covering_radius_deg(pts)
        if current_score < best_score - 1e-12:
            best_score = current_score
            best_pts = pts.copy()
    
    return best_pts, best_score


def _grad_descent_optimize(pts, deadline):
    """
    Gradient descent with momentum, targeting equalization of circumradii.
    """
    best_pts = pts.copy()
    best_score = _covering_radius_deg(pts)
    step = 0.003
    mom = np.zeros_like(pts)
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline - 2:
        radii, facets = _compute_circumradii(pts)
        if len(facets) == 0:
            break
        
        theta_target = np.max(radii)
        
        # Compute gradient
        _, grad = _smooth_max_objective_and_grad(pts, theta_target)
        
        # Momentum update
        mom = 0.5 * mom + 0.5 * grad
        new_pts = pts - step * mom
        new_pts = _normalize(new_pts)
        
        new_score = _covering_radius_deg(new_pts)
        if new_score < best_score - 1e-12:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts
            no_improve = 0
            step = min(step * 1.02, 0.01)
        else:
            no_improve += 1
            step *= 0.85
            if step < 1e-7:
                step = 0.003
                mom *= 0.2
            if no_improve > max_no_improve:
                rng = np.random.default_rng(hash(tuple(pts.ravel().round(6))) % 2**32)
                noise = rng.normal(0, 0.05, size=pts.shape)
                pts = _normalize(best_pts + noise)
                step = 0.003
                mom = np.zeros_like(pts)
                no_improve = 0
    
    return best_pts, best_score


def construct():
    t0 = time.time()
    deadline = 145.0  # safety margin under 150s limit

    # Multiple starts with different jitter levels
    jitter_levels = [0.0, 0.01, 0.02, 0.05, 0.08]
    rng_main = np.random.default_rng(42)

    best_pts = None
    best_score = 180.0

    for jitter in jitter_levels:
        if time.time() > deadline - 5:
            break

        pts = _fibonacci_spiral(N, jitter=jitter, rng=rng_main)
        # Give each start a reasonable time budget
        remaining = deadline - 5 - (time.time() - t0)
        start_budget = min(25.0, max(8.0, remaining))
        
        # Use gradient descent first
        pts_opt, score = _grad_descent_optimize(pts, t0 + start_budget)

        if score < best_score:
            best_score = score
            best_pts = pts_opt

    # Final polish on the best configuration if time remains
    if best_pts is not None and time.time() < deadline - 10:
        pts_opt, score = _grad_descent_optimize(best_pts, deadline - 2)
        if score < best_score:
            best_score = score
            best_pts = pts_opt

    if best_pts is None:
        # Fallback: Fibonacci spiral
        best_pts = _fibonacci_spiral(N)

    return best_pts
# EVOLVE-BLOCK-END
