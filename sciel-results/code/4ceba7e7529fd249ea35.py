"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 50


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute covering radius in degrees via convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    # For each facet, find the circumcenter and compute the angular radius
    max_ang = 0.0
    for simplex in hull.simplices:
        # Get the facet points
        facet_pts = pts[simplex]
        # Solve for circumcenter in the plane of the facet
        # The circumcenter c satisfies |c - p_i|^2 = R^2 for all i
        # c = p_0 + sum_j alpha_j (p_j - p_0)
        # Minimize |c - p_i|^2
        # For a simplex (triangle), solve linear system
        n = len(simplex)
        if n == 3:
            p0, p1, p2 = facet_pts
            # Set up: find c such that |c-p0|^2 = |c-p1|^2 = |c-p2|^2
            # 2 c . (p1-p0) = |p1|^2 - |p0|^2 = 0 (unit sphere)
            # 2 c . (p2-p0) = 0
            # Also c lies in the plane: c = p0 + a*(p1-p0) + b*(p2-p0)
            # Substitute:
            # 2(p0 + a v1 + b v2) . v1 = 0  where v1 = p1-p0, v2 = p2-p0
            # 2(p0 + a v1 + b v2) . v2 = 0
            v1 = p1 - p0
            v2 = p2 - p0
            A = np.array([[v1 @ v1, v1 @ v2],
                          [v2 @ v1, v2 @ v2]])
            b = np.array([-(p0 @ v1), -(p0 @ v2)])
            try:
                ab = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            c = p0 + ab[0] * v1 + ab[1] * v2
            # Check c is not zero
            cnorm = np.linalg.norm(c)
            if cnorm < 1e-10:
                continue
            c = c / cnorm  # normalize to unit sphere
            # The circumradius is the angle between c and any facet point
            # Actually the covering radius for this facet is the max angle from c to facet points
            # But we need the angular distance from c to the farthest point on the facet's cap
            # The cap is defined by the plane through the facet points
            # The circumcenter of the facet (on the unit sphere) gives the center of the cap
            # The radius is the angle from c to any of the facet vertices
            ang = np.arccos(np.clip(np.dot(c, p0), -1.0, 1.0))
            # Also check p1 and p2
            ang1 = np.arccos(np.clip(np.dot(c, p1), -1.0, 1.0))
            ang2 = np.arccos(np.clip(np.dot(c, p2), -1.0, 1.0))
            facet_ang = max(ang, ang1, ang2)
            max_ang = max(max_ang, facet_ang)
    return np.degrees(max_ang)


def _soft_max_covering(pts, beta=50.0):
    """
    Soft-max surrogate for the covering radius.
    Uses the convex hull facets' circumradii with a log-sum-exp.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    angles = []
    for simplex in hull.simplices:
        facet_pts = pts[simplex]
        n = len(simplex)
        if n == 3:
            p0, p1, p2 = facet_pts
            v1 = p1 - p0
            v2 = p2 - p0
            A = np.array([[v1 @ v1, v1 @ v2],
                          [v2 @ v1, v2 @ v2]])
            b = np.array([-(p0 @ v1), -(p0 @ v2)])
            try:
                ab = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            c = p0 + ab[0] * v1 + ab[1] * v2
            cnorm = np.linalg.norm(c)
            if cnorm < 1e-10:
                continue
            c = c / cnorm
            ang = np.arccos(np.clip(np.dot(c, p0), -1.0, 1.0))
            angles.append(ang)
    
    if not angles:
        return 180.0
    
    angles = np.array(angles)
    # Log-sum-exp for soft max
    m = np.max(angles)
    if m < 1e-10:
        return 0.0
    log_sum = m + np.log(np.sum(np.exp(angles - m)))
    return np.degrees(log_sum / beta)


def _grad_soft_max_covering(pts, beta=50.0):
    """Numerical gradient of the soft-max covering radius."""
    pts = _normalize(pts)
    eps = 1e-6
    grad = np.zeros_like(pts)
    f0 = _soft_max_covering(pts, beta)
    for i in range(N):
        for j in range(3):
            pts_p = pts.copy()
            pts_p[i, j] += eps
            pts_p = _normalize(pts_p)
            f_p = _soft_max_covering(pts_p, beta)
            grad[i, j] = (f_p - f0) / eps
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Fibonacci spiral initial points
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_radius = _covering_radius(best_pts)
    
    # Try multiple random restarts with optimization
    np.random.seed(42)
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            current = best_pts.copy()
        else:
            # Random perturbation of the best or a fresh start
            if restart < 3:
                current = best_pts + 0.3 * np.random.randn(N, 3)
            else:
                # Fresh Fibonacci with different offset
                i2 = np.arange(N) + 0.5 + np.random.rand()
                z2 = 1.0 - 2.0 * np.minimum(i2, N) / N
                phi2 = np.pi * (1.0 + np.sqrt(5.0)) * i2
                r2 = np.sqrt(1.0 - z2 * z2)
                current = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
            current = _normalize(current)
        
        # Phase 1: L-BFGS on soft-max surrogate
        try:
            res = minimize(
                lambda x: _soft_max_covering(x, beta=50.0),
                current.flatten(),
                jac=lambda x: _grad_soft_max_covering(x.reshape(N, 3), beta=50.0).flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if time.time() > deadline - 15.0:
                break
            if res.success or res.fun < best_radius:
                current = _normalize(res.x.reshape(N, 3))
                cr = _covering_radius(current)
                if cr < best_radius:
                    best_radius = cr
                    best_pts = current.copy()
        except Exception:
            pass
        
        # Phase 2: Polish with tighter tolerances
        if time.time() > deadline - 10.0:
            break
        
        try:
            res2 = minimize(
                lambda x: _soft_max_covering(x, beta=100.0),
                current.flatten(),
                jac=lambda x: _grad_soft_max_covering(x.reshape(N, 3), beta=100.0).flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.success or res2.fun < best_radius:
                current = _normalize(res2.x.reshape(N, 3))
                cr = _covering_radius(current)
                if cr < best_radius:
                    best_radius = cr
                    best_pts = current.copy()
        except Exception:
            pass
    
    # Final check
    final = _normalize(best_pts)
    return final


# EVOLVE-BLOCK-START
# (The code above is the implementation; this block marker is for the evolution system)
# EVOLVE-BLOCK-END
