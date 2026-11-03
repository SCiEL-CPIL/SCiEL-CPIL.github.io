import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130
_DEADLINE = time.time() + 130.0  # safety margin before 150s kill


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _covering_radius(points, beta=200.0):
    """Smooth approximation of max angular distance to nearest point."""
    points = _normalize(points)
    # Pairwise dot products
    dots = points @ points.T
    np.fill_diagonal(dots, 1.0)
    # For each point, find max distance (min dot) to nearest other point
    # But covering radius needs max over sphere, which equals max over facets
    # Use the fact that for a spherical code, covering radius = max circumradius of facets
    # Approximate: for each point, the local covering is half the max gap to neighbors
    # Better: use the convex hull facet approach
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_r = 0.0
        for tri in facets:
            a, b, c = points[tri[0]], points[tri[1]], points[tri[2]]
            # Circumcenter of spherical triangle
            # Use the fact that the circumcenter is along the normal to the plane
            # containing a, b, c at the appropriate distance
            normal = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(normal)
            if n_norm < 1e-12:
                continue
            normal = normal / n_norm
            # The circumcenter on the sphere is at distance d from origin
            # where d is such that the angle from circumcenter to each vertex is equal
            # For points on unit sphere, the circumcenter of a spherical triangle
            # with angular side lengths can be computed
            # Alternative: the covering radius for this facet is the max angular
            # distance from the facet's center to its vertices
            # The center of the spherical triangle is the normalized sum (for acute)
            # or we can use the circumcenter formula
            # Simple approach: the farthest point on the sphere from all three vertices
            # is the circumcenter of the great-circle triangle
            # For a triangle with vertices a,b,c on unit sphere, the circumcenter
            # (as a point on the sphere) can be found as follows:
            # The plane containing a,b,c has equation n . x = d where d = n . a
            # The circumcenter is at n * (1/d) if |d| > 0
            d = np.dot(normal, a)
            if abs(d) < 1e-12:
                # Degenerate: triangle contains origin
                # Use the largest angle vertex as approximate center
                # and estimate radius from there
                continue
            circum = normal / d
            c_norm = np.linalg.norm(circum)
            if c_norm < 1e-12:
                continue
            circum = circum / c_norm
            # Angular distance from circumcenter to each vertex
            for v in [a, b, c]:
                ang = np.arccos(np.clip(np.dot(circum, v), -1.0, 1.0))
                max_r = max(max_r, ang)
    except Exception:
        pass
    # Also compute a lower bound based on nearest-neighbor distances
    dots = points @ points.T
    np.fill_diagonal(dots, 1.0)
    min_dots = np.min(dots, axis=1)
    max_nn_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
    max_r = max(max_r, np.pi / 2.0)  # at least 90 deg if in hemisphere
    # For a proper covering, the covering radius is the max facet circumradius
    # Combine with a smooth surrogate
    return np.degrees(max_r)


def _covering_radius_lse(points, beta=50.0):
    """Log-sum-exp based smooth surrogate for covering radius."""
    points = _normalize(points)
    # Sample points on the sphere and find max min-distance
    # Use Fibonacci sphere sampling
    n_samples = 2000
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # Compute angular distances from each sample to all N points
    dots = samples @ points.T  # (n_samples, N)
    # Angular distance = arccos(dot)
    # We want max over samples of min over points of arccos(dot)
    # = max over samples of arccos(max over points of dot)
    max_dots = np.max(dots, axis=1)
    angles = np.arccos(np.clip(max_dots, -1.0, 1.0))
    # Smooth max via log-sum-exp
    max_angle = np.max(angles)
    logsum = np.log(np.sum(np.exp(beta * (angles - max_angle)))) / beta + max_angle
    return np.degrees(logsum)


def _optimize_from_start(points, max_iter=100, tol=1e-8):
    """Local optimization to minimize covering radius."""
    points = _normalize(points.copy())
    
    def objective(x):
        pts = _normalize(x.reshape(N, 3))
        return _covering_radius_lse(pts, beta=30.0)
    
    def gradient(x):
        pts = _normalize(x.reshape(N, 3))
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(x)
        f0 = objective(x)
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            fp = objective(xp)
            xm = x.copy()
            xm[i] -= eps
            fm = objective(xm)
            grad[i] = (fp - fm) / (2 * eps)
        return grad
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'gtol': tol, 'maxfun': 5000})
    
    if result.success or (result.fun < objective(x0)):
        best_pts = _normalize(result.x.reshape(N, 3))
    else:
        best_pts = points
    
    return best_pts, result.fun


def construct():
    global _DEADLINE
    t0 = time.time()
    
    # Stage 1: Multiple initializations
    inits = []
    
    # Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(_normalize(fib))
    
    # Random initializations
    rng = np.random.default_rng(42)
    for seed in [1, 2, 3]:
        rng_s = np.random.default_rng(seed)
        rand_pts = rng_s.normal(size=(N, 3))
        rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1, keepdims=True)
        inits.append(rand_pts)
    
    # Stage 2: Electrostatic repulsion to spread points
    def electrostatic_energy(pts):
        dots = pts @ pts.T
        np.fill_diagonal(dots, 1.0)
        # Repulsion: -log(1 - dot) or similar
        # Use 1/(1-dot) for strong repulsion when close
        repulsion = np.sum(1.0 / (1.0 - dots + 1e-10))
        # Also penalize being in a hemisphere: encourage isotropy
        mean = np.mean(pts, axis=0)
        anisotropy = np.linalg.norm(mean) ** 2
        return repulsion + 10.0 * anisotropy
    
    best_points = None
    best_score = np.inf
    
    for init in inits:
        if time.time() > _DEADLINE - 30:
            break
        
        # Optimize electrostatic energy
        try:
            res_e = minimize(electrostatic_energy, init.flatten(), method='L-BFGS-B',
                           options={'maxiter': 50, 'maxfun': 2000})
            e_pts = _normalize(res_e.x.reshape(N, 3))
        except Exception:
            e_pts = init
        
        # Now minimize covering radius
        if time.time() > _DEADLINE - 20:
            break
        
        try:
            opt_pts, score = _optimize_from_start(e_pts, max_iter=50, tol=1e-7)
            if score < best_score:
                best_score = score
                best_points = opt_pts
        except Exception:
            pass
        
        if time.time() > _DEADLINE - 10:
            break
    
    if best_points is None:
        best_points = inits[0]
    
    # Final polish: more iterations on best
    if time.time() < _DEADLINE - 5:
        try:
            opt_pts, score = _optimize_from_start(best_points, max_iter=200, tol=1e-10)
            if score < best_score:
                best_points = opt_pts
        except Exception:
            pass
    
    # Ensure valid: check no coincident points, not in hemisphere
    best_points = _normalize(best_points)
    # Verify shape
    assert best_points.shape == (N, 3)
    assert np.all(np.isfinite(best_points))
    
    return best_points
