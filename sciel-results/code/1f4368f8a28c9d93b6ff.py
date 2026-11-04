"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms


def _covering_radius(points):
    """
    Compute the covering radius in degrees for a set of unit points.
    Uses convex hull facet circumradii (vectorized).
    """
    if len(points) < 3:
        return 180.0
    try:
        hull = ConvexHull(points)
    except:
        return 180.0
    
    simplices = hull.simplices
    if len(simplices) == 0:
        return 180.0

    p1 = points[simplices[:, 0]]
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1, keepdims=True)
    mask = norm_n[:, 0] > 1e-10
    if not np.any(mask):
        return 180.0
    
    n = n[mask] / norm_n[mask]
    p1m = p1[mask]
    
    # Ensure normals point outward
    dots = np.sum(n * p1m, axis=1)
    n[dots < 0] *= -1
    
    h = np.sum(n * p1m, axis=1)
    h = np.clip(h, -1.0, 1.0)
    angles = np.arccos(h)
    
    return float(np.max(angles)) * 180.0 / np.pi


def _covering_radius_radians(points):
    """Covering radius in radians (for optimization)."""
    return _covering_radius(points) * np.pi / 180.0


def _facet_smooth_surrogate(params, beta=50.0):
    """
    Smooth surrogate based on convex hull facet circumradii (vectorized).
    This provides a gradient for L-BFGS-B.
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
    except:
        return np.pi
        
    simplices = hull.simplices
    if len(simplices) == 0:
        return np.pi

    p1 = pts[simplices[:, 0]]
    p2 = pts[simplices[:, 1]]
    p3 = pts[simplices[:, 2]]
    
    # Normal to the plane
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1, keepdims=True)
    
    # Handle degenerate triangles
    mask = norm_n[:, 0] > 1e-10
    if not np.all(mask):
        return np.pi
        
    n = n[mask] / norm_n[mask]
    p1m = p1[mask]
    
    # Ensure normals point towards the interior of the sphere relative to the triangle?
    # The circumcenter c is parallel to n.
    # We need c.p1 > 0.
    dots = np.sum(n * p1m, axis=1)
    n[dots < 0] *= -1
    
    # Calculate circumradius
    # cos(R) = c.p1
    cos_r = np.sum(n * p1m, axis=1)
    cos_r = np.clip(cos_r, -1.0, 1.0)
    radii = np.arccos(cos_r)
    
    # Soft-max (log-sum-exp)
    # We want to minimize the max radius, so we approximate max(radii) with logsumexp(beta * radii) / beta
    max_val = np.max(beta * radii)
    logsumexp = max_val + np.log(np.sum(np.exp(beta * radii - max_val)))
    return logsumexp / beta


def _fibonacci_points():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(seed=42):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(N, 3)
    return _normalize(points)


def _icosahedron_based_points():
    """Generate points based on icosahedron vertices with perturbations."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = _normalize(verts)
    
    rng = np.random.RandomState(123)
    points = [verts[i] for i in range(12)]
    
    while len(points) < N:
        base = verts[rng.randint(12)]
        noise = rng.randn(3) * 0.3
        new_pt = base + noise
        new_pt = new_pt / np.linalg.norm(new_pt)
        min_dist = min(np.linalg.norm(new_pt - p) for p in points)
        if min_dist > 0.1:
            points.append(new_pt)
    
    return np.array(points[:N])


def _spherical_lloyd_iteration(points, max_iter=30, time_limit=5, damping=0.3):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation) with adaptive damping.
    Uses vectorized operations for speed.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    current_damping = damping
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(points)
        except:
            break
        
        simplices = hull.simplices
        if len(simplices) == 0:
            break

        p1 = points[simplices[:, 0]]
        p2 = points[simplices[:, 1]]
        p3 = points[simplices[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n, axis=1, keepdims=True)
        mask = norm_n[:, 0] > 1e-10
        if not np.any(mask):
            break
        
        n_valid = n[mask] / norm_n[mask]
        p1_valid = p1[mask]
        
        dots = np.sum(n_valid * p1_valid, axis=1)
        n_valid[dots < 0] *= -1
        
        h = np.sum(n_valid * p1_valid, axis=1)
        h = np.clip(h, -1.0, 1.0)
        facet_angles = np.arccos(h)
        
        # Accumulate targets for each point
        acc = np.zeros((N, 3))
        weights = np.zeros(N)
        
        simplices_valid = simplices[mask]
        for fi in range(len(simplices_valid)):
            angle = facet_angles[fi]
            n_vec = n_valid[fi]
            idxs = simplices_valid[fi]
            acc[idxs] += angle * n_vec
            weights[idxs] += angle
        
        new_points = points.copy()
        for pi in range(N):
            if weights[pi] > 1e-10:
                target = acc[pi] / np.linalg.norm(acc[pi])
                new_points[pi] = points[pi] + current_damping * (target - points[pi])
                new_points[pi] = new_points[pi] / np.linalg.norm(new_points[pi])
        
        new_points = _normalize(new_points)
        new_cost = _covering_radius(new_points)
        
        if new_cost < current_cost:
            current_cost = new_cost
            points = new_points
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
            current_damping = min(0.8, current_damping * 1.1)
        else:
            current_damping *= 0.5
            if current_damping < 0.01:
                break
    
    return best_points


def _nelder_mead_optimize(points, time_limit=5):
    """
    Local optimization using Nelder-Mead on the true covering radius.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        return _covering_radius_radians(pts)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'maxfev': 2000})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    total_time_limit = 145.0
    start_time = time.time()
    
    def _time_left():
        return total_time_limit - (time.time() - start_time)
    
    # Initializations
    initializations = [
        _fibonacci_points(),
        _random_points(seed=42),
        _random_points(seed=123),
        _icosahedron_based_points()
    ]
    
    best_points = None
    best_cost = np.inf
    
    # Cascade of betas for the smooth surrogate
    betas = [10.0, 50.0, 100.0, 200.0]
    
    for init_pts in initializations:
        if _time_left() < 20:
            break
        
        # 1. Quick Lloyd's relaxation to get out of bad local minima
        pts = _spherical_lloyd_iteration(init_pts, max_iter=20, time_limit=min(5, _time_left() - 15), damping=0.3)
        cost = _covering_radius(pts)
        
        # 2. L-BFGS-B cascade on the facet smooth surrogate
        for beta in betas:
            if _time_left() < 10:
                break
            
            def objective(params):
                return _facet_smooth_surrogate(params, beta=beta)
            
            x0 = pts.flatten()
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                  options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000})
                new_pts = _normalize(result.x.reshape(N, 3))
                new_cost = _covering_radius(new_pts)
                
                if new_cost < cost:
                    pts = new_pts
                    cost = new_cost
            except:
                pass
        
        # 3. Final Nelder-Mead polish on the true objective
        if _time_left() > 5:
            pts = _nelder_mead_optimize(pts, time_limit=min(5, _time_left() - 2))
            cost = _covering_radius(pts)
        
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
            
    if best_points is None:
        best_points = _fibonacci_points()
    
    # Final high-precision polish if time allows
    if _time_left() > 10:
        best_points = _nelder_mead_optimize(best_points, time_limit=min(8, _time_left() - 2))
    
    return _normalize(best_points)
# EVOLVE-BLOCK-END
