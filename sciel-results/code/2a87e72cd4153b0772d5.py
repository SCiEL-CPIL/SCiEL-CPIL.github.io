import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _covering_radius(points):
    """Compute the covering radius in radians using the convex hull."""
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use random sampling
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        rng = np.random.RandomState(42)
        n_samples = 5000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return np.max(2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1)))
    
    max_r = 0.0
    for simplex in hull.simplices:
        pts = points[simplex]
        if len(pts) != 3:
            continue
        a, b, c = pts[0], pts[1], pts[2]
        v = np.cross(b - a, c - a)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            continue
        v = v / v_norm
        dot_a = np.dot(v, a)
        if dot_a < 0:
            v = -v
            dot_a = -dot_a
        cos_r = np.clip(dot_a, -1.0, 1.0)
        r = np.arccos(cos_r)
        if r > max_r:
            max_r = r
    
    return max_r

def _fibonacci_points(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=42):
    """Generate random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _icosahedron_based_points(n):
    """Generate points based on icosahedron symmetry with perturbations."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    base = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    
    pts = base.copy()
    rng = np.random.RandomState(123)
    while len(pts) < n:
        i, j = rng.randint(0, len(pts), 2)
        if i == j:
            continue
        mid = (pts[i] + pts[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        if mid_norm > 1e-10:
            mid /= mid_norm
            dists = 1.0 - np.sum(mid * pts, axis=1)
            if np.min(dists) > 0.01:
                pts = np.vstack([pts, mid])
    
    return pts[:n]

def _spherical_lloyd_iteration(points, max_iters=200, tol=1e-10, time_limit=None, start_time=None):
    """Spherical Lloyd's algorithm."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        new_points = np.zeros_like(points)
        point_to_facets = [[] for _ in range(len(points))]
        for fi, simplex in enumerate(hull.simplices):
            for idx in simplex:
                point_to_facets[idx].append(fi)
        
        for i in range(len(points)):
            if not point_to_facets[i]:
                new_points[i] = points[i]
                continue
            
            circumcenters = []
            for fi in point_to_facets[i]:
                simplex = hull.simplices[fi]
                pts = points[simplex]
                a, b, c = pts[0], pts[1], pts[2]
                v = np.cross(b - a, c - a)
                v_norm = np.linalg.norm(v)
                if v_norm < 1e-12:
                    continue
                v = v / v_norm
                dot_a = np.dot(v, a)
                if dot_a < 0:
                    v = -v
                circumcenters.append(v)
            
            if not circumcenters:
                new_points[i] = points[i]
                continue
            
            centroid = np.mean(circumcenters, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-12:
                new_points[i] = points[i]
            else:
                new_points[i] = centroid / centroid_norm
        
        diff = np.max(np.linalg.norm(new_points - points, axis=1))
        if diff < tol:
            points = new_points
            break
        
        points = new_points
        cr = _covering_radius(points)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    return best_points, best_cr

def _lbfgs_refine(points, beta, time_limit=None, start_time=None):
    """
    Refine points using L-BFGS-B on a log-sum-exp surrogate of the covering radius.
    """
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    def _facet_data(pts):
        try:
            hull = ConvexHull(pts)
        except Exception:
            return None, None, None
        
        centers = []
        radii = []
        facets = []
        for simplex in hull.simplices:
            pts_f = pts[simplex]
            if len(pts_f) != 3:
                continue
            a, b, c = pts_f[0], pts_f[1], pts_f[2]
            v = np.cross(b - a, c - a)
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-12:
                continue
            v = v / v_norm
            dot_a = np.dot(v, a)
            if dot_a < 0:
                v = -v
                dot_a = -dot_a
            cos_r = np.clip(dot_a, -1.0, 1.0)
            r = np.arccos(cos_r)
            centers.append(v)
            radii.append(r)
            facets.append(simplex)
        
        if not centers:
            return None, None, None
        
        return np.array(centers), np.array(radii), facets
    
    def _objective_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        x = x / norms
        
        centers, radii, facets = _facet_data(x)
        if centers is None:
            return 100.0, np.zeros_like(x_flat)
        
        scaled = radii * beta
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_max = log_sum_exp / beta
        
        weights = np.exp(scaled - max_scaled) / np.sum(np.exp(scaled - max_scaled))
        
        grad_flat = np.zeros_like(x_flat)
        for i, (center, r, weight) in enumerate(zip(centers, radii, weights)):
            simplex = facets[i]
            for idx in simplex:
                p = x[idx]
                cos_r = np.clip(np.dot(center, p), -1.0, 1.0)
                sin_r = np.sqrt(np.maximum(0.0, 1.0 - cos_r * cos_r))
                if sin_r < 1e-10:
                    continue
                grad_p = -center / sin_r
                grad_p_tangent = grad_p - np.dot(grad_p, p) * p
                grad_flat[idx * 3:(idx + 1) * 3] += weight * grad_p_tangent
        
        return soft_max, grad_flat.flatten()
    
    x0 = points.flatten()
    
    try:
        result = minimize(
            _objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-10, 'maxfun': 200}
        )
    except Exception:
        result = None
    
    if result is not None:
        new_points = result.x.reshape(N, 3)
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        new_cr = _covering_radius(new_points)
        if new_cr < best_cr:
            best_cr = new_cr
            best_points = new_points.copy()
    
    return best_points, best_cr

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    initial_configs = [
        _fibonacci_points(N),
        _random_points(N, seed=1),
        _random_points(N, seed=2),
        _icosahedron_based_points(N),
    ]
    
    best_points = initial_configs[0].copy()
    best_cr = _covering_radius(best_points)
    
    for i, init_pts in enumerate(initial_configs):
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        # Stage 1: Lloyd
        lloyd_time = min(remaining * 0.3, 10)
        points, cr = _spherical_lloyd_iteration(init_pts, max_iters=100, time_limit=lloyd_time, start_time=start_time)
        
        # Stage 2: L-BFGS with low beta (smooth)
        remaining = deadline - time.time()
        if remaining < 5:
            continue
        lbfgs_time1 = min(remaining * 0.3, 10)
        points, cr = _lbfgs_refine(points, beta=20.0, time_limit=lbfgs_time1, start_time=start_time)
        
        # Stage 3: L-BFGS with high beta (sharp)
        remaining = deadline - time.time()
        if remaining < 5:
            continue
        lbfgs_time2 = min(remaining * 0.3, 10)
        points, cr = _lbfgs_refine(points, beta=100.0, time_limit=lbfgs_time2, start_time=start_time)
        
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Final polish with highest beta
    remaining = deadline - time.time()
    if remaining > 10:
        final_time = min(remaining * 0.8, 20)
        final_points, final_cr = _lbfgs_refine(best_points, beta=200.0, time_limit=final_time, start_time=start_time)
        if final_cr < best_cr:
            best_cr = final_cr
            best_points = final_points
    
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    return best_points
