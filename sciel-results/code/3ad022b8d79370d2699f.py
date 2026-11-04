"""Rewritten program for sphere covering with 50 points using a different structural approach.

Key structural changes:
1. Replace convex-hull-based Lloyd with a direct gradient-based optimization of a smooth
   log-sum-exp surrogate for the covering radius, using the Delaunay triangulation on the sphere.
2. Use a multi-stage optimization: first Riesz energy to get a good initial spread, then
   smooth covering radius minimization with adaptive temperature, then local search with
   random restarts on the worst facets.
3. Use spherical geometry directly: represent points as unit vectors, compute gradients
   on the tangent space, and use L-BFGS-B with a smooth objective that differentiates
   through the max over facets via log-sum-exp.
"""

import numpy as np
import time
from scipy.spatial import ConvexHull, cKDTree
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 50

def _normalize(pts):
    """Normalize points to unit sphere."""
    pts = np.asarray(pts, dtype=float)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius(points):
    """Compute the covering radius in radians using the convex hull."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        tree = cKDTree(points)
        rng = np.random.RandomState(42)
        n_samples = 20000
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

def _facet_circumcenters(points):
    """Compute circumcenters and radii for all triangular facets of the convex hull."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None
    
    centers = []
    radii = []
    simplices_valid = []
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
        centers.append(v)
        radii.append(r)
        simplices_valid.append(simplex)
    
    if not centers:
        return None, None, None
    
    return np.array(centers), np.array(radii), simplices_valid

def _smooth_covering_radius_grad(x, temperature=0.05):
    """
    Compute smooth log-sum-exp approximation of covering radius and its gradient.
    x is a flat array of shape (N*3,).
    """
    pts = x.reshape(N, 3)
    pts = _normalize(pts)
    
    centers, radii, simplices_valid = _facet_circumcenters(pts)
    if centers is None:
        tree = cKDTree(pts)
        rng = np.random.RandomState(42)
        n_samples = 5000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        max_d = np.max(dists)
        max_r = 2.0 * np.arcsin(np.clip(max_d / 2.0, 0, 1))
        return max_r, np.zeros_like(x)
    
    T = max(temperature, 1e-6)
    scaled = radii / T
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    obj = log_sum_exp * T
    
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)
    
    grad_pts = np.zeros_like(pts)
    for v, simplex, w in zip(centers, simplices_valid, weights):
        if w < 1e-10:
            continue
        for idx in simplex:
            p = pts[idx]
            dot_vp = np.dot(v, p)
            sin_r = np.sqrt(np.maximum(0.0, 1.0 - dot_vp * dot_vp))
            if sin_r < 1e-12:
                continue
            # d(r)/d(p) = -v / sin(r) projected onto tangent space
            dr_dp = -v / sin_r
            dr_dp_tangent = dr_dp - np.dot(dr_dp, p) * p
            grad_pts[idx] += w * dr_dp_tangent
    
    return obj, (grad_pts * T).flatten()

def _riesz_energy_grad(x, s=2.0):
    """Compute Riesz s-energy and its gradient. Lower energy = more uniform."""
    pts = x.reshape(N, 3)
    pts = _normalize(pts)
    
    # Vectorized pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(np.maximum(dist2, 1e-15))
    
    energy = np.sum(dist ** (-s))
    
    # Gradient
    grad = np.zeros_like(pts)
    for i in range(N):
        d = dist[i, :]
        d[i] = np.inf
        # d/dp_i sum_{j!=i} |p_i - p_j|^(-s) = -s * sum_{j!=i} (p_i - p_j) / |p_i - p_j|^(s+2)
        contrib = -s * (pts[i] - pts) / (d ** (s + 2))[:, np.newaxis]
        contrib[i, :] = 0
        grad[i] = np.sum(contrib, axis=0)
        # Project onto tangent plane
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    
    return energy, grad.flatten()

def _lbfgs_smooth(points, time_limit=None, start_time=None, temperature=0.05, maxiter=300):
    """Refine points using L-BFGS-B with smooth covering radius objective."""
    points = _normalize(points)
    x0 = points.flatten()
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    try:
        result = minimize(
            lambda x: _smooth_covering_radius_grad(x, temperature)[0],
            x0,
            jac=lambda x: _smooth_covering_radius_grad(x, temperature)[1],
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 5000}
        )
        if result.fun < best_cr:
            new_pts = _normalize(result.x.reshape(N, 3))
            cr = _covering_radius(new_pts)
            if cr < best_cr:
                best_cr = cr
                best_points = new_pts.copy()
    except Exception:
        pass
    
    # Check time
    if time_limit is not None and start_time is not None:
        if time.time() - start_time > time_limit:
            return best_points, best_cr
    
    return best_points, best_cr

def _lbfgs_riesz(points, time_limit=None, start_time=None, s=2.0, maxiter=200):
    """Refine points using L-BFGS-B with Riesz energy objective."""
    points = _normalize(points)
    x0 = points.flatten()
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    try:
        result = minimize(
            lambda x: _riesz_energy_grad(x, s)[0],
            x0,
            jac=lambda x: _riesz_energy_grad(x, s)[1],
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 3000}
        )
        if result.success:
            new_pts = _normalize(result.x.reshape(N, 3))
            cr = _covering_radius(new_pts)
            if cr < best_cr:
                best_cr = cr
                best_points = new_pts.copy()
    except Exception:
        pass
    
    return best_points, best_cr

def _spherical_gd(points, obj_func, grad_func, lr=0.01, max_iters=500, time_limit=None, start_time=None):
    """Spherical gradient descent with adaptive learning rate."""
    points = _normalize(points)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        x = points.flatten()
        obj, grad = grad_func(x)
        
        # Adaptive learning rate
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        step = lr * grad / grad_norm
        
        new_points = _normalize(points + step.reshape(N, 3))
        new_cr = _covering_radius(new_points)
        
        if new_cr < best_cr:
            best_cr = new_cr
            best_points = new_points.copy()
            points = new_points
        else:
            lr *= 0.9
    
    return best_points, best_cr

def _spherical_lloyd(points, max_iters=1000, tol=1e-12, time_limit=None, start_time=None):
    """Spherical Lloyd's algorithm using Voronoi cell centroids."""
    points = _normalize(points)
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
        
        # For each point, find all facets it belongs to, compute circumcenters, average
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
            else:
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

def _simulated_annealing(points, time_limit=None, start_time=None, T0=0.12, T1=0.005, n_iter=3000):
    """Simulated annealing to escape local minima."""
    points = _normalize(points)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    current_cr = best_cr
    
    rng = np.random.RandomState(42)
    
    for it in range(n_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        alpha = it / max(n_iter - 1, 1)
        T = T0 * (T1 / T0) ** alpha
        
        # Perturb random points
        n_pert = rng.randint(1, 5)
        indices = rng.choice(N, size=n_pert, replace=False)
        perturbed = points.copy()
        for idx in indices:
            noise = rng.randn(3) * T
            perturbed[idx] = points[idx] + noise
            perturbed[idx] /= np.linalg.norm(perturbed[idx])
        
        cr_new = _covering_radius(perturbed)
        
        if cr_new < current_cr:
            points = perturbed
            current_cr = cr_new
            if cr_new < best_cr:
                best_cr = cr_new
                best_points = points.copy()
        else:
            accept_prob = np.exp(-(cr_new - current_cr) / max(T, 1e-10))
            if rng.random() < accept_prob:
                points = perturbed
                current_cr = cr_new
    
    return best_points, best_cr

def _fibonacci_points(n, offset=0.5):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _jittered_fibonacci_points(n, jitter=0.05, seed=42):
    """Generate jittered Fibonacci spiral points."""
    rng = np.random.RandomState(seed)
    pts = _fibonacci_points(n)
    for i in range(n):
        noise = rng.randn(3) * jitter
        pts[i] = pts[i] + noise
        pts[i] /= np.linalg.norm(pts[i])
    return pts

def _random_points(n, seed=42):
    """Generate random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _icosahedron_subdivided_points(n):
    """Generate points by subdividing an icosahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Subdivide each triangle
    def subdivide_triangle(a, b, c, depth):
        if depth == 0:
            return [a, b, c]
        ab = (a + b) / 2
        ab /= np.linalg.norm(ab)
        bc = (b + c) / 2
        bc /= np.linalg.norm(bc)
        ca = (c + a) / 2
        ca /= np.linalg.norm(ca)
        return (subdivide_triangle(a, ab, ca, depth-1) +
                subdivide_triangle(ab, b, bc, depth-1) +
                subdivide_triangle(ca, bc, c, depth-1) +
                subdivide_triangle(ab, bc, ca, depth-1))
    
    # Get icosahedron faces
    hull = ConvexHull(ico)
    points = []
    for simplex in hull.simplices:
        a, b, c = ico[simplex[0]], ico[simplex[1]], ico[simplex[2]]
        points.extend(subdivide_triangle(a, b, c, 2))
    
    points = np.array(points)
    # Remove duplicates approximately
    unique = [points[0]]
    for p in points[1:]:
        if np.min(np.linalg.norm(p - np.array(unique), axis=1)) > 1e-6:
            unique.append(p)
    points = np.array(unique)
    
    # If we have more than n, sample; if less, add random
    if len(points) >= n:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(points), size=n, replace=False)
        points = points[indices]
    else:
        rng = np.random.RandomState(42)
        extra = rng.randn(n - len(points), 3)
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        points = np.vstack([points, extra])
    
    return points

def _multi_stage_optimize(points, time_limit=None, start_time=None):
    """Multi-stage optimization: Riesz -> smooth CR -> Lloyd -> SA -> final polish."""
    points = _normalize(points)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    # Stage 1: Riesz energy to get good initial spread
    remaining = (deadline - time.time()) if time_limit is None else time_limit
    if remaining > 20:
        points, cr = _lbfgs_riesz(points, time_limit=remaining * 0.2, start_time=start_time, s=2.0, maxiter=150)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Stage 2: Smooth covering radius optimization
    remaining = (deadline - time.time()) if time_limit is None else time_limit
    if remaining > 15:
        points, cr = _lbfgs_smooth(points, time_limit=remaining * 0.3, start_time=start_time, temperature=0.05, maxiter=300)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Stage 3: Lloyd relaxation
    remaining = (deadline - time.time()) if time_limit is None else time_limit
    if remaining > 10:
        points, cr = _spherical_lloyd(points, max_iters=500, time_limit=remaining * 0.25, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Stage 4: Simulated annealing
    remaining = (deadline - time.time()) if time_limit is None else time_limit
    if remaining > 15:
        points, cr = _simulated_annealing(points, time_limit=remaining * 0.3, start_time=start_time, T0=0.1, T1=0.005, n_iter=2000)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Stage 5: Final smooth CR polish with lower temperature
    remaining = (deadline - time.time()) if time_limit is None else time_limit
    if remaining > 10:
        points, cr = _lbfgs_smooth(points, time_limit=remaining * 0.3, start_time=start_time, temperature=0.02, maxiter=200)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Stage 6: Final Lloyd
    remaining = (deadline - time.time()) if time_limit is None else time_limit
    if remaining > 5:
        points, cr = _spherical_lloyd(points, max_iters=1000, tol=1e-13, time_limit=remaining * 0.5, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    return best_points, best_cr

# Global deadline
deadline = None

def construct():
    global deadline
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Generate diverse starting points
    starts = [
        _fibonacci_points(N, offset=0.5),
        _fibonacci_points(N, offset=0.0),
        _fibonacci_points(N, offset=1.0),
        _jittered_fibonacci_points(N, jitter=0.05, seed=42),
        _jittered_fibonacci_points(N, jitter=0.1, seed=123),
        _random_points(N, seed=42),
        _random_points(N, seed=123),
        _random_points(N, seed=999),
        _icosahedron_subdivided_points(N),
    ]
    
    best_points = None
    best_cr = np.inf
    
    # Phase 1: Multi-start with full multi-stage optimization
    remaining = deadline - time.time()
    time_per_start = max(10, remaining * 0.5 / len(starts))
    
    for i, sp in enumerate(starts):
        remaining = deadline - time.time()
        if remaining < 15:
            break
        
        t_limit = min(time_per_start, remaining * 0.8)
        points, cr = _multi_stage_optimize(sp, time_limit=t_limit, start_time=start_time)
        
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
        
        # Update time budget
        elapsed = time.time() - start_time
        remaining_starts = len(starts) - i - 1
        if remaining_starts > 0:
            time_per_start = max(10, (deadline - time.time()) * 0.6 / remaining_starts)
    
    if best_points is None:
        best_points = _fibonacci_points(N)
        best_cr = _covering_radius(best_points)
    
    # Phase 2: Additional refinement on best solution
    remaining = deadline - time.time()
    if remaining > 20:
        points, cr = _multi_stage_optimize(best_points, time_limit=remaining * 0.4, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Phase 3: Perturbation-based local search
    remaining = deadline - time.time()
    if remaining > 15:
        rng = np.random.RandomState(42)
        for _ in range(10):
            remaining = deadline - time.time()
            if remaining < 10:
                break
            
            n_pert = rng.randint(2, 6)
            indices = rng.choice(N, size=n_pert, replace=False)
            perturbed = best_points.copy()
            for idx in indices:
                noise = rng.randn(3) * 0.06
                perturbed[idx] = best_points[idx] + noise
                perturbed[idx] /= np.linalg.norm(perturbed[idx])
            
            t_limit = min(8, remaining * 0.5)
            points, cr = _multi_stage_optimize(perturbed, time_limit=t_limit, start_time=start_time)
            if cr < best_cr:
                best_cr = cr
                best_points = points.copy()
    
    # Phase 4: Final high-precision polish
    remaining = deadline - time.time()
    if remaining > 10:
        points, cr = _lbfgs_smooth(best_points, time_limit=remaining * 0.3, start_time=start_time, temperature=0.01, maxiter=300)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
        
        remaining = deadline - time.time()
        if remaining > 5:
            points, cr = _spherical_lloyd(best_points, max_iters=2000, tol=1e-14, time_limit=remaining * 0.5, start_time=start_time)
            if cr < best_cr:
                best_cr = cr
                best_points = points.copy()
    
    # Normalize final result
    best_points = _normalize(best_points)
    
    return best_points
