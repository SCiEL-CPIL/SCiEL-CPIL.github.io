"""Sphere covering with 50 equal caps - structural rewrite using Riesz energy + LSE cascade."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree

N = 50
TIME_LIMIT = 145.0  # safety margin below 150s

# EVOLVE-BLOCK-START

def _normalize(pts):
    """Normalize points to unit sphere."""
    pts = np.asarray(pts, dtype=float)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute the covering radius in degrees via convex-hull facet circumcenters."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: sample-based estimation
        tree = cKDTree(pts)
        rng = np.random.RandomState(0)
        n_samples = 80000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return float(np.max(2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1)))) * 180.0 / np.pi

    max_r = 0.0
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        v = np.cross(b - a, c - a)
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            continue
        v /= vn
        d = np.dot(v, a)
        if d < 0:
            v = -v
            d = -d
        cos_r = np.clip(d, -1.0, 1.0)
        r = np.arccos(cos_r)
        if r > max_r:
            max_r = r
    return float(np.degrees(max_r))

def _fast_facets(pts):
    """Return list of (radius, circumcenter, facet_indices) for triangular facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return []
    out = []
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        v = np.cross(b - a, c - a)
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            continue
        v /= vn
        d = np.dot(v, a)
        if d < 0:
            v = -v
            d = -d
        cos_r = np.clip(d, -1.0, 1.0)
        r = np.arccos(cos_r)
        out.append((r, v, simplex))
    return out

def _riesz_energy(pts, s=2.0):
    """Compute Riesz s-energy: sum over pairs of |pi - pj|^(-s)."""
    pts = _normalize(pts)
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)  # avoid division by zero
    return float(np.sum(dists ** (-s)))

def _riesz_energy_grad(pts, s=2.0):
    """Gradient of Riesz s-energy w.r.t. points (tangent space)."""
    pts = _normalize(pts)
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    
    # grad_i = -s * sum_{j != i} |pi - pj|^(-s-2) * (pi - pj)
    # Note: for energy E = sum |pi-pj|^(-s), dE/dpi = -s * sum_j |pi-pj|^(-s-2) * (pi-pj)
    # But we want to MINIMIZE energy, so gradient of energy is what we compute.
    # Actually for minimization we want -gradient or we minimize the energy directly.
    
    # dE/dpi = -s * sum_{j!=i} (pi - pj) * |pi - pj|^(-s-2)
    # Let's compute: weight_ij = s * dists[i,j]^(-s-2)
    weights = s * (dists ** (-(s + 2)))
    np.fill_diagonal(weights, 0.0)
    
    # grad_i = -sum_j weights[i,j] * (pi - pj) = sum_j weights[i,j] * (pj - pi)
    # = sum_j weights[i,j] * pj - pi * sum_j weights[i,j]
    sum_w = np.sum(weights, axis=1, keepdims=True)
    weighted_sum = weights @ pts  # (n, 3)
    grad = weighted_sum - sum_w * pts
    
    # Project to tangent space
    grad_tan = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
    return float(np.sum(dists ** (-s))), grad_tan

def _smooth_lse_covering(pts, beta=50.0):
    """Smooth log-sum-exp surrogate for covering radius and its gradient."""
    pts = _normalize(pts)
    facets = _fast_facets(pts)
    if not facets:
        return 0.0, np.zeros_like(pts)

    radii = np.array([f[0] for f in facets])
    centers = np.array([f[1] for f in facets])
    simplices = [f[2] for f in facets]

    m = np.max(radii)
    if m < 1e-12:
        return 0.0, np.zeros_like(pts)
    val = m + np.log(np.sum(np.exp(beta * (radii - m)))) / beta

    w = np.exp(beta * (radii - m))
    w /= w.sum()

    grad = np.zeros_like(pts)
    for i in range(len(radii)):
        center = centers[i]
        simplex = simplices[i]
        for idx in simplex:
            p = pts[idx]
            cos_r = np.clip(np.dot(center, p), -1.0, 1.0)
            sin_r = np.sqrt(np.maximum(0.0, 1.0 - cos_r * cos_r))
            if sin_r < 1e-10:
                continue
            g = -center / sin_r
            g_tan = g - np.dot(g, p) * p
            grad[idx] += w[i] * g_tan

    return val, grad

def _combined_objective(x_flat, riesz_weight=0.01, lse_beta=50.0, riesz_s=2.0):
    """Combined Riesz + LSE covering objective."""
    x = x_flat.reshape(-1, 3)
    x = _normalize(x)
    
    lse_val, lse_grad = _smooth_lse_covering(x, beta=lse_beta)
    riesz_val, riesz_grad = _riesz_energy_grad(x, s=riesz_s)
    
    val = lse_val + riesz_weight * riesz_val
    grad = lse_grad + riesz_weight * riesz_grad
    
    # Project gradient to tangent space
    grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
    return val, grad_tan.flatten()

def _lbfgs_combined(points, max_time=20.0, riesz_weight=0.01, lse_beta=50.0, riesz_s=2.0):
    """L-BFGS on combined Riesz + LSE objective."""
    points = _normalize(points.copy())
    best = points.copy()
    best_cr = _covering_radius(points)
    
    def obj(x_flat):
        return _combined_objective(x_flat, riesz_weight, lse_beta, riesz_s)
    
    x0 = points.flatten()
    try:
        result = minimize(
            obj, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 500}
        )
        x_opt = _normalize(result.x.reshape(-1, 3))
        cr_opt = _covering_radius(x_opt)
        if cr_opt < best_cr:
            best = x_opt.copy()
            best_cr = cr_opt
    except Exception:
        pass
    
    return best, best_cr

def _lbfgs_lse(points, max_time=20.0, beta=50.0):
    """L-BFGS on pure LSE covering objective."""
    points = _normalize(points.copy())
    best = points.copy()
    best_cr = _covering_radius(points)
    
    def obj(x_flat):
        x = x_flat.reshape(-1, 3)
        x = _normalize(x)
        val, grad = _smooth_lse_covering(x, beta=beta)
        grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        return val, grad_tan.flatten()
    
    x0 = points.flatten()
    try:
        result = minimize(
            obj, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 800}
        )
        x_opt = _normalize(result.x.reshape(-1, 3))
        cr_opt = _covering_radius(x_opt)
        if cr_opt < best_cr:
            best = x_opt.copy()
            best_cr = cr_opt
    except Exception:
        pass
    
    return best, best_cr

def _lbfgs_riesz(points, max_time=15.0, s=2.0):
    """L-BFGS on pure Riesz energy (diversification step)."""
    points = _normalize(points.copy())
    best = points.copy()
    best_cr = _covering_radius(points)
    
    def obj(x_flat):
        x = x_flat.reshape(-1, 3)
        x = _normalize(x)
        val, grad = _riesz_energy_grad(x, s=s)
        grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        return val, grad_tan.flatten()
    
    x0 = points.flatten()
    try:
        result = minimize(
            obj, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 300}
        )
        x_opt = _normalize(result.x.reshape(-1, 3))
        cr_opt = _covering_radius(x_opt)
        if cr_opt < best_cr:
            best = x_opt.copy()
            best_cr = cr_opt
    except Exception:
        pass
    
    return best, best_cr

def _spherical_lloyd(points, max_iters=100, time_limit=None, start_time=None):
    """Lloyd-style: each point moves to centroid of adjacent facet circumcenters."""
    points = _normalize(points.copy())
    best = points.copy()
    best_cr = _covering_radius(points)

    if start_time is None:
        start_time = time.time()

    for it in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        facets = _fast_facets(points)
        if not facets:
            break
        centers_per_point = [[] for _ in range(len(points))]
        for r, center, simplex in facets:
            for idx in simplex:
                centers_per_point[idx].append(center)
        new_points = np.zeros_like(points)
        for i in range(len(points)):
            if not centers_per_point[i]:
                new_points[i] = points[i]
            else:
                c = np.mean(centers_per_point[i], axis=0)
                cn = np.linalg.norm(c)
                if cn < 1e-12:
                    new_points[i] = points[i]
                else:
                    new_points[i] = c / cn
        alpha = 0.5
        blended = (1 - alpha) * points + alpha * new_points
        blended = _normalize(blended)
        cr = _covering_radius(blended)
        if cr < best_cr:
            points = blended
            best = points.copy()
            best_cr = cr
        else:
            alpha = 0.2
            blended = (1 - alpha) * points + alpha * new_points
            blended = _normalize(blended)
            cr = _covering_radius(blended)
            if cr < best_cr:
                points = blended
                best = points.copy()
                best_cr = cr
            else:
                break

    return best, best_cr

def _worst_facet_shrinkage(points, max_iters=200, start_time=None, time_limit=None):
    """Shrink worst facet by moving its vertices toward circumcenter."""
    points = _normalize(points.copy())
    best = points.copy()
    best_cr = _covering_radius(points)
    
    if start_time is None:
        start_time = time.time()

    for it in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        facets = _fast_facets(points)
        if not facets:
            break
        
        worst_idx = np.argmax([f[0] for f in facets])
        worst_r, worst_center, worst_simplex = facets[worst_idx]
        
        if worst_r < 1e-10:
            break
        
        step_size = 0.3
        new_points = points.copy()
        for idx in worst_simplex:
            p = points[idx]
            c = worst_center
            cos_angle = np.clip(np.dot(p, c), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            if angle < 1e-10:
                continue
            sin_angle = np.sin(angle)
            if sin_angle < 1e-10:
                continue
            t = step_size
            new_p = (np.sin((1-t)*angle) * p + np.sin(t*angle) * c) / sin_angle
            new_points[idx] = new_p
        
        new_points = _normalize(new_points)
        new_cr = _covering_radius(new_points)
        
        if new_cr < best_cr - 1e-12:
            points = new_points
            best = points.copy()
            best_cr = new_cr
        else:
            step_size *= 0.5
            if step_size < 0.01:
                break

    return best, best_cr

def _simulated_annealing(points, max_iters=500, t_start=0.3, t_end=0.01, time_limit=None, start_time=None):
    """Simulated annealing on covering radius."""
    points = _normalize(points.copy())
    best = points.copy()
    best_cr = _covering_radius(points)
    current_cr = best_cr
    
    if start_time is None:
        start_time = time.time()
    
    n_iter = max_iters
    for it in range(n_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        t = t_start * (t_end / t_start) ** (it / max(n_iter - 1, 1))
        
        # Random perturbation
        rng = np.random.RandomState(it)
        perturbed = points.copy()
        idx = rng.randint(N)
        # Random tangent direction
        rand_vec = rng.randn(3)
        rand_vec -= np.dot(rand_vec, points[idx]) * points[idx]
        rn = np.linalg.norm(rand_vec)
        if rn < 1e-10:
            rand_vec = rng.randn(3)
            rand_vec -= np.dot(rand_vec, points[idx]) * points[idx]
            rn = np.linalg.norm(rand_vec)
        rand_vec /= rn
        step = t * 0.5
        perturbed[idx] = points[idx] + step * rand_vec
        perturbed = _normalize(perturbed)
        
        new_cr = _covering_radius(perturbed)
        delta = new_cr - current_cr
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(t, 1e-10)):
            points = perturbed
            current_cr = new_cr
            if new_cr < best_cr:
                best = points.copy()
                best_cr = new_cr
    
    return best, best_cr

def _fibonacci_points(n=N):
    """Golden-angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(seed):
    """Random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    return _normalize(pts)

def _jittered_fibonacci(seed, jitter_scale=0.15):
    """Fibonacci with random jitter."""
    base = _fibonacci_points()
    rng = np.random.RandomState(seed)
    noise = rng.randn(N, 3) * jitter_scale
    return _normalize(base + noise)

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    
    # Generate diverse initial configurations
    starts = []
    starts.append(_fibonacci_points())
    for seed in [1, 7, 42, 99, 123, 256, 512, 1024]:
        starts.append(_random_points(seed))
    for seed in [3, 11, 37, 73]:
        starts.append(_jittered_fibonacci(seed, 0.15))
    
    best_pts = starts[0]
    best_val = _covering_radius(best_pts)
    
    n_starts = len(starts)
    
    for idx, start in enumerate(starts):
        elapsed = time.time() - start_time
        remaining = TIME_LIMIT - elapsed - 8.0
        if remaining < 5.0:
            break
        
        # Time allocation per start
        time_per_start = min(remaining / max(n_starts - idx, 1), 12.0)
        if time_per_start < 3.0:
            break
        
        try:
            # Phase 1: Riesz energy minimization (diversification)
            pts, val = _lbfgs_riesz(start, max_time=time_per_start * 0.25, s=2.0)
            if val < best_val:
                best_val = val
                best_pts = pts.copy()
            
            # Phase 2: Combined Riesz + LSE
            pts, val = _lbfgs_combined(pts, max_time=time_per_start * 0.35,
                                        riesz_weight=0.005, lse_beta=40.0, riesz_s=2.0)
            if val < best_val:
                best_val = val
                best_pts = pts.copy()
            
            # Phase 3: Pure LSE refinement (cascade beta)
            pts, val = _lbfgs_lse(pts, max_time=time_per_start * 0.2, beta=50.0)
            if val < best_val:
                best_val = val
                best_pts = pts.copy()
            
            pts, val = _lbfgs_lse(pts, max_time=time_per_start * 0.1, beta=100.0)
            if val < best_val:
                best_val = val
                best_pts = pts.copy()
            
            # Phase 4: Lloyd + worst facet
            remaining_for_lloyd = time_per_start * 0.1
            if remaining_for_lloyd > 1.0:
                pts, val = _spherical_lloyd(pts, max_iters=50,
                                             start_time=start_time,
                                             time_limit=time.time() - start_time + remaining_for_lloyd)
                if val < best_val:
                    best_val = val
                    best_pts = pts.copy()
                    
        except Exception:
            continue
    
    # Final polish on best configuration
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed - 3.0
    if remaining > 10.0:
        try:
            # High-precision LSE cascade
            pts, val = _lbfgs_lse(best_pts, max_time=remaining * 0.3, beta=80.0)
            if val < best_val:
                best_val = val
                best_pts = pts.copy()
            
            pts, val = _lbfgs_lse(pts, max_time=remaining * 0.2, beta=150.0)
            if val < best_val:
                best_val = val
                best_pts = pts.copy()
            
            # Simulated annealing polish
            remaining_sa = remaining * 0.3
            if remaining_sa > 3.0:
                pts, val = _simulated_annealing(pts, max_iters=200, t_start=0.15, t_end=0.005,
                                                 time_limit=remaining_sa, start_time=start_time)
                if val < best_val:
                    best_val = val
                    best_pts = pts.copy()
            
            # Final LSE
            remaining_final = remaining * 0.2
            if remaining_final > 2.0:
                pts, val = _lbfgs_lse(pts, max_time=remaining_final, beta=200.0)
                if val < best_val:
                    best_val = val
                    best_pts = pts.copy()
                    
        except Exception:
            pass
    
    return _normalize(best_pts)

# EVOLVE-BLOCK-END
