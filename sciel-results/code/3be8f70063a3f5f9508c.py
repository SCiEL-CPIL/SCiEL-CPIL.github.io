"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 50
START_TIME = time.time()
TIME_LIMIT = 115.0  # safety margin below 120s

# EVOLVE-BLOCK-START

def _normalize(pts):
    """Normalize points to unit sphere."""
    pts = np.asarray(pts, dtype=float)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius(pts):
    """
    Compute the covering radius in degrees via convex-hull facet circumcenters.
    For each triangular facet, the circumcenter (on the sphere) is the farthest
    point in that Voronoi cell. The covering radius is the max angular distance
    from any facet circumcenter to its vertices.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: sample-based estimation
        from scipy.spatial import cKDTree
        rng = np.random.RandomState(0)
        tree = cKDTree(pts)
        n_samples = 50000
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

def _smooth_sample_objective(x_flat, samples, beta):
    """
    Compute smooth covering radius using log-sum-exp of distances from samples to points.
    x_flat: flattened points (N, 3)
    samples: (M, 3) points on sphere
    beta: temperature for log-sum-exp
    """
    x = x_flat.reshape(-1, 3)
    x = _normalize(x)
    # Compute angular distances between samples and points
    # dists[i, j] is angular distance between samples[i] and x[j]
    dots = samples @ x.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    # For each sample, find min distance to any point
    min_dists = np.min(dists, axis=1)
    
    # Log-sum-exp of min_dists
    m = np.max(min_dists)
    if m < 1e-12:
        val = 0.0
        w = np.ones_like(min_dists) / len(min_dists)
    else:
        val = m + np.log(np.sum(np.exp(beta * (min_dists - m)))) / beta
        w = np.exp(beta * (min_dists - m))
        w /= w.sum()
    
    # Gradient: sum over samples of w[i] * grad(min_dist_i)
    # min_dist_i is the distance to the nearest point x[j_i]
    # We need to identify which point was nearest for each sample.
    # To make it differentiable and handle ties, we use the soft-min logic implicitly.
    # However, for gradient, we can compute the gradient of the distance to the nearest point.
    # d/dx_j (arccos(x_j . y)) = -y / sin(arccos(x_j . y)) projected to tangent space.
    # We weight this by the probability that x_j is the nearest, which is approximated by the softmax weights over distances.
    
    # Let's compute the softmax weights over distances for each sample to distribute the gradient.
    # P(j is nearest for sample i) ~ exp(-beta * dists[i, j]) / sum_k exp(-beta * dists[i, k])
    # This is a soft assignment.
    
    # For efficiency, we only compute gradients for the "active" points, but soft assignment is smoother.
    # Let's use the soft assignment weights.
    # Note: dists is (M, N). We want weights (M, N).
    # exp(-beta * dists) might underflow if beta is large and dists are large.
    # Shift by max dist for numerical stability? No, we want relative weights.
    # Actually, the log-sum-exp was on min_dists. The gradient of the log-sum-exp of min_dists is complex.
    # Alternative: Gradient of the soft-max of the *minimums* is equivalent to the expectation of the gradient of the min.
    # A simpler, robust surrogate: Minimize the sum of squared distances to a "virtual" target? No.
    
    # Let's stick to the facet-based gradient for precision, but use the sample-based for global search?
    # The facet-based gradient is exact for the hull. The sample-based is an approximation.
    # I will implement a hybrid: use sample-based for initial "global" smoothing, then facet-based.
    
    # Gradient calculation for sample-based:
    # val = (1/beta) * log(sum_i exp(beta * min_j d_ij))
    # grad_x_k val = (1/beta) * sum_i exp(beta * min_j d_ij) / sum_l exp(beta * min_l d_l) * grad_x_k (min_j d_ij)
    # The term grad_x_k (min_j d_ij) is non-zero only if k is the argmin for i.
    # In the soft version, we distribute the gradient according to the probability that k is the minimizer.
    
    # Let's compute the soft weights for each sample i over points j.
    # w_ij = exp(-beta * d_ij) / sum_k exp(-beta * d_ik)
    # This is a standard softmax over distances.
    
    # To save memory, we process in chunks of samples if M is large.
    # For M=10000, N=50, this is fine.
    
    grad = np.zeros_like(x)
    if len(samples) > 0:
        # Compute weights (M, N)
        # Use log-sum-exp trick for stability
        max_dists = np.max(dists, axis=1, keepdims=True)
        exp_dists = np.exp(-beta * (dists - max_dists))
        sum_exp = np.sum(exp_dists, axis=1, keepdims=True)
        weights = exp_dists / sum_exp # (M, N)
        
        # For each sample i, and each point j, compute gradient of d_ij
        # d_ij = arccos(x_j . y_i)
        # grad_x_j d_ij = -y_i / sin(d_ij) (tangent)
        # We can vectorize this.
        
        # dists is (M, N). weights is (M, N).
        # We need sin(dists).
        sin_dists = np.sin(dists)
        sin_dists[sin_dists < 1e-12] = 1e-12
        
        # grad_d_ij = -y_i / sin(d_ij)
        # We want sum_i weights[i, j] * grad_x_j (d_ij)
        # grad_x_j (d_ij) is a vector in R^3.
        # Let's compute this term by term.
        
        # For each point j, the gradient contribution is:
        # sum_i weights[i, j] * (-y_i / sin(d_ij))
        # Then project to tangent space at x_j.
        
        # This is a matrix multiplication:
        # W_j = weights[:, j] (M,)
        # Y = samples (M, 3)
        # S_j = 1/sin(dists[:, j]) (M,)
        # vec = sum_i W_j[i] * (-Y[i] * S_j[i])
        # vec = - (W_j * S_j) @ Y  -> (M,) * (M, 3) = (3,)
        
        for j in range(len(x)):
            w_j = weights[:, j]
            s_j = 1.0 / sin_dists[:, j]
            # vec = - (w_j * s_j) @ samples
            vec = -np.dot(w_j * s_j, samples)
            # Project to tangent space at x[j]
            g_tan = vec - np.dot(vec, x[j]) * x[j]
            grad[j] += g_tan
            
        # Scale by beta? 
        # The derivative of (1/beta) log sum exp(beta * min) involves 1/beta * beta * ... = 1.
        # The weights already account for the distribution.
        # So the gradient is just the weighted sum of the distance gradients.
        
    return val, grad.flatten()

def _lbfgs_refine(points, max_time=30.0, start_time=None, beta=50.0):
    """Refine points using L-BFGS-B on a smooth log-sum-exp surrogate."""
    from scipy.optimize import minimize

    points = _normalize(points.copy())
    best = points.copy()
    best_cr = _covering_radius(points)

    if start_time is None:
        start_time = time.time()

    # Pre-compute samples for the global phase
    rng = np.random.RandomState(42)
    n_samples = 5000
    samples = rng.randn(n_samples, 3)
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)

    def objective_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        x = _normalize(x)
        val, grad = _smooth_covering_radius_grad(x, beta=beta)
        grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        return val, grad_tan.flatten()

    def objective_and_grad_sample(x_flat):
        val, grad = _smooth_sample_objective(x_flat, samples, beta)
        x = x_flat.reshape(-1, 3)
        x = _normalize(x)
        grad_tan = grad.reshape(-1, 3) - np.sum(grad.reshape(-1, 3) * x, axis=1, keepdims=True) * x
        return val, grad_tan.flatten()

    x0 = points.flatten()
    
    # Phase 1: Global optimization with sample-based smooth objective
    if time.time() - start_time < max_time * 0.5:
        try:
            result = minimize(
                objective_and_grad_sample,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 200}
            )
            x_opt = _normalize(result.x.reshape(-1, 3))
            cr_opt = _covering_radius(x_opt)
            if cr_opt < best_cr:
                best = x_opt.copy()
                best_cr = cr_opt
        except Exception:
            pass

    # Phase 2: Local optimization with facet-based smooth objective
    if time.time() - start_time < max_time * 0.8:
        try:
            x0 = best.flatten()
            result = minimize(
                objective_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 500}
            )
            x_opt = _normalize(result.x.reshape(-1, 3))
            cr_opt = _covering_radius(x_opt)
            if cr_opt < best_cr:
                best = x_opt.copy()
                best_cr = cr_opt
        except Exception:
            pass

    # Phase 3: High precision polish
    if time.time() - start_time < max_time:
        try:
            x0 = best.flatten()
            def obj_high_beta(x_flat):
                x = x_flat.reshape(-1, 3)
                x = _normalize(x)
                val, grad = _smooth_covering_radius_grad(x, beta=200.0)
                grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
                return val, grad_tan.flatten()
                
            result = minimize(
                obj_high_beta,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-14, 'maxfun': 300}
            )
            x_opt = _normalize(result.x.reshape(-1, 3))
            cr_opt = _covering_radius(x_opt)
            if cr_opt < best_cr:
                best = x_opt.copy()
                best_cr = cr_opt
        except Exception:
            pass

    return best, best_cr

def _worst_facet_shrinkage(points, max_iters=500, start_time=None, time_limit=None):
    """
    Deterministic iterative method: identify the convex hull facet with the largest
    spherical circumradius and move its three vertices along the great circle toward
    the facet's circumcenter. This directly shrinks the worst empty cap.
    """
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
        
        # Find the worst facet
        worst_idx = np.argmax([f[0] for f in facets])
        worst_r, worst_center, worst_simplex = facets[worst_idx]
        
        if worst_r < 1e-10:
            break
        
        # Move the three vertices of the worst facet toward the circumcenter
        # using spherical interpolation (great circle)
        step_size = 0.3  # fraction of the way to move
        new_points = points.copy()
        for idx in worst_simplex:
            p = points[idx]
            c = worst_center
            # Spherical interpolation: move p toward c by step_size * angular distance
            cos_angle = np.clip(np.dot(p, c), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            if angle < 1e-10:
                continue
            # Interpolate along great circle
            t = step_size
            sin_angle = np.sin(angle)
            if sin_angle < 1e-10:
                continue
            new_p = (np.sin((1-t)*angle) * p + np.sin(t*angle) * c) / sin_angle
            new_points[idx] = new_p
        
        new_points = _normalize(new_points)
        new_cr = _covering_radius(new_points)
        
        if new_cr < best_cr - 1e-12:
            points = new_points
            best = points.copy()
            best_cr = new_cr
        else:
            # Try smaller step
            step_size *= 0.5
            if step_size < 0.01:
                break

    return best, best_cr

def _spherical_lloyd(points, max_iters=200, time_limit=None, start_time=None):
    """Lloyd-style: each point moves to the average circumcenter of adjacent facets."""
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

def _fibonacci_points(n=N):
    """Generate points using golden-angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(seed):
    """Generate random points on sphere."""
    np.random.seed(seed)
    pts = np.random.randn(N, 3)
    return _normalize(pts)

def _icosahedron_based_points():
    """Generate points based on icosahedron symmetry, then add more."""
    # Start with icosahedron vertices (12 points)
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico = _normalize(ico)
    
    # Add 38 more points using Fibonacci spiral, then optimize
    fib = _fibonacci_points(N)
    # Blend: use icosahedron as base, add perturbed Fibonacci points
    # Actually, just use Fibonacci as it's a good uniform distribution
    return fib

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    global START_TIME
    START_TIME = time.time()
    
    # Generate diverse initial configurations
    starts = []
    
    # 1. Golden-angle spiral
    starts.append(_fibonacci_points())
    
    # 2. Random starts with different seeds
    for seed in [1, 2, 3, 4, 5, 7, 13, 42, 99, 123]:
        starts.append(_random_points(seed))
    
    # 3. Perturbed golden spiral
    np.random.seed(7)
    base = starts[0].copy()
    noise = np.random.randn(N, 3) * 0.1
    starts.append(_normalize(base + noise))
    
    # 4. More perturbation
    np.random.seed(13)
    noise = np.random.randn(N, 3) * 0.3
    starts.append(_normalize(base + noise))
    
    # Optimize from each start, keep the best
    best_pts = starts[0]
    best_val = _covering_radius(best_pts)
    
    # Allocate time per start
    n_starts = len(starts)
    elapsed = time.time() - START_TIME
    time_per_start = max(5.0, (TIME_LIMIT - elapsed - 20.0) / n_starts)
    
    for idx, start in enumerate(starts):
        remaining = TIME_LIMIT - (time.time() - START_TIME) - 10.0
        if remaining < 5.0:
            break
        
        max_time_for_this = min(time_per_start, remaining - 5.0)
        if max_time_for_this < 3.0:
            break
        
        try:
            # Phase 1: Worst facet shrinkage (fast, deterministic) - reduced time
            pts, val = _worst_facet_shrinkage(start, max_iters=100, 
                                               start_time=START_TIME,
                                               time_limit=time.time() - START_TIME + max_time_for_this * 0.2)
            if val < best_val:
                best_val = val
                best_pts = pts
            
            # Phase 2: Lloyd iteration - reduced time
            pts, val = _spherical_lloyd(pts, max_iters=50,
                                         start_time=START_TIME,
                                         time_limit=time.time() - START_TIME + max_time_for_this * 0.4)
            if val < best_val:
                best_val = val
                best_pts = pts
            
            # Phase 3: L-BFGS refinement - increased time for new sample-based global phase
            remaining_for_lbfgs = max(5.0, max_time_for_this * 0.6)
            pts, val = _lbfgs_refine(pts, max_time=remaining_for_lbfgs,
                                      start_time=START_TIME)
            if val < best_val:
                best_val = val
                best_pts = pts
        except Exception:
            continue
    
    # Final polish with tighter tolerances if time allows
    remaining = TIME_LIMIT - (time.time() - START_TIME) - 3.0
    if remaining > 15.0:
        try:
            # More aggressive refinement
            pts, val = _worst_facet_shrinkage(best_pts, max_iters=300,
                                               start_time=START_TIME,
                                               time_limit=time.time() - START_TIME + remaining * 0.3)
            if val < best_val:
                best_val = val
                best_pts = pts
            
            pts, val = _spherical_lloyd(pts, max_iters=200,
                                         start_time=START_TIME,
                                         time_limit=time.time() - START_TIME + remaining * 0.5)
            if val < best_val:
                best_val = val
                best_pts = pts
            
            remaining_for_lbfgs = max(5.0, remaining * 0.4)
            pts, val = _lbfgs_refine(pts, max_time=remaining_for_lbfgs,
                                      start_time=START_TIME)
            if val < best_val:
                best_val = val
                best_pts = pts
        except Exception:
            pass
    
    # Final safety check
    remaining = TIME_LIMIT - (time.time() - START_TIME)
    if remaining < 5.0:
        # Return immediately
        return _normalize(best_pts)
    
    return _normalize(best_pts)

# EVOLVE-BLOCK-END
