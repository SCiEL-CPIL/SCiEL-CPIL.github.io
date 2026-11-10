"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

_DEADLINE = 142.0  # Safety margin before 150s kill

def _normalize(pts):
    """Normalize points to unit sphere."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        norm = np.linalg.norm(pts)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0])
        return pts / norm
    else:
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

def _generate_fibonacci_samples(n_samples):
    """Generate well-distributed sample points on the sphere using Fibonacci spiral."""
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return samples

def _covering_radius_exact(pts):
    """
    Compute exact covering radius by finding the maximum circumradius of all 
    triangular facets of the convex hull.
    """
    try:
        hull = ConvexHull(pts)
        facets = hull.simplices
        max_r = 0.0
        for facet in facets:
            a = pts[facet[0]]
            b = pts[facet[1]]
            c = pts[facet[2]]
            
            # Circumcenter of triangle on sphere
            # Using vector algebra
            v1 = b - a
            v2 = c - a
            
            cross = np.cross(v1, v2)
            cross_norm_sq = np.dot(cross, cross)
            
            if cross_norm_sq < 1e-12:
                # Degenerate
                continue
                
            # Distance from a to circumcenter
            # d = 2 * |a x b| * |b x c| * |c x a| / |...| ... simpler formula:
            # R = abc / (4 * Area)
            side_ab = np.linalg.norm(b - a)
            side_bc = np.linalg.norm(c - b)
            side_ca = np.linalg.norm(a - c)
            
            # Heron's formula for area
            s = (side_ab + side_bc + side_ca) / 2.0
            area_sq = s * (s - side_ab) * (s - side_bc) * (s - side_ca)
            if area_sq < 1e-12:
                continue
            area = np.sqrt(area_sq)
            
            R = (side_ab * side_bc * side_ca) / (4.0 * area)
            
            # R is chordal radius. Convert to angular radius.
            # chord = 2 * sin(theta/2) => theta = 2 * arcsin(chord/2)
            if R <= 2.0:
                theta = 2.0 * np.arcsin(R / 2.0)
            else:
                theta = np.pi # Should not happen for small facets
            
            if theta > max_r:
                max_r = theta
                
        return np.degrees(max_r)
    except Exception:
        return np.pi # Invalid or degenerate hull, return max possible

def _objective_and_grad(x, samples, beta):
    """
    Compute the smooth surrogate objective and its gradient.
    Objective: sum(softmin_i (d(y, x_i)^2)) approximated by -logsumexp(-beta * d^2)
    We want to minimize the max distance. 
    Let f(y) = min_i ||y - x_i||^2.
    Approximate max_y f(y) by integrating or sampling.
    
    Actually, a better surrogate for "minimize max distance" is:
    L = sum_{y in samples} softmin_i (||y - x_i||^2)
    where softmin is a smooth approximation of min.
    softmin_i(z_i) = -1/beta * log(sum(exp(-beta * z_i)))
    
    Gradient of softmin w.r.t x_j:
    Let z_i = ||y - x_i||^2 = 1 - 2 y.x_i + 1 = 2 - 2 y.x_i
    p_j(y) = exp(-beta * z_j) / sum_k exp(-beta * z_k)
    d(softmin)/dx_j = sum_y p_j(y) * d z_j / d x_j
    d z_j / d x_j = -2 (y - x_j)
    So grad_j = sum_y p_j(y) * (-2) * (y - x_j) = 2 * sum_y p_j(y) * (x_j - y)
    """
    M = len(samples)
    # x is (N, 3)
    # samples is (M, 3)
    
    # Compute distances squared: z[i, j] = ||samples[i] - x[j]||^2
    # z = 2 - 2 * samples @ x.T
    dots = samples @ x.T
    z = 2.0 - 2.0 * dots
    
    # For numerical stability in log-sum-exp
    # We need -1/beta * log(sum(exp(-beta * z)))
    # Let u = -beta * z
    u = -beta * z
    u_max = np.max(u, axis=1, keepdims=True)
    
    # Shift
    exp_u = np.exp(u - u_max)
    sum_exp_u = np.sum(exp_u, axis=1, keepdims=True)
    
    # softmin value
    val = -(u_max[:, 0] + np.log(sum_exp_u[:, 0])) / beta
    obj = np.sum(val)
    
    # Probabilities p_j(i) = exp_u[i, j] / sum_exp_u[i, j]
    p = exp_u / sum_exp_u
    
    # Gradient for each point x_j
    # grad_j = 2 * sum_i p[i, j] * (x_j - samples[i])
    # = 2 * x_j * sum_i p[i, j] - 2 * sum_i p[i, j] * samples[i]
    
    # sum_p[j] = sum_i p[i, j]
    sum_p = np.sum(p, axis=0) # (N,)
    
    # weighted_samples[j] = sum_i p[i, j] * samples[i]
    # This is p.T @ samples
    weighted_samples = p.T @ samples # (N, 3)
    
    grad = 2.0 * (x * sum_p[:, np.newaxis] - weighted_samples)
    
    return obj, grad

def _local_optimize_smooth(pts, samples, beta=50.0, max_iters=100, time_limit=10.0, start_time=None):
    """
    Optimize using L-BFGS-B on the smooth surrogate.
    """
    if start_time is None:
        start_time = time.time()
        
    x0 = pts.flatten()
    
    def fun(x):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Timeout")
        val, _ = _objective_and_grad(x.reshape(N, 3), samples, beta)
        return val
        
    def jac(x):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Timeout")
        _, g = _objective_and_grad(x.reshape(N, 3), samples, beta)
        return g.flatten()
        
    try:
        res = minimize(fun, x0, jac=jac, method='L-BFGS-B', 
                       options={'maxiter': max_iters, 'maxfun': 2000})
        x_opt = res.x.reshape(N, 3)
        x_opt = _normalize(x_opt)
        return x_opt
    except Exception:
        return pts

def construct():
    """
    Construct 50 points on unit sphere minimizing covering radius.
    Uses smooth surrogate optimization.
    """
    start_time = time.time()
    deadline = _DEADLINE
    
    # Use a moderate number of samples for the surrogate
    # Too few -> inaccurate gradient. Too many -> slow.
    n_samples = 5000
    samples = _generate_fibonacci_samples(n_samples)
    
    best_pts = None
    best_radius = np.inf
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Start from Fibonacci spiral
    if time.time() - start_time < deadline - 20:
        pts_init = _generate_initial_points('fibonacci', rng)
        remaining = deadline - (time.time() - start_time) - 10
        if remaining > 5:
            opt_pts = _local_optimize_smooth(pts_init, samples, beta=50.0, time_limit=remaining)
            r = _covering_radius_exact(opt_pts)
            if r < best_radius:
                best_radius = r
                best_pts = opt_pts.copy()
                
    # Strategy 2: Start from Random
    for i in range(5):
        if time.time() - start_time > deadline - 15:
            break
        pts_init = _generate_initial_points('random', rng)
        remaining = deadline - (time.time() - start_time) - 10
        if remaining < 5:
            break
        opt_pts = _local_optimize_smooth(pts_init, samples, beta=50.0, time_limit=remaining)
        r = _covering_radius_exact(opt_pts)
        if r < best_radius:
            best_radius = r
            best_pts = opt_pts.copy()
            
    # Strategy 3: Start from Perturbed Fibonacci
    for i in range(5):
        if time.time() - start_time > deadline - 15:
            break
        pts_init = _generate_initial_points('perturbed_fibonacci', rng)
        remaining = deadline - (time.time() - start_time) - 10
        if remaining < 5:
            break
        opt_pts = _local_optimize_smooth(pts_init, samples, beta=50.0, time_limit=remaining)
        r = _covering_radius_exact(opt_pts)
        if r < best_radius:
            best_radius = r
            best_pts = opt_pts.copy()
            
    # Final Polish: Increase beta for sharper approximation and refine
    if best_pts is not None and time.time() - start_time < deadline - 5:
        remaining = deadline - (time.time() - start_time) - 2
        if remaining > 2:
            # Use more samples for final precision? 
            # The gradient is based on samples. 
            # Let's stick to same samples but higher beta.
            opt_pts = _local_optimize_smooth(best_pts, samples, beta=100.0, time_limit=remaining)
            r = _covering_radius_exact(opt_pts)
            if r < best_radius:
                best_radius = r
                best_pts = opt_pts.copy()
                
    if best_pts is None:
        best_pts = _generate_initial_points('fibonacci', rng)
        
    return _normalize(best_pts)

def _generate_initial_points(method='fibonacci', rng=None):
    """Generate initial point configurations using different methods."""
    if method == 'fibonacci':
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return _normalize(pts)
    elif method == 'random':
        if rng is None:
            rng = np.random.default_rng(42)
        pts = rng.normal(size=(N, 3))
        return _normalize(pts)
    elif method == 'perturbed_fibonacci':
        base = _generate_initial_points('fibonacci', rng)
        if rng is None:
            rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.15, size=base.shape)
        return _normalize(base + noise)
    else:
        raise ValueError(f"Unknown method: {method}")
# EVOLVE-BLOCK-END
