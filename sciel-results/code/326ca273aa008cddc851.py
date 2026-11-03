import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius_exact(points):
    """Compute the exact covering radius in radians using convex hull."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: sample-based estimate
        rng = np.random.RandomState(42)
        samples = rng.randn(10000, 3)
        samples = _normalize(samples)
        # Compute min distance from samples to points
        # This is O(N*M)
        dists = np.zeros((len(samples),))
        for i in range(len(samples)):
            # Vectorized dot product
            dots = np.clip(np.sum(points * samples[i], axis=1), -1.0, 1.0)
            min_dist = np.arccos(np.max(dots))
            dists[i] = min_dist
        return np.max(dists)
    
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

def _generate_samples(n_samples):
    """Generate fixed sample points on the sphere for the surrogate."""
    # Use Fibonacci spiral for uniform distribution
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return samples

def _soft_objective_and_grad(x_flat, samples, alpha, beta):
    """
    Objective: Soft-max over samples of soft-min distance to points.
    x_flat: flattened array of N*3 points.
    samples: array of M*3 sample points.
    alpha: sharpness of soft-min.
    beta: sharpness of soft-max.
    """
    x = x_flat.reshape(N, 3)
    x = _normalize(x)
    
    # Compute distance matrix D of shape (M, N)
    # D[j, k] = arccos(clip(sum(x[k] * samples[j]), -1, 1))
    # To avoid arccos in the inner loop for speed and stability, we can work with cosines?
    # No, the distance metric is angular.
    # Let's compute dots first.
    dots = np.clip(samples @ x.T, -1.0, 1.0) # Shape (M, N)
    dists = np.arccos(dots) # Shape (M, N)
    
    # Soft-min over k (points) for each j (sample)
    # d_j = -1/alpha * log( sum_k exp(-alpha * dists[j, k]) )
    # For numerical stability, subtract max
    neg_alpha_dists = -alpha * dists
    max_neg = np.max(neg_alpha_dists, axis=1, keepdims=True)
    exp_neg = np.exp(neg_alpha_dists - max_neg)
    sum_exp = np.sum(exp_neg, axis=1, keepdims=True)
    log_sum_exp = max_neg + np.log(sum_exp)
    d_soft = -log_sum_exp / alpha # Shape (M, 1) -> flatten to (M,)
    d_soft = d_soft.flatten()
    
    # Soft-max over j (samples)
    # L = 1/beta * log( sum_j exp(beta * d_soft[j]) )
    beta_d = beta * d_soft
    max_beta_d = np.max(beta_d)
    exp_beta_d = np.exp(beta_d - max_beta_d)
    sum_exp_beta = np.sum(exp_beta_d)
    log_sum_exp_beta = max_beta_d + np.log(sum_exp_beta)
    obj = log_sum_exp_beta / beta
    
    # Gradient
    # We need d(obj)/dx
    # obj = (1/beta) log( sum_j exp(beta * d_j) )
    # d(obj)/dx = (1/beta) * ( sum_j exp(beta * d_j) * beta * d(d_j)/dx ) / ( sum_j exp(beta * d_j) )
    #           = sum_j w_j * d(d_j)/dx  where w_j = exp(beta * d_j) / sum_k exp(beta * d_k)
    
    weights_j = exp_beta_d / sum_exp_beta # Shape (M,)
    
    # Now d(d_j)/dx
    # d_j = -1/alpha log( sum_k exp(-alpha * dists[j, k]) )
    # d(d_j)/dx = -1/alpha * ( 1 / sum_k exp(-alpha * dists[j, k]) ) * sum_k exp(-alpha * dists[j, k]) * (-alpha) * d(dists[j, k])/dx
    #           = ( sum_k exp(-alpha * dists[j, k]) * d(dists[j, k])/dx ) / ( sum_k exp(-alpha * dists[j, k]) )
    #           = sum_k v_{jk} * d(dists[j, k])/dx
    # where v_{jk} = exp(-alpha * dists[j, k]) / sum_m exp(-alpha * dists[j, m])
    
    # We have exp_neg (shifted) and sum_exp (shifted) from above.
    # v_{jk} = exp_neg[j, k] / sum_exp[j]
    v_jk = exp_neg / sum_exp # Shape (M, N)
    
    # Now d(dists[j, k])/x_i
    # dists[j, k] = arccos( dots[j, k] )
    # d(arccos(u))/dx = -1/sqrt(1-u^2) * d(u)/dx
    # dots[j, k] = sum_m x_{k,m} * samples_{j,m}
    # d(dots[j, k])/x_{i, n} = delta_{ki} * samples_{j, n}
    
    # So d(dists[j, k])/x_{i, n} = - delta_{ki} * samples_{j, n} / sqrt(1 - dots[j, k]^2)
    
    # Gradient wrt x_i is sum_j weights_j * sum_k v_{jk} * ( - delta_{ki} * samples_{j, n} / sqrt(1 - dots[j, k]^2) )
    # Gradient wrt x_{i, n} = - sum_j weights_j * v_{j, i} * samples_{j, n} / sqrt(1 - dots[j, i]^2)
    
    # Let's compute this efficiently.
    # Term for x_i:
    # grad_i = - sum_j ( weights_j * v_{j, i} / sqrt(1 - dots[j, i]^2) ) * samples_j
    
    # Compute denominator
    sin_dists = np.sqrt(np.maximum(0, 1.0 - dots * dots)) # Shape (M, N)
    # Avoid division by zero
    safe_sin = np.where(sin_dists < 1e-10, 1e-10, sin_dists)
    
    # Coefficients for each (j, k)
    # coeff[j, k] = weights_j * v_{j, k} / safe_sin[j, k]
    coeff = (weights_j[:, np.newaxis] * v_jk) / safe_sin # Shape (M, N)
    
    # grad_x[k] = - sum_j coeff[j, k] * samples[j]
    # grad_x = - (coeff.T @ samples)
    grad_x = -(coeff.T @ samples) # Shape (N, 3)
    
    # Project gradient to tangent space of sphere
    # grad_tangent = grad_x - (grad_x . x) * x
    grad_flat = np.zeros_like(x_flat)
    for i in range(N):
        grad_i = grad_x[i]
        proj = np.dot(grad_i, x[i])
        grad_tangent = grad_i - proj * x[i]
        grad_flat[i*3:(i+1)*3] = grad_tangent
        
    return obj, grad_flat

def _optimize_from_start(start_pts, samples, time_limit, start_time):
    """Run L-BFGS-B optimization from a starting configuration."""
    best_cr = _covering_radius_exact(start_pts)
    best_pts = start_pts.copy()
    
    # Schedule for alpha and beta
    # Start with smooth (low alpha, beta) and increase
    schedule = [
        (5.0, 5.0),
        (10.0, 10.0),
        (20.0, 20.0),
        (50.0, 50.0),
        (100.0, 100.0),
        (200.0, 200.0)
    ]
    
    current_pts = start_pts.copy()
    
    for alpha, beta in schedule:
        if time.time() - start_time > time_limit:
            break
            
        x0 = current_pts.flatten()
        
        try:
            res = minimize(
                _soft_objective_and_grad,
                x0,
                args=(samples, alpha, beta),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-7}
            )
            if res.success or res.status in [0, 1, 2, 3, 4, 5, 6]:
                new_pts = _normalize(res.x.reshape(N, 3))
                cr = _covering_radius_exact(new_pts)
                if cr < best_cr:
                    best_cr = cr
                    best_pts = new_pts.copy()
                # Use the improved points for the next stage
                if cr <= best_cr + 1e-6: # Allow slight increase if it's a good basin? No, stick to best.
                    current_pts = new_pts
                else:
                    current_pts = best_pts
        except Exception as e:
            # print(f"Optimization failed: {e}")
            continue
            
    return best_pts, best_cr

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _normalize(pts)

def _icosahedron_based_points(n):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    base = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    base = _normalize(base)
    
    pts = base.copy()
    rng = np.random.RandomState(123)
    while len(pts) < n:
        i, j = rng.randint(0, len(pts), 2)
        if i == j:
            continue
        mid = (pts[i] + pts[j]) / 2.0
        mid = _normalize(mid)
        # Check if it's far enough from existing points
        dots = np.sum(mid * pts, axis=1)
        if np.max(dots) < 0.95: # ~18 degrees
            pts = np.vstack([pts, mid])
    return pts[:n]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Generate samples for surrogate
    n_samples = 5000
    samples = _generate_samples(n_samples)
    
    initial_configs = [
        _fibonacci_points(N),
        _random_points(N, seed=1),
        _random_points(N, seed=2),
        _random_points(N, seed=3),
        _icosahedron_based_points(N),
    ]
    
    best_points = initial_configs[0].copy()
    best_cr = _covering_radius_exact(best_points)
    
    for i, init_pts in enumerate(initial_configs):
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        # Allocate time per start
        time_per_start = min(remaining / (len(initial_configs) - i), 25.0)
        
        pts, cr = _optimize_from_start(init_pts, samples, time_per_start, start_time)
        
        if cr < best_cr:
            best_cr = cr
            best_points = pts.copy()
            
    # Final polish with high precision
    remaining = deadline - time.time()
    if remaining > 10:
        # Use a smaller number of samples for faster gradient? No, keep same.
        # Just run one more high-beta optimization
        high_beta_schedule = [
            (500.0, 500.0),
            (1000.0, 1000.0)
        ]
        # Re-implement small loop for final polish
        current_pts = best_points.copy()
        for alpha, beta in high_beta_schedule:
            if time.time() - start_time > deadline - 2:
                break
            try:
                x0 = current_pts.flatten()
                res = minimize(
                    _soft_objective_and_grad,
                    x0,
                    args=(samples, alpha, beta),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.success or res.status in [0, 1, 2, 3, 4, 5, 6]:
                    new_pts = _normalize(res.x.reshape(N, 3))
                    cr = _covering_radius_exact(new_pts)
                    if cr < best_cr:
                        best_cr = cr
                        best_points = new_pts.copy()
                        current_pts = new_pts
            except Exception:
                pass
                
    best_points = _normalize(best_points)
    return best_points
