import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius(points):
    """Compute covering radius in degrees using convex hull facets."""
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            if len(simplex) == 3:
                a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
                # Compute circumradius of spherical triangle
                # Using the formula: R = arcsin( (sin(a/2)sin(b/2)sin(c/2)) / (2 sin(E/2)) ) is not quite right for all cases
                # Better: find circumcenter.
                cross1 = np.cross(a - b, b - c)
                if np.linalg.norm(cross1) < 1e-15:
                    r_facet = 0.0
                else:
                    x = cross1 / np.linalg.norm(cross1)
                    d_xa = np.arccos(np.clip(np.dot(x, a), -1, 1))
                    d_xb = np.arccos(np.clip(np.dot(x, b), -1, 1))
                    d_xc = np.arccos(np.clip(np.dot(x, c), -1, 1))
                    d_negxa = np.arccos(np.clip(-np.dot(x, a), -1, 1))
                    d_negxb = np.arccos(np.clip(-np.dot(x, b), -1, 1))
                    d_negxc = np.arccos(np.clip(-np.dot(x, c), -1, 1))
                    r1 = max(d_xa, d_xb, d_xc)
                    r2 = max(d_negxa, d_negxb, d_negxc)
                    r_circum = min(r1, r2)
                    # Check if circumcenter is inside
                    # For now, take min of circumradius and max half-side
                    d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
                    d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
                    d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
                    r_max_side = max(d_ab, d_bc, d_ca) / 2.0
                    r_facet = min(r_circum, r_max_side)
                max_r = max(max_r, r_facet)
        return np.degrees(max_r)
    except Exception:
        return 180.0

def _smooth_max_loss_and_grad(points, samples, beta, gamma):
    """
    Compute smooth max loss and gradient.
    Loss: (1/beta) log sum_k exp(beta * min_i arccos(p_i . s_k))
    Approx min_i by soft-min: -1/gamma log sum_i exp(-gamma arccos(p_i . s_k))
    """
    # points: (N, 3)
    # samples: (M, 3)
    
    # Compute distances: (M, N)
    dots = np.clip(samples @ points.T, -1, 1)
    dists = np.arccos(dots)
    
    # Soft min for each sample
    # shape (M, N)
    scaled_min = -dists / gamma
    max_scaled_min = np.max(scaled_min, axis=1, keepdims=True)
    exp_scaled_min = np.exp(scaled_min - max_scaled_min)
    sum_exp_scaled_min = np.sum(exp_scaled_min, axis=1, keepdims=True)
    
    # soft_min_dist = -gamma * (max_scaled_min + log(sum_exp_scaled_min))
    # Wait: log(sum exp(x)) = max(x) + log(sum exp(x - max(x)))
    # soft_min = -1/gamma * log(sum exp(-gamma d))
    # = -1/gamma * ( -gamma max(d) + log(sum exp(-gamma(d - max(d)))) )
    # = max(d) - 1/gamma log(sum exp(-gamma(d - max(d))))
    
    # Let's just use the standard log-sum-exp for stability
    # soft_min_d[k] = -1/gamma * LSE(-gamma * dists[k])
    
    # We need gradient of soft_min w.r.t points
    # d(soft_min_k)/dp_i = (1/gamma) * [ exp(-gamma d_ki) / sum_j exp(-gamma d_kj) ] * d(d_ki)/dp_i
    # d(d_ki)/dp_i = -s_k / sqrt(1 - (p_i.s_k)^2)
    
    # Probabilities for soft min
    # P[k, i] = exp(-gamma d_ki) / sum_j exp(-gamma d_kj)
    # We already have exp_scaled_min = exp(-gamma d_ki - (-gamma max d_k)) = exp(-gamma (d_ki - max d_k))
    # sum_exp_scaled_min is the denominator for the scaled version.
    # P[k, i] = exp_scaled_min[k, i] / sum_exp_scaled_min[k, 0]
    
    probs = exp_scaled_min / sum_exp_scaled_min # (M, N)
    
    # Soft min distances
    # soft_min_d[k] = -1/gamma * ( -gamma max_d_k + log(sum_exp_scaled_min[k]) )
    # = max_d_k - 1/gamma * log(sum_exp_scaled_min[k])
    max_dists = np.max(dists, axis=1) # (M,)
    soft_min_dists = max_dists - (1.0 / gamma) * np.log(sum_exp_scaled_min[:, 0])
    
    # Now smooth max over samples
    # Loss = 1/beta * log sum_k exp(beta * soft_min_d_k)
    scaled_max = soft_min_dists / beta
    max_scaled_max = np.max(scaled_max)
    exp_scaled_max = np.exp(scaled_max - max_scaled_max)
    sum_exp_scaled_max = np.sum(exp_scaled_max)
    
    loss = beta * (max_scaled_max + np.log(sum_exp_scaled_max))
    
    # Gradient of Loss w.r.t soft_min_dists
    # d Loss / d soft_min_d_k = (1/beta) * beta * exp(beta soft_min_d_k) / sum ...
    # = exp_scaled_max[k] / sum_exp_scaled_max
    weights_k = exp_scaled_max / sum_exp_scaled_max # (M,)
    
    # Gradient w.r.t points
    # d Loss / d p_i = sum_k weights_k * d soft_min_d_k / d p_i
    # d soft_min_d_k / d p_i = P[k, i] * d dist_ki / d p_i
    # d dist_ki / d p_i = -s_k / sqrt(1 - (p_i . s_k)^2)
    
    # Compute denom for dist gradient
    # denom[k, i] = sqrt(1 - dots[k, i]^2)
    denom = np.sqrt(np.maximum(1.0 - dots * dots, 1e-15)) # (M, N)
    
    # grad_p[i] = sum_k weights_k * P[k, i] * (-s_k / denom[k, i])
    # Let's vectorize this.
    # coeff[k, i] = weights_k[k] * probs[k, i] / denom[k, i]
    # grad_p[i] = - sum_k coeff[k, i] * s_k
    
    # coeff shape (M, N)
    coeff = (weights_k[:, np.newaxis] * probs) / denom
    
    # grad_p = - (coeff.T @ samples)
    grad_p = - (coeff.T @ samples)
    
    return loss, grad_p

def _exp_map(p_ref, v):
    """
    p_ref: (N, 3)
    v: (N, 3)
    Returns points on sphere.
    """
    v_norms = np.linalg.norm(v, axis=1, keepdims=True)
    # Handle v=0
    safe_norms = np.maximum(v_norms, 1e-15)
    
    # p = cos(||v||) p_ref + sin(||v||)/||v|| v
    # If v is small, sin(||v||)/||v|| ~ 1, cos(||v||) ~ 1
    # p ~ p_ref + v
    
    cos_v = np.cos(v_norms)
    sin_v_over_norm = np.sin(v_norms) / safe_norms
    
    # For v=0, sin_v_over_norm is 0/0 -> 0, but should be 1.
    # We can use np.where
    sin_v_over_norm = np.where(v_norms < 1e-8, 1.0, sin_v_over_norm)
    cos_v = np.where(v_norms < 1e-8, 1.0, cos_v)
    
    p = cos_v * p_ref + sin_v_over_norm * v
    return _normalize(p)

def _opt_step(p_ref, samples, beta, gamma, max_iters=50):
    """
    Optimize v such that p = exp(p_ref, v) minimizes loss.
    """
    def func_and_grad(v_flat):
        v = v_flat.reshape(N, 3)
        p = _exp_map(p_ref, v)
        loss, grad_p = _smooth_max_loss_and_grad(p, samples, beta, gamma)
        
        # Compute grad_v
        # d p_i / d v_i
        # J_i = cos(||v_i||) u_i u_i^T + sin(||v_i||)/||v_i|| (I - u_i u_i^T)
        # grad_v_i = J_i^T grad_p_i = J_i grad_p_i (since J is symmetric)
        
        v_norms = np.linalg.norm(v, axis=1, keepdims=True)
        safe_norms = np.maximum(v_norms, 1e-15)
        
        cos_v = np.cos(v_norms)
        sin_v_over_norm = np.sin(v_norms) / safe_norms
        sin_v_over_norm = np.where(v_norms < 1e-8, 1.0, sin_v_over_norm)
        cos_v = np.where(v_norms < 1e-8, 1.0, cos_v)
        
        # u = v / ||v||
        u = v / safe_norms
        u = np.where(v_norms < 1e-8, 0.0, u)
        
        # J = cos_v * (u u^T) + sin_v_over_norm * (I - u u^T)
        # J = (cos_v - sin_v_over_norm) u u^T + sin_v_over_norm I
        
        # grad_v_i = sin_v_over_norm grad_p_i + (cos_v - sin_v_over_norm) (u_i . grad_p_i) u_i
        
        # Compute u . grad_p
        u_dot_gp = np.sum(u * grad_p, axis=1, keepdims=True)
        
        # Coefficients
        c1 = sin_v_over_norm
        c2 = cos_v - sin_v_over_norm
        
        grad_v = c1 * grad_p + c2 * (u_dot_gp * u)
        
        return loss, grad_v.flatten()
    
    # Initial v = 0
    v0 = np.zeros(N * 3)
    
    res = minimize(func_and_grad, v0, method='L-BFGS-B', jac=True, options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8})
    
    v_opt = res.x.reshape(N, 3)
    p_opt = _exp_map(p_ref, v_opt)
    
    return p_opt

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Samples for smooth max
    rng = np.random.RandomState(42)
    n_samples = 1000
    theta = rng.uniform(0, np.pi, n_samples)
    phi_s = rng.uniform(0, 2 * np.pi, n_samples)
    sx = np.sin(theta) * np.cos(phi_s)
    sy = np.sin(theta) * np.sin(phi_s)
    sz = np.cos(theta)
    samples = np.column_stack([sx, sy, sz])
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            current = points.copy()
        else:
            # Perturb
            rng_r = np.random.RandomState(restart * 100)
            noise = rng_r.randn(N, 3) * 0.1
            current = points + noise
            current = _normalize(current)
        
        # Iterative retraction
        p_curr = current.copy()
        for it in range(10):
            if time.time() > deadline - 10.0:
                break
            
            # Update beta and gamma
            beta = 20.0 + it * 5.0
            gamma = 50.0 + it * 10.0
            
            p_next = _opt_step(p_curr, samples, beta, gamma, max_iters=20)
            p_next = _normalize(p_next)
            
            # Check improvement
            # We don't want to evaluate covering radius (expensive) at every sub-step
            # Just accept if loss decreased? Or just take the last one.
            p_curr = p_next
        
        radius = _covering_radius(p_curr)
        if radius < best_radius:
            best_radius = radius
            best_points = p_curr.copy()
    
    return best_points
