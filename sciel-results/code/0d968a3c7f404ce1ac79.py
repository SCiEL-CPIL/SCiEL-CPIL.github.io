"""
Strategy: Direct covering radius minimization using smooth LogSumExp approximation.
Instead of maximizing minimum pairwise distance (Tammes problem) which is a proxy,
we directly minimize the maximum angular distance to the nearest point.
We approximate the "max over y of min over p" using:
1. A dense set of sample points on the sphere to represent the continuous domain.
2. A smooth approximation of the min function: min_i f_i ≈ -logsumexp(-beta * f_i) / beta.
3. Gradient descent with adaptive beta (annealing) to navigate the landscape.
"""

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 145.0
_N_SAMPLES = 20000  # Number of sample points on the sphere

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

def _generate_fibonacci_samples(n_samples, rng):
    """
    Generate well-distributed points on the sphere using a Fibonacci spiral.
    This provides better coverage for the sample set than pure random.
    """
    idx = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * idx / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * idx
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Add slight random perturbation to break symmetry and avoid degeneracies
    noise = rng.normal(scale=0.01, size=(n_samples, 3))
    pts = pts + noise
    return _normalize(pts)

def _covering_radius_smooth(pts, samples, beta):
    """
    Compute the smooth approximation of the covering radius and its gradient w.r.t pts.
    
    The covering radius is R = max_y min_i arccos(y . p_i).
    Let d_i(y) = arccos(y . p_i).
    min_i d_i(y) ≈ -1/beta * log( sum_i exp(-beta * d_i(y)) )  (for large beta)
    
    We want to minimize R_smooth = max_y ( -1/beta * log( sum_i exp(-beta * d_i(y)) ) )
    
    To handle the outer max efficiently, we can just evaluate it on the samples.
    However, for gradient descent, we need the gradient of the objective w.r.t pts.
    
    Let's define the objective as the mean of the soft-min distances raised to a power, 
    or simply the max soft-min distance. Using the max is non-smooth.
    
    Alternative: Minimize the "soft" covering radius defined as:
    R_soft = (1/n) * sum_y [ -1/beta * log( sum_i exp(-beta * d_i(y)) ) ]^p
    For p=1, this is the average soft-min distance. Minimizing this tends to minimize the max.
    To be more aggressive towards the max, we can use a higher power p, or just rely on 
    the fact that the worst points will dominate if we use a max-like operation.
    
    Let's use the following objective:
    F = max_y [ -1/beta * log( sum_i exp(-beta * d_i(y)) ) ]
    
    To make it differentiable, we can approximate the outer max with a LogSumExp as well, 
    but that's computationally heavy. Instead, we can use gradient descent on the 
    "worst-case" samples.
    
    Actually, a simpler and effective approach for the gradient:
    The gradient of d_i(y) = arccos(y . p_i) w.r.t p_i is:
    grad_{p_i} d_i(y) = - y / sqrt(1 - (y . p_i)^2)
    
    Let w_i(y) = exp(-beta * d_i(y)) / sum_j exp(-beta * d_j(y))
    Then grad_{p_i} (soft_min) = -1/beta * sum_y w_i(y) * grad_{p_i} d_i(y)
    
    So grad_{p_i} F ≈ -1/beta * sum_{y in S} w_i(y) * (- y / sqrt(1 - (y . p_i)^2))
    = 1/beta * sum_{y in S} w_i(y) * y / sqrt(1 - (y . p_i)^2)
    
    We want to minimize F, so we move p_i in the direction of -grad.
    """
    pts = _normalize(pts)
    
    # Compute dot products: samples (M x 3) @ pts.T (3 x N) -> M x N
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
    
    # Distances
    dists = np.arccos(dots)  # M x N
    
    # Soft min weights: w_i(y) = exp(-beta * d_i) / sum_j exp(-beta * d_j)
    exp_neg_beta_d = np.exp(-beta * dists)
    # Handle numerical underflow/overflow by subtracting max in each row
    row_max = np.max(exp_neg_beta_d, axis=1, keepdims=True)
    # If row_max is 0, all exps underflowed, which shouldn't happen with reasonable beta
    # But for safety:
    row_max[row_max == 0] = 1.0
    exp_neg_beta_d = exp_neg_beta_d / row_max
    sum_exp = np.sum(exp_neg_beta_d, axis=1, keepdims=True)
    sum_exp[sum_exp == 0] = 1.0
    weights = exp_neg_beta_d / sum_exp  # M x N
    
    # Soft min value for each sample
    soft_min_vals = -1.0 / beta * np.log(sum_exp * row_max) # Wait, logsumexp trick
    # Correct logsumexp: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    # Here x = -beta * dists
    x = -beta * dists
    max_x = np.max(x, axis=1, keepdims=True)
    log_sum_exp = max_x + np.log(np.sum(np.exp(x - max_x), axis=1, keepdims=True))
    soft_min_vals = -1.0 / beta * log_sum_exp.flatten()  # M
    
    # Objective: We want to minimize the max of soft_min_vals.
    # To make it smooth, we can use a high-order max approximation or just the mean of top-k.
    # Let's use the mean of the top 10% of samples as a proxy for the max.
    # Or simply the max itself, but use subgradients? No, let's use a smooth max.
    # Smooth max: 1/gamma * log(sum(exp(gamma * v_i)))
    gamma = 20.0
    v = soft_min_vals
    max_v = np.max(v)
    smooth_max = max_v + 1.0/gamma * np.log(np.sum(np.exp(gamma * (v - max_v))))
    
    # Gradient w.r.t pts
    # d(smooth_max)/d(v_y) = exp(gamma * (v_y - max_v)) / sum(exp(gamma * (v_y' - max_v)))
    exp_gv = np.exp(gamma * (v - max_v))
    sum_exp_gv = np.sum(exp_gv)
    if sum_exp_gv == 0:
        sum_exp_gv = 1.0
    dv_dv_y = exp_gv / sum_exp_gv  # M
    
    # d(v_y)/d(p_i) = 1/beta * sum_z weights[z, i] * (samples[z] / sqrt(1 - dots[z, i]^2))
    # Note: weights[z, i] is the soft-min weight for point i at sample z.
    # sqrt(1 - dots^2) is sin(d)
    sin_d = np.sqrt(1.0 - dots * dots)
    sin_d[sin_d < 1e-12] = 1e-12
    
    # Term: weights[z, i] * samples[z] / sin_d[z, i]
    # We need sum_z dv_dv_y[z] * weights[z, i] * samples[z] / sin_d[z, i]
    
    # Efficient computation:
    # coeff[z, i] = dv_dv_y[z] * weights[z, i] / sin_d[z, i]
    coeff = dv_dv_y[:, None] * weights / sin_d  # M x N
    
    # grad_pt[i] = 1/beta * sum_z coeff[z, i] * samples[z]
    # This is (coeff.T @ samples) * (1/beta)
    grad_pts = (coeff.T @ samples) * (1.0 / beta)  # N x 3
    
    # Project gradient to tangent space
    # grad_tan[i] = grad_pts[i] - (grad_pts[i] . pts[i]) pts[i]
    dots_grad = np.sum(grad_pts * pts, axis=1, keepdims=True)
    grad_tan = grad_pts - dots_grad * pts
    
    return smooth_max, grad_tan

def construct():
    start_time = time.time()
    
    # Generate sample points
    rng = np.random.default_rng(42)
    samples = _generate_fibonacci_samples(_N_SAMPLES, rng)
    
    # Initial configuration: Fibonacci spiral
    idx = np.arange(N) + 0.5
    z = 1.0 - 2.0 * idx / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * idx
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_radius = np.inf
    
    # Calculate true covering radius for tracking
    def _true_covering_radius(pts, samples):
        dots = samples @ pts.T
        max_dots = dots.max(axis=1)
        min_max_dot = np.min(max_dots)
        return np.degrees(np.arccos(np.clip(min_max_dot, -1.0, 1.0)))
    
    best_radius = _true_covering_radius(pts, samples)
    
    # Annealing schedule for beta
    # Start with low beta (smooth landscape) and increase to high beta (sharp, approximates true min)
    beta_start = 5.0
    beta_end = 100.0
    n_steps = 3000
    
    # Adaptive learning rate
    lr = 0.05
    momentum = 0.9
    
    v = np.zeros_like(pts)
    
    for step in range(n_steps):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 10:
            break
        
        # Anneal beta
        progress = step / n_steps
        # Linear annealing
        beta = beta_start + (beta_end - beta_start) * progress
        
        # Compute objective and gradient
        obj, grad = _covering_radius_smooth(pts, samples, beta)
        
        # Update with momentum
        v = momentum * v - lr * grad
        new_pts = pts + v
        new_pts = _normalize(new_pts)
        
        # Check improvement
        new_radius = _true_covering_radius(new_pts, samples)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            # Increase LR slightly if improving
            lr = min(0.1, lr * 1.01)
        else:
            # Decrease LR
            lr = max(0.005, lr * 0.95)
            # Revert
            pts = best_pts.copy()
            v = 0.5 * v  # Dampen momentum
        
        # Periodic random perturbation to escape local minima
        if step % 500 == 499:
            noise = rng.normal(scale=0.05, size=(N, 3))
            perturbed = _normalize(pts + noise)
            pert_radius = _true_covering_radius(perturbed, samples)
            if pert_radius < best_radius:
                best_radius = pert_radius
                best_pts = perturbed.copy()
                pts = perturbed.copy()
    
    # Final polish with higher beta and smaller steps
    remaining_time = _DEADLINE - (time.time() - start_time)
    if remaining_time > 10:
        n_polish = int(remaining_time * 20) # ~20 steps per second
        beta_final = 200.0
        lr_polish = 0.01
        v = np.zeros_like(pts)
        
        for step in range(n_polish):
            if time.time() - start_time > _DEADLINE - 5:
                break
                
            obj, grad = _covering_radius_smooth(pts, samples, beta_final)
            v = 0.9 * v - lr_polish * grad
            new_pts = _normalize(pts + v)
            
            new_radius = _true_covering_radius(new_pts, samples)
            if new_radius < best_radius:
                best_radius = new_radius
                best_pts = new_pts.copy()
                pts = new_pts.copy()
            else:
                lr_polish = max(0.001, lr_polish * 0.9)
                pts = best_pts.copy()
                v = 0.5 * v

    return _normalize(best_pts)
# EVOLVE-BLOCK-END
