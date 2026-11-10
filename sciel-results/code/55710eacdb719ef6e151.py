"""Rewritten program for 130-point sphere covering; uses log-sum-exp gradient descent."""
import numpy as np
import time

N = 130
SEED = 42
np.random.seed(SEED)


def _initial_points():
    """Fibonacci spiral initialization."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


def _covering_radius(pts):
    """Compute the covering radius (max angular distance to nearest point).
    
    Uses a dense sampling grid on the sphere.
    """
    # Sample points on sphere using a fine grid
    n_theta = 180
    n_phi = 360
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    sample_pts = np.stack([
        np.sin(TH) * np.cos(PH),
        np.sin(TH) * np.sin(PH),
        np.cos(TH)
    ], axis=-1).reshape(-1, 3)
    
    # For each sample point, find nearest configuration point
    # Use batched computation
    batch_size = 5000
    max_dist = 0.0
    for start in range(0, len(sample_pts), batch_size):
        batch = sample_pts[start:start + batch_size]
        # Compute distances to all N points
        # dot products
        dots = batch @ pts.T  # (batch, N)
        dots = np.clip(dots, -1.0, 1.0)
        # angular distance = arccos(dot)
        min_dot = np.max(dots, axis=1)  # max dot = min angular distance
        max_ang = np.arccos(min_dot)
        max_dist = max(max_dist, np.max(max_ang))
    return np.degrees(max_dist)


def _logsumexp_covering(pts, beta=50.0):
    """Smooth approximation of covering radius using log-sum-exp.
    
    For each sample point y, the distance to nearest point is -log(sum exp(-beta*d_i)).
    We maximize this over a set of sample points, but here we use a different approach:
    We want to minimize the max over y of min_i d(y, x_i).
    
    Alternative: use the fact that the covering radius equals the max circumradius of facets.
    For gradient-based optimization, we use a soft version.
    
    Here I'll use: for a set of test points, compute soft-min of distances, then take max.
    """
    # Use a moderate number of test points for gradient computation
    n_theta = 90
    n_phi = 180
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    sample_pts = np.stack([
        np.sin(TH) * np.cos(PH),
        np.sin(TH) * np.sin(PH),
        np.cos(TH)
    ], axis=-1).reshape(-1, 3)
    
    # Compute dot products: (M, N)
    dots = sample_pts @ pts.T
    dots = np.clip(dots, -1.0 + 1e-7, 1.0 - 1e-7)
    
    # Angular distances
    dists = np.arccos(dots)  # (M, N)
    
    # Soft min: -log(sum exp(-beta * dists)) / beta
    # For numerical stability
    shifted = -beta * dists
    max_shifted = np.max(shifted, axis=1, keepdims=True)
    log_sum = max_shifted.squeeze() + np.log(np.sum(np.exp(shifted - max_shifted), axis=1))
    soft_min = -log_sum / beta  # (M,)
    
    # The covering radius is approximated by the max of soft_min over test points
    # For gradient descent, we want to minimize this max, so we use a soft-max over test points
    # Or simply use the mean as a proxy, or the max with a soft version
    
    # Use soft max of soft_min values
    beta2 = 20.0
    shifted2 = beta2 * soft_min
    max_shifted2 = np.max(shifted2)
    log_sum2 = max_shifted2 + np.log(np.sum(np.exp(shifted2 - max_shifted2)))
    soft_max = log_sum2 / beta2
    
    return soft_max


def _gradient_covering(pts, beta=50.0, beta2=20.0):
    """Compute gradient of the soft covering radius w.r.t. points.
    
    Returns gradient array of shape (N, 3).
    """
    n_theta = 90
    n_phi = 180
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    sample_pts = np.stack([
        np.sin(TH) * np.cos(PH),
        np.sin(TH) * np.sin(PH),
        np.cos(TH)
    ], axis=-1).reshape(-1, 3)
    
    # dots: (M, N)
    dots = sample_pts @ pts.T
    dots_clipped = np.clip(dots, -1.0 + 1e-7, 1.0 - 1e-7)
    
    # dists: (M, N)
    dists = np.arccos(dots_clipped)
    
    # Soft min weights: w_ij = exp(-beta * d_ij) / sum_k exp(-beta * d_ik)
    neg_beta_dists = -beta * dists
    max_nb = np.max(neg_beta_dists, axis=1, keepdims=True)
    exp_nb = np.exp(neg_beta_dists - max_nb)
    sum_exp = np.sum(exp_nb, axis=1, keepdims=True)
    weights = exp_nb / sum_exp  # (M, N)
    
    # d(soft_min)/d(d_ij) = -w_ij / beta
    # d(d_ij)/d(x_j) where d_ij = arccos(y_i . x_j)
    # = -(y_i - (y_i . x_j) x_j) / sqrt(1 - (y_i . x_j)^2)
    #   = -(y_i - dots_ij * x_j) / sin(d_ij)
    
    # sin(d_ij) = sqrt(1 - cos^2(d_ij)) = sqrt(1 - dots^2)
    sin_dists = np.sqrt(np.maximum(1.0 - dots_clipped ** 2, 1e-12))  # (M, N)
    
    # For each sample point i and config point j:
    # grad_{x_j} d_ij = -(y_i - dots_ij * x_j) / sin_dists_ij
    
    # We need: grad_{x_j} soft_max_i soft_min_j
    # = sum_i (d soft_max_i / d soft_min_i) * (d soft_min_i / d d_ij) * (d d_ij / d x_j)
    
    # soft_max_i soft_min_i = logsumexp(beta2 * soft_min) / beta2
    # d soft_max / d soft_min_i = beta2 * exp(beta2 * soft_min_i) / sum_k exp(beta2 * soft_min_k)
    
    soft_min = -(-np.log(np.sum(exp_nb, axis=1))) / beta  # same as before
    # Actually: soft_min_i = -(max_nb_i + log(sum exp)) / beta = -log(sum exp exp_nb) / beta + max_nb/beta... 
    # Let me recompute: soft_min_i = -(max_nb_i + log(sum_k exp(neg_beta_dists_ik - max_nb_i))) / beta
    # But we already have weights, so let's use the formula directly.
    
    # Recompute soft_min properly
    log_sum_exp = max_nb.squeeze() + np.log(np.sum(exp_nb, axis=1))
    soft_min = -log_sum_exp / beta  # (M,)
    
    # Soft max weights
    beta2_sm = beta2 * soft_min
    max_b2 = np.max(beta2_sm)
    exp_b2 = np.exp(beta2_sm - max_b2)
    sum_exp_b2 = np.sum(exp_b2)
    sm_weights = exp_b2 / sum_exp_b2 * beta2  # (M,)
    
    # Now gradient for each config point j:
    # grad_{x_j} = sum_i sm_weights_i * (-w_ij / beta) * (-(y_i - dots_ij * x_j) / sin_dists_ij)
    #            = sum_i sm_weights_i * w_ij / beta * (y_i - dots_ij * x_j) / sin_dists_ij
    
    # This is: (1/beta) * sum_i sm_weights_i * w_ij * (y_i - dots_ij * x_j) / sin_dists_ij
    #         = (1/beta) * [ sum_i (sm_weights_i * w_ij / sin_dists_ij) * y_i 
    #                       - sum_i (sm_weights_i * w_ij * dots_ij / sin_dists_ij) * x_j ]
    
    # Let a_ij = sm_weights_i * w_ij / sin_dists_ij
    # Let b_ij = sm_weights_i * w_ij * dots_ij / sin_dists_ij
    
    a = sm_weights[:, None] * weights / sin_dists  # (M, N)
    b = sm_weights[:, None] * weights * dots_clipped / sin_dists  # (M, N)
    
    # sum_i a_ij * y_i for each j
    term1 = a.T @ sample_pts  # (N, 3)
    # sum_i b_ij for each j
    term2 = b.sum(axis=0)  # (N,)
    
    grad = (term1 - term2[:, None] * pts) / beta
    
    # Project gradient onto tangent space (since points must stay on sphere)
    # The constraint is ||x_j|| = 1, so gradient should be projected: 
    # grad_tangent = grad - (grad . x_j) x_j
    grad_dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tangent = grad - grad_dot * pts
    
    return grad_tangent


def construct():
    """Construct 130 points on unit sphere minimizing covering radius."""
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s
    
    # Initialize
    pts = _initial_points()
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Gradient descent with adaptive step size
    lr = 0.01
    beta = 50.0
    beta2 = 20.0
    
    # Adaptive beta: increase over time for sharper approximation
    n_steps = 0
    min_lr = 1e-5
    
    while time.time() < deadline:
        step_start = time.time()
        
        # Compute gradient
        try:
            grad = _gradient_covering(pts, beta=beta, beta2=beta2)
        except Exception:
            break
        
        # Check for NaN
        if not np.all(np.isfinite(grad)):
            lr *= 0.5
            continue
        
        # Normalize gradient for consistent step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        grad_normalized = grad / grad_norm
        
        # Take step
        new_pts = pts + lr * grad_normalized
        
        # Project back to sphere
        new_pts = _normalize(new_pts)
        
        # Evaluate covering radius (use a cheaper estimate for speed)
        # Use the soft covering radius for quick check
        soft_cr = _logsumexp_covering(new_pts, beta=beta, beta2=beta2)
        
        # Occasionally compute true covering radius
        if n_steps % 5 == 0:
            true_cr = _covering_radius(new_pts)
            if true_cr < best_cr:
                best_cr = true_cr
                best_pts = new_pts.copy()
        
        # Adaptive learning rate
        if n_steps % 10 == 0 and n_steps > 0:
            # Simple adaptive: if improving, keep lr; if stuck, reduce
            pass
        
        pts = new_pts
        n_steps += 1
        
        # Gradually increase beta for sharper optimization
        if n_steps % 50 == 0:
            beta = min(beta * 1.2, 200.0)
            beta2 = min(beta2 * 1.1, 100.0)
            lr = max(lr * 0.95, min_lr)
        
        # Check time
        if time.time() - step_start > 5.0:
            # Each step is too slow, reduce sample density or break
            break
    
    # Final evaluation
    final_cr = _covering_radius(pts)
    if final_cr < best_cr:
        best_cr = final_cr
        best_pts = pts.copy()
    
    return _normalize(best_pts)
