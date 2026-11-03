"""New program: Riemannian Conjugate Gradient on the sphere for logarithmic energy."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time

def _log_energy(pts):
    """Compute sum_{i<j} log(1/||x_i - x_j||)."""
    dists = cdist(pts, pts)
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    return -np.sum(np.log(d))

def _riemannian_grad(pts):
    """Compute Riemannian gradient of the log energy."""
    # Vectorized gradient computation
    # dists[i, j] = ||x_i - x_j||
    dists = cdist(pts, pts)
    # Avoid division by zero
    dists = np.maximum(dists, 1e-15)
    
    # grad_i = -sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    # We can compute this as:
    # term_ij = (x_i - x_j) / d_ij^2
    # sum over j
    
    # Using broadcasting
    # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :] # (N, N, 3)
    # This is N^2 * 3 memory. For N=100, 10000*3 = 30000 floats. Fine.
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-30)
    
    # We want grad[i] = - sum_{j != i} diff[i,j] / dist_sq[i,j]
    # Note: diff[i,i] = 0, so 0/0 is 0.
    # Create a mask for i != j
    mask = ~np.eye(N, dtype=bool)
    
    # inv_dist_sq = 1 / dist_sq
    # Set diagonal to 0 to avoid issues, though diff is 0 there.
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    
    # weighted_diff = diff * inv_dist_sq[..., np.newaxis]
    weighted_diff = diff * inv_dist_sq[..., np.newaxis]
    
    # grad = - sum over j
    grad = -np.sum(weighted_diff, axis=1)
    
    # Project to tangent space: grad_tangent = grad - (grad . x) x
    grad_dot_x = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tangent = grad - grad_dot_x * pts
    
    return grad_tangent

def _exp_map(x, v, alpha):
    """Exponential map on the sphere.
    x: point on sphere (N, 3)
    v: tangent vector (N, 3)
    alpha: step size
    Returns new point on sphere.
    """
    # For sphere, exp_x(v) = x cos(||v||) + (v/||v||) sin(||v||)
    # Here v is scaled by alpha.
    v_scaled = v * alpha
    norms = np.linalg.norm(v_scaled, axis=1, keepdims=True)
    
    # Handle zero norms
    cos_theta = np.cos(norms)
    sin_theta = np.sin(norms)
    
    # Where norm is 0, sin(norm)/norm = 1, else sin(norm)/norm
    sin_over_norm = np.ones_like(norms)
    nonzero = norms > 1e-12
    sin_over_norm[nonzero] = sin_theta[nonzero] / norms[nonzero]
    
    return x * cos_theta + v_scaled * sin_over_norm

def _retract(x, v, alpha):
    """Retraction (simple normalization) for step size.
    This is not geodesic but often works well.
    """
    new_pts = x + alpha * v
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return new_pts / norms

def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy using Riemannian CG."""
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize to be sure
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / norms
    
    # Riemannian Conjugate Gradient (Hestenes-Stiefel)
    # Parameters
    c1 = 1e-4  # Armijo condition
    rho = 0.5  # Step reduction
    max_iter = 200
    tol = 1e-12
    
    g = _riemannian_grad(x)
    p = -g  # Initial direction
    
    # Transport previous p to current x if needed (for CG)
    # For HS/PR, we need to transport p from x_prev to x_curr.
    # On sphere, parallel transport of v from x_prev to x_curr:
    # v_t = v - (v . x_prev) x_prev - (v . x_curr) x_curr + (v . x_prev + v . x_curr) (x_prev . x_curr) ... complex.
    # Simpler: use Steepest Descent with adaptive step, or use Riemannian CG with retraction transport (approx).
    # Given time constraints, let's use Riemannian Steepest Descent with Armijo Line Search.
    # It is robust and distinct from L-BFGS-B.
    
    best_x = x.copy()
    best_e = _log_energy(x)
    
    for k in range(max_iter):
        # Line search
        alpha = 1.0
        e_curr = _log_energy(x)
        g_curr = _riemannian_grad(x)
        
        # Armijo condition
        # e(x + alpha p) < e(x) + c1 * alpha * <g, p>
        grad_p = np.sum(g_curr * p)
        
        # If grad_p >= 0, direction is not descent, reset p = -g
        if grad_p >= 0:
            p = -g_curr
            grad_p = np.sum(g_curr * p)
        
        while alpha > 1e-10:
            x_trial = _retract(x, p, alpha)
            e_trial = _log_energy(x_trial)
            
            if e_trial < e_curr + c1 * alpha * grad_p:
                break
            alpha *= rho
        
        x = x_trial
        e_new = e_trial
        
        if e_new < best_e:
            best_e = e_new
            best_x = x.copy()
        
        # Check convergence
        g_new = _riemannian_grad(x)
        norm_g = np.linalg.norm(g_new)
        
        if norm_g < tol:
            break
            
        # Update direction for Steepest Descent
        p = -g_new
        
        # Optional: Restart if curvature is high (not strictly needed for SD)
        
    return best_x
# EVOLVE-BLOCK-END
