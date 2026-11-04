"""
Combined approach: Robust Riemannian L-BFGS optimization for logarithmic energy minimization.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|xi-xj|)
    and the Riemannian gradient (projected to tangent space) on the sphere.
    
    Returns:
    energy: float
    rgrad: np.ndarray of shape (N, 3), the Riemannian gradient (tangent vectors)
    """
    n = len(pts)
    # Vectorized distance calculation
    # diff[i, j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0) # Avoid log(0) and div(0)
    
    # Energy calculation using upper triangle
    iu = np.triu_indices(n, k=1)
    dist = np.sqrt(dist_sq)
    e = -np.sum(np.log(dist[iu]))
    
    # Gradient calculation
    # Euclidean gradient for point i: sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2
    # Note: d/dx_i log(1/d_ij) = d/dx_i (-log d_ij) = - (1/d_ij) * (x_i - x_j)/d_ij = -(x_i - x_j)/d_ij^2
    # Wait, let's re-verify the sign.
    # E = sum_{i<j} -log(||x_i - x_j||)
    # grad_i E = sum_{j!=i} - (1/||x_i - x_j||) * (x_i - x_j)/||x_i - x_j|| = - sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    # Let's check: if x_i moves towards x_j, distance decreases, log(1/d) increases. 
    # If x_i -> x_j, E -> +inf. So gradient should point away from x_j?
    # Let's test 1D: f(x) = -log|x-1|. f'(x) = -1/(x-1). 
    # If x=2, f'(2) = -1. Gradient points left (towards 1). 
    # But we want to MINIMIZE energy. We should move AGAINST the gradient.
    # If x=2, moving left increases distance? No, moving left decreases distance to 1.
    # Wait. log(1/d) is large when d is small. We want d to be large.
    # So we want to move AWAY from other points.
    # f(x) = -log|x-1|. If x=2, f(2)=-log(1)=0. If x=3, f(3)=-log(2)<0.
    # So we want to increase x. Gradient is -1. Update: x = x - lr * grad = 2 - lr*(-1) = 2+lr. Correct.
    
    # So Euclidean Grad_i = - sum_{j!=i} (x_i - x_j) / d_ij^2
    # Which is equal to sum_{j!=i} (x_j - x_i) / d_ij^2
    
    inv_d2 = np.zeros_like(dist_sq)
    mask = dist_sq > 1e-12
    inv_d2[mask] = 1.0 / dist_sq[mask]
    np.fill_diagonal(inv_d2, 0.0)
    
    # euc_grad[i] = sum_j (x_j - x_i) * inv_d2[i,j]
    # = sum_j x_j * inv_d2[i,j] - x_i * sum_j inv_d2[i,j]
    # = (inv_d2 @ pts)[i] - pts[i] * (inv_d2 @ ones)[i]
    
    row_sums = np.sum(inv_d2, axis=1)
    weighted_sum = inv_d2 @ pts
    euc_grad = weighted_sum - pts * row_sums[:, np.newaxis]
    
    # Project to tangent space: rgrad = euc_grad - (euc_grad . x_i) x_i
    dot_products = np.sum(euc_grad * pts, axis=1)
    rgrad = euc_grad - dot_products[:, np.newaxis] * pts
    
    return e, rgrad

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _fibonacci_spiral(n):
    """Generate n points on sphere using golden angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _lbfgs_optimize_sphere(pts, time_budget, maxiter=2000, ftol=1e-12, gtol=1e-10):
    """
    L-BFGS optimization on the sphere using retraction.
    We optimize in R^(3N) but project back to sphere after each step.
    The gradient provided is the Riemannian gradient (tangent).
    """
    t_start = time.time()
    n = len(pts)
    
    def objective_and_grad(x_flat):
        if time.time() - t_start > time_budget:
            return np.inf, np.zeros_like(x_flat)
        x = x_flat.reshape(n, 3)
        # Project to sphere to ensure we are evaluating on the manifold
        x = _project_to_sphere(x)
        e, rgrad = _log_energy_and_grad(x)
        return e, rgrad.flatten()
    
    x0 = pts.flatten()
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'maxfun': maxiter * 2,
                'ftol': ftol,
                'gtol': gtol,
                'maxcor': 20,
                'maxls': 50
            }
        )
        
        if result.fun < np.inf:
            best_pts = _project_to_sphere(result.x.reshape(n, 3))
            best_e, _ = _log_energy_and_grad(best_pts)
            return best_pts, best_e
    except Exception:
        pass
        
    return pts, _log_energy_and_grad(pts)[0]

def construct():
    t_start = time.time()
    
    # Initial guess: Fibonacci spiral
    pts = _fibonacci_spiral(N)
    pts = _project_to_sphere(pts)
    
    best_pts = pts.copy()
    best_energy, _ = _log_energy_and_grad(pts)
    
    # Phase 1: Initial L-BFGS polish
    # Use a generous budget for the first pass to get into the basin
    remaining = TIME_LIMIT - (time.time() - t_start)
    budget1 = min(30.0, remaining * 0.4)
    pts, e1 = _lbfgs_optimize_sphere(pts, budget1, maxiter=1000)
    
    if e1 < best_energy:
        best_energy = e1
        best_pts = pts.copy()
        
    # Phase 2: Basin Hopping with L-BFGS
    # Perform several restarts with increasing noise to escape local minima
    remaining = TIME_LIMIT - (time.time() - t_start)
    n_restarts = 5
    # Allocate time for restarts, leaving some for final polish
    restart_time_budget = min(60.0, remaining * 0.6)
    time_per_restart = restart_time_budget / n_restarts
    
    rng = np.random.default_rng(42)
    
    for i in range(n_restarts):
        if time.time() - t_start > TIME_LIMIT - 10: # Keep 10s for final polish
            break
            
        # Create perturbed points
        # Noise scale decreases as we do more restarts? Or increases?
        # Usually, we want to explore more early on.
        # Let's try fixed moderate noise.
        noise_scale = 0.1
        perturbed = _project_to_sphere(best_pts + rng.standard_normal((N, 3)) * noise_scale)
        
        # Polish the perturbed points
        # Use a smaller budget for each restart
        b_h = min(time_per_restart, TIME_LIMIT - (time.time() - t_start) - 10)
        if b_h < 1.0:
            continue
            
        polished_pts, e_polished = _lbfgs_optimize_sphere(perturbed, b_h, maxiter=500)
        
        if e_polished < best_energy:
            best_energy = e_polished
            best_pts = polished_pts.copy()
            
    # Phase 3: Final High-Precision Polish
    remaining = TIME_LIMIT - (time.time() - t_start)
    if remaining > 5.0:
        final_budget = min(30.0, remaining * 0.8)
        pts, e_final = _lbfgs_optimize_sphere(best_pts, final_budget, maxiter=2000, ftol=1e-15, gtol=1e-12)
        
        if e_final < best_energy:
            best_energy = e_final
            best_pts = pts.copy()
            
    # Final Validation and Cleanup
    best_pts = _project_to_sphere(best_pts)
    
    # Ensure no points are too close (validity check)
    # If points are too close, the energy is huge, so this shouldn't happen for a minimizer,
    # but just in case of numerical issues.
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge point j
                best_pts[j] = best_pts[j] + 1e-8 * rng.standard_normal(3)
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
                
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_spiral(N)
        
    return best_pts
