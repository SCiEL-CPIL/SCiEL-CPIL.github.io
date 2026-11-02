"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 150

# EVOLVE-BLOCK-START

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _energy_and_grad(pts):
    """
    Compute logarithmic energy and Riemannian gradient on the sphere.
    pts: (N, 3) array of points on the unit sphere.
    Returns: (energy, grad) where grad is (N, 3) projected to tangent space.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dists_sq = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(dists_sq, 1.0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d_safe = np.maximum(d, 1e-15)
    energy = -np.sum(np.log(d_safe))
    
    # Gradient computation
    # grad_i = sum_{j != i} - (x_i - x_j) / |x_i - x_j|^2
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # weighted_sum_pts[i] = sum_j inv_d2[i,j] * x_j
    weighted_sum_pts = inv_d2 @ pts  # (N, 3)
    sum_inv_d2 = np.sum(inv_d2, axis=1)  # (N,)
    
    # grad_i = -x_i * sum_j inv_d2[i,j] + sum_j inv_d2[i,j] * x_j
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space: g - (g . p) p
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    return energy, grad


def _exponential_map(pts, tangent_vecs):
    """
    Apply exponential map on sphere: move points along tangent vectors.
    pts: (N, 3) points on sphere
    tangent_vecs: (N, 3) tangent vectors
    Returns: new points on sphere
    """
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms_safe = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms_safe) / norms_safe
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs


def _riemannian_cg(pts, max_iter=500, tol=1e-10, time_limit=None, start_time=None):
    """
    Riemannian conjugate gradient on the sphere for log-energy minimization.
    Uses Polak-Ribiere beta and backtracking Armijo line search.
    """
    if start_time is None:
        start_time = time.time()
    pts = pts.copy()
    n = len(pts)
    
    def _time_exceeded():
        if time_limit is None:
            return False
        return time.time() - start_time > time_limit
    
    # Initial gradient
    energy, grad = _energy_and_grad(pts)
    p = -grad.copy()  # initial search direction (tangent)
    
    for it in range(max_iter):
        if _time_exceeded():
            break
        
        # Ensure p is in tangent space
        dot_p = np.sum(p * pts, axis=1, keepdims=True)
        p -= dot_p * pts
        
        # Check if p is a descent direction
        grad_dot_p = np.sum(grad * p)
        if grad_dot_p >= 0:
            # Reset direction if not a descent direction
            p = -grad.copy()
            dot_p = np.sum(p * pts, axis=1, keepdims=True)
            p -= dot_p * pts
            grad_dot_p = np.sum(grad * p)
            if grad_dot_p >= 0:
                break
        
        # Adaptive Line Search: Backtracking Armijo
        c1 = 1e-4
        alpha = 0.1
        new_energy = energy
        new_pts = pts
        new_grad = grad
        found = False
        
        for _ in range(20):
            trial_pts = _exponential_map(pts, alpha * p)
            trial_pts = _normalize(trial_pts)
            trial_e, trial_grad = _energy_and_grad(trial_pts)
            
            if trial_e <= energy - c1 * alpha * grad_dot_p:
                new_energy = trial_e
                new_pts = trial_pts
                new_grad = trial_grad
                found = True
                break
            
            alpha *= 0.5
            if _time_exceeded():
                break
        
        if not found:
            break
        
        # Check convergence
        grad_norm = np.linalg.norm(new_grad)
        pts = new_pts
        energy = new_energy
        grad = new_grad
        
        if grad_norm < tol:
            break
        
        # Conjugate gradient update (Polak-Ribiere)
        beta_num = np.sum(new_grad * (new_grad - grad))
        beta_den = np.sum(grad * grad)
        if beta_den > 1e-15:
            beta = beta_num / beta_den
        else:
            beta = 0.0
        beta = max(0.0, min(beta, 1.0))
        
        p = -new_grad + beta * p
        # Project p to tangent space
        dot_p = np.sum(p * new_pts, axis=1, keepdims=True)
        p -= dot_p * new_pts
    
    return pts, _log_energy_fast(pts)


def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))


def _lbfgs_sphere(pts, time_limit=15.0, maxiter=500):
    """
    L-BFGS-B optimization on the sphere using ambient coordinates with
    normalization. This is a practical approach that works well for moderate N.
    """
    start = time.time()
    x0 = pts.flatten().copy()
    best_pts = pts.copy()
    best_energy = _log_energy_fast(best_pts)
    
    def obj_and_grad(x):
        if time.time() - start > time_limit:
            return float('inf'), np.zeros_like(x)
        pts_local = x.reshape(N, 3)
        pts_local = _normalize(pts_local)
        energy, grad = _energy_and_grad(pts_local)
        # For L-BFGS-B in ambient space, we use the unprojected gradient
        # but since we normalize, the effective gradient direction is what matters
        # We recompute ambient gradient
        n = N
        diff = pts_local[:, np.newaxis, :] - pts_local[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, 1.0)
        inv_d2 = 1.0 / dists_sq
        np.fill_diagonal(inv_d2, 0.0)
        weighted_sum_pts = inv_d2 @ pts_local
        sum_inv_d2 = np.sum(inv_d2, axis=1)
        ambient_grad = -pts_local * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
        return energy, ambient_grad.flatten()
    
    try:
        result = minimize(
            obj_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': maxiter,
                'ftol': 1e-14,
                'gtol': 1e-10,
                'maxfun': maxiter * 10,
            }
        )
        
        if result.fun < float('inf') and result.fun < best_energy:
            best_pts = _normalize(result.x.reshape(N, 3))
            best_energy = _log_energy_fast(best_pts)
    except Exception:
        pass
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Start from Fibonacci spiral
    x0 = _fibonacci_spiral(N)
    best_x = x0.copy()
    best_energy = _log_energy_fast(best_x)
    
    # Try multiple random perturbations and optimize
    np.random.seed(42)
    
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline - 25.0:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            # Add random perturbation
            noise = np.random.randn(N, 3) * 0.15
            x_perturbed = x0 + noise
            x_perturbed = _normalize(x_perturbed)
            x_start = x_perturbed
        
        # First pass: Riemannian CG
        x_opt, e_opt = _riemannian_cg(
            x_start, 
            max_iter=300, 
            tol=1e-10, 
            time_limit=deadline - time.time() - 15.0,
            start_time=start_time
        )
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_x = x_opt.copy()
        
        # Second pass: L-BFGS-B polish if time allows
        if time.time() < deadline - 10.0:
            x_opt2, e_opt2 = _lbfgs_sphere(
                x_opt, 
                time_limit=min(15.0, deadline - time.time() - 5.0),
                maxiter=300
            )
            if e_opt2 < best_energy:
                best_energy = e_opt2
                best_x = x_opt2.copy()
    
    # Final polish with Riemannian CG with tighter tolerance
    if time.time() < deadline - 5.0:
        x_final, e_final = _riemannian_cg(
            best_x,
            max_iter=500,
            tol=1e-12,
            time_limit=deadline - time.time() - 2.0,
            start_time=start_time
        )
        if e_final < best_energy:
            best_energy = e_final
            best_x = x_final.copy()
    
    # Final normalization
    pts = _normalize(best_x)
    
    # Validate
    norms = np.linalg.norm(pts, axis=1)
    if np.any(norms < 1e-9) or np.any(~np.isfinite(pts)):
        # Fallback to Fibonacci spiral
        pts = _normalize(_fibonacci_spiral(N))
    
    return pts

# EVOLVE-BLOCK-END
