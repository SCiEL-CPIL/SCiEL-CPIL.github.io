import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 145.0  # Safety margin from 150s

def _log_energy(pts):
    """
    Compute logarithmic energy.
    pts: (N, 3) array of points on the unit sphere.
    Returns: energy
    """
    n = len(pts)
    if n < 2:
        return 0.0
    # Compute pairwise distances
    # dist_sq[i,j] = ||x_i - x_j||^2
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    
    # Check for very small distances
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-12):
        return float('inf')
        
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i<j} log(dist_sq_ij)
    # Note: log(1/d) = -log(d). d = sqrt(d_sq). log(d) = 0.5 log(d_sq).
    # So term is -0.5 log(d_sq).
    # Sum over i<j is half the sum over i!=j.
    # E = -0.5 * (1/2) * sum_{i!=j} log(d_sq_ij) = -0.25 * sum_{i!=j} log(d_sq_ij)
    log_d_sq = np.log(np.maximum(dist_sq, 1e-300))
    np.fill_diagonal(log_d_sq, 0.0)
    energy = -0.25 * np.sum(log_d_sq)
    return float(energy)

def _energy_and_grad(pts):
    """
    Compute energy and Riemannian gradient on the sphere.
    pts: (N, 3) array of points on the unit sphere.
    Returns: energy, grad (N, 3) array of tangent vectors.
    """
    n = len(pts)
    if n < 2:
        return 0.0, np.zeros_like(pts)
    
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-12):
        return float('inf'), np.zeros_like(pts)

    # Euclidean gradient of E = -0.25 * sum_{i!=j} log(d_ij^2)
    # dE/dx_i = -0.25 * sum_{j!=i} (1/d_ij^2) * 2*(x_i - x_j) = -0.5 * sum_{j!=i} (x_i - x_j)/d_ij^2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    weights = -0.5 / dist_sq
    np.fill_diagonal(weights, 0.0)
    
    grad_eucl = np.einsum('ij,ijk->ik', weights, diff)
    
    # Riemannian gradient: project Euclidean gradient onto tangent space
    # grad_riem = grad_eucl - <grad_eucl, x_i> x_i
    # Since x_i is unit, <grad_eucl, x_i> is the radial component.
    proj_factors = np.sum(grad_eucl * pts, axis=1, keepdims=True)
    grad_riem = grad_eucl - proj_factors * pts
    
    # Energy calculation
    log_d_sq = np.log(np.maximum(dist_sq, 1e-300))
    np.fill_diagonal(log_d_sq, 0.0)
    energy = -0.25 * np.sum(log_d_sq)
    
    return float(energy), grad_riem

def _exp_map(pts, v):
    """
    Exponential map on the unit sphere.
    pts: (N, 3) points on sphere.
    v: (N, 3) tangent vectors.
    Returns: (N, 3) points on sphere.
    """
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    # Avoid division by zero
    safe_norms = np.where(norms > 1e-15, norms, 1.0)
    cos_theta = np.cos(norms.squeeze())
    sin_theta = np.sin(norms.squeeze())
    
    # x' = cos(|v|) x + sin(|v|)/|v| v
    # If |v| is small, sin(|v|)/|v| ~ 1.
    factor = np.where(norms.squeeze() > 1e-15, sin_theta / norms.squeeze(), 1.0)
    factor = factor.reshape(-1, 1)
    
    new_pts = cos_theta.reshape(-1, 1) * pts + factor * v
    # Normalize to ensure we stay on sphere
    new_norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_norms = np.where(new_norms > 1e-15, new_norms, 1.0)
    return new_pts / new_norms

def _retract(pts, v):
    """
    Retraction map (simpler, faster than exp map).
    x' = normalize(x + v)
    """
    new_pts = pts + v
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    norms = np.where(norms > 1e-15, norms, 1.0)
    return new_pts / norms

def _riemannian_cg(pts, max_iter=50, tol=1e-8):
    """
    Riemannian Conjugate Gradient method.
    """
    x = pts.copy()
    e, g = _energy_and_grad(x)
    if not np.isfinite(e):
        return x, e
    
    # Project g to ensure it's tangent (it should be, but numerical errors)
    proj_factors = np.sum(g * x, axis=1, keepdims=True)
    g = g - proj_factors * x
    
    p = -g
    alpha = 1.0
    beta = 0.0
    
    for k in range(max_iter):
        # Line search: minimize f(exp(x, alpha * p))
        # Use simple backtracking
        best_alpha = alpha
        best_e = e
        for _ in range(20):
            x_trial = _exp_map(x, alpha * p)
            e_trial, _ = _energy_and_grad(x_trial)
            if e_trial < best_e:
                best_e = e_trial
                best_alpha = alpha
            else:
                alpha *= 0.5
                if alpha < 1e-10:
                    break
        
        x = _exp_map(x, best_alpha * p)
        e_new, g_new = _energy_and_grad(x)
        
        if not np.isfinite(e_new):
            # Revert
            x = _exp_map(x, -best_alpha * p)
            e_new, g_new = _energy_and_grad(x)
            if not np.isfinite(e_new):
                break
        
        # Check convergence
        grad_norm = np.linalg.norm(g_new)
        if grad_norm < tol:
            break
        
        # Compute beta (Polak-Ribiere)
        proj_factors_new = np.sum(g_new * x, axis=1, keepdims=True)
        g_new_tan = g_new - proj_factors_new * x
        
        # <g_new, p> - <g_old, p>
        # Note: p is tangent at x_old, g_new is tangent at x_new.
        # We need to parallel transport p to x_new or just use inner product in R^3.
        # For simplicity, use R^3 inner product as an approximation.
        inner_new = np.sum(g_new_tan * p)
        inner_old = np.sum(g * p)
        beta = max(0, inner_new - inner_old) / (np.sum(g*g) + 1e-15)
        
        p = -g_new_tan + beta * p
        
        # Ensure p is tangent
        proj_factors_p = np.sum(p * x, axis=1, keepdims=True)
        p = p - proj_factors_p * x
        
        g = g_new_tan
        e = e_new
        
    return x, e

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    theta = np.random.uniform(0, 2 * math.pi, n)
    phi = np.arccos(2 * np.random.uniform(0, 1) - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

def _random_rotation_matrix(seed=None):
    if seed is not None:
        np.random.seed(seed)
    # Generate random rotation using QR decomposition of a random matrix
    A = np.random.randn(3, 3)
    Q, R = np.linalg.qr(A)
    # Ensure determinant is 1 (proper rotation)
    d = np.sign(np.linalg.det(Q))
    Q[:, 0] *= d
    return Q

def construct():
    start_time = time.time()
    
    best_energy = float('inf')
    best_pts = None
    
    # Initializations
    inits = []
    
    # 1. Fibonacci Sphere
    inits.append(_fibonacci_sphere(N))
    
    # 2. Random Uniform
    for seed in [42, 123, 999]:
        inits.append(_random_uniform_sphere(N, seed=seed))
        
    # Optimization Loop using Riemannian CG
    for i, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
            
        try:
            # Riemannian CG
            opt_pts, e = _riemannian_cg(init_pts, max_iter=100, tol=1e-8)
            
            if e < best_energy and np.isfinite(e):
                best_energy = e
                best_pts = opt_pts.copy()
                
        except Exception:
            continue
            
    # Basin Hopping Phase
    if best_pts is not None:
        current_pts = best_pts.copy()
        current_energy = best_energy
        T = 1.0  # Temperature for Metropolis acceptance
        min_T = 0.01
        cooling_rate = 0.95
        
        while time.time() - start_time < DEADLINE - 5:
            # Perturb current configuration
            # Apply a small random rotation to the whole set
            # Or perturb individual points? Global rotation is a symmetry, so it doesn't change energy.
            # We need to break symmetry. Perturb individual points by adding random tangent noise.
            noise_scale = 0.1 * T
            noise = np.random.randn(N, 3) * noise_scale
            # Project noise to tangent space
            proj_factors = np.sum(noise * current_pts, axis=1, keepdims=True)
            noise_tan = noise - proj_factors * current_pts
            
            trial_pts = _exp_map(current_pts, noise_tan)
            trial_energy, _ = _energy_and_grad(trial_pts)
            
            if np.isfinite(trial_energy):
                # Metropolis acceptance
                delta_e = trial_energy - current_energy
                if delta_e < 0 or np.random.random() < math.exp(-delta_e / T):
                    current_pts = trial_pts
                    current_energy = trial_energy
                    
                    # Local optimization from this new point
                    opt_pts, opt_energy = _riemannian_cg(current_pts, max_iter=50, tol=1e-8)
                    if np.isfinite(opt_energy) and opt_energy < best_energy:
                        best_energy = opt_energy
                        best_pts = opt_pts.copy()
                        current_pts = opt_pts
                        current_energy = opt_energy
                        
            T *= cooling_rate
            if T < min_T:
                break

    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)

    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts
