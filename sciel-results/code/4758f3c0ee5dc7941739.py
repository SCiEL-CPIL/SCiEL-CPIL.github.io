"""
Strategy: Riemannian Conjugate Gradient (CG) with Simulated Annealing restarts.
Operates directly on the sphere manifold using exponential/logarithmic maps.
This avoids the singularities and parameterization issues of stereographic coordinates
and allows for global exploration via temperature-controlled perturbations.
"""
N = 150
import numpy as np
import time
import math
import random

def _log_energy_and_grad_sphere(x):
    """
    Computes the logarithmic energy and its Euclidean gradient for points on the sphere.
    x: (N, 3) array of points on the unit sphere.
    Returns: energy, grad (N, 3)
    """
    # diff is (N, N, 3) where diff[i,j] = x_i - x_j
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: 0.5 * sum_{i!=j} -log(dist)
    safe_dist = np.where(mask, dist, 1.0)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
    # Gradient: grad_i = - sum_{j != i} (x_i - x_j) / dist_ij^2
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _exp_map(x, v):
    """
    Exponential map on the unit sphere.
    x: (N, 3) current points.
    v: (N, 3) tangent vectors.
    Returns: new points on the sphere.
    """
    # Ensure v is in the tangent space: v is orthogonal to x
    # v_proj = v - (x . v) x
    dot_xv = np.sum(x * v, axis=1, keepdims=True)
    v_tangent = v - dot_xv * x
    
    theta = np.linalg.norm(v_tangent, axis=1, keepdims=True)
    
    # Avoid division by zero
    theta_safe = np.where(theta < 1e-10, 1e-10, theta)
    
    # Formula: x_new = cos(theta) * x + sin(theta)/theta * v_tangent
    cos_t = np.cos(theta_safe)
    sin_t_over_t = np.sin(theta_safe) / theta_safe
    
    return cos_t * x + sin_t_over_t * v_tangent

def _log_map(x, y):
    """
    Logarithmic map on the unit sphere.
    x: (N, 3) base points.
    y: (N, 3) target points.
    Returns: tangent vectors v such that exp_map(x, v) = y.
    """
    # v = theta * (y - cos(theta) * x) / sin(theta)
    # theta = arccos(x . y)
    dot_xy = np.sum(x * y, axis=1, keepdims=True)
    dot_xy = np.clip(dot_xy, -1.0 + 1e-15, 1.0 - 1e-15)
    
    theta = np.arccos(dot_xy)
    theta_safe = np.where(theta < 1e-10, 1e-10, theta)
    
    sin_t = np.sin(theta_safe)
    
    v = (theta_safe / sin_t) * (y - np.cos(theta_safe) * x)
    
    return v

def _riemannian_cg(x, max_iter=100, tol=1e-8):
    """
    Riemannian Conjugate Gradient method to minimize energy on the sphere.
    x: (N, 3) initial points.
    """
    x_curr = x.copy()
    e_curr, g_curr = _log_energy_and_grad_sphere(x_curr)
    
    # Project gradient to tangent space
    dot_xg = np.sum(x_curr * g_curr, axis=1, keepdims=True)
    g_tangent = g_curr - dot_xg * x_curr
    
    # Initial search direction
    d = -g_tangent
    
    # Initial step size via line search (backtracking)
    alpha = 1.0
    
    for i in range(max_iter):
        # Check convergence
        norm_g = np.linalg.norm(g_tangent)
        if norm_g < tol:
            break
        
        # Line search (backtracking Armijo)
        # We want to decrease energy.
        # f(x + alpha * d) <= f(x) + c * alpha * grad . d
        c = 1e-4
        step_ok = False
        alpha_try = alpha
        for ls_iter in range(20):
            x_try = _exp_map(x_curr, alpha_try * d)
            e_try, _ = _log_energy_and_grad_sphere(x_try)
            
            # Armijo condition
            if e_try <= e_curr + c * alpha_try * np.sum(g_tangent * d):
                step_ok = True
                break
            alpha_try *= 0.5
        
        if not step_ok:
            # If line search fails, try a small fixed step
            alpha_try = 1e-3
            x_try = _exp_map(x_curr, alpha_try * d)
            e_try, _ = _log_energy_and_grad_sphere(x_try)
            if e_try < e_curr:
                step_ok = True
            else:
                # If even small step fails, break or reset
                break
        
        if not step_ok:
            break
            
        # Update x
        x_curr = x_try
        e_curr = e_try
        
        # Compute new gradient
        _, g_new = _log_energy_and_grad_sphere(x_curr)
        dot_xg_new = np.sum(x_curr * g_new, axis=1, keepdims=True)
        g_new_tangent = g_new - dot_xg_new * x_curr
        
        # Check convergence
        if np.linalg.norm(g_new_tangent) < tol:
            break
            
        # Compute beta (Polak-Ribiere)
        # beta = (g_new . (g_new - g_old)) / (g_old . g_old)
        g_diff = g_new_tangent - g_tangent
        num = np.sum(g_new_tangent * g_diff)
        den = np.sum(g_tangent * g_tangent)
        
        if den < 1e-15:
            beta = 0.0
        else:
            beta = max(0.0, num / den)
            
        # Update direction
        d_new = -g_new_tangent + beta * d
        
        # Ensure d is in tangent space
        dot_xd = np.sum(x_curr * d_new, axis=1, keepdims=True)
        d_new = d_new - dot_xd * x_curr
        
        # Check if direction is descent
        if np.sum(g_new_tangent * d_new) >= 0:
            d_new = -g_new_tangent # Reset to steepest descent
            dot_xd = np.sum(x_curr * d_new, axis=1, keepdims=True)
            d_new = d_new - dot_xd * x_curr
            alpha = 1.0 # Reset step size
            
        d = d_new
        g_tangent = g_new_tangent
        
        # Decay alpha slightly to prevent overshoot in later iterations
        alpha = max(0.1, alpha * 0.9)
        
    return x_curr, e_curr

def _fibonacci_sphere(n):
    """Generate a Fibonacci spiral distribution on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return x

def _random_sphere(n, seed):
    """Generate random points on the sphere."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_energy = float('inf')
    best_x = None
    
    # Initial configurations
    inits = [
        _fibonacci_sphere(N),
        _random_sphere(N, 1),
        _random_sphere(N, 42),
        _random_sphere(N, 123)
    ]
    
    for i, x_init in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        # Run Riemannian CG
        x_opt, e_opt = _riemannian_cg(x_init, max_iter=50)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_x = x_opt
            
    # Simulated Annealing / Restart Strategy
    # If we have time, try to escape local minima by perturbing the best solution
    # and re-optimizing.
    if best_x is not None and time.time() < deadline - 20:
        current_best = best_x.copy()
        current_energy = best_energy
        temperature = 0.1
        rng = np.random.default_rng(999)
        
        for restart in range(10):
            if time.time() > deadline - 5:
                break
                
            # Perturb current best
            # Generate random tangent vectors
            v = rng.normal(size=(N, 3))
            # Project to tangent space
            dot_xv = np.sum(current_best * v, axis=1, keepdims=True)
            v_tangent = v - dot_xv * current_best
            
            # Scale by temperature
            v_perturb = temperature * v_tangent
            
            x_perturbed = _exp_map(current_best, v_perturb)
            
            # Optimize from perturbed state
            x_opt, e_opt = _riemannian_cg(x_perturbed, max_iter=30)
            
            if e_opt < current_energy:
                current_energy = e_opt
                current_best = x_opt
                if e_opt < best_energy:
                    best_energy = e_opt
                    best_x = x_opt
                temperature *= 0.5 # Cool down
            else:
                # If worse, accept with probability exp(-deltaE / T) or just reject
                # For simplicity and speed, we mostly reject unless significantly better
                # But to escape, we might accept slightly worse if T is high.
                # Let's just reject to keep it simple and fast, but decrease T.
                temperature *= 0.7

    # Final polish: Run CG again on the absolute best found
    if best_x is not None and time.time() < deadline - 2:
        x_final, e_final = _riemannian_cg(best_x, max_iter=100)
        if e_final < best_energy:
            best_energy = e_final
            best_x = x_final

    if best_x is None:
        best_x = inits[0]
        
    return best_x
