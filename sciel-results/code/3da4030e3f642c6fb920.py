"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import numpy as np
import time
import math
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _log_energy_and_grad_sphere(x):
    """
    Computes the logarithmic energy and its Euclidean gradient for points on the sphere.
    x: (N, 3) array of points on the unit sphere.
    Returns: energy, grad (N, 3)
    """
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    
    safe_dist = np.where(mask, dist, 1.0)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _riemannian_grad(x, grad_euclidean):
    """
    Project Euclidean gradient onto the tangent space of the sphere at x.
    T_x S^2 = {v : <v, x> = 0}
    """
    # grad_tangent = grad_euclidean - <grad_euclidean, x> * x
    proj = np.sum(grad_euclidean * x, axis=1, keepdims=True)
    return grad_euclidean - proj * x

def _exp_map(x, v):
    """
    Exponential map on the unit sphere.
    x: (N, 3) points on sphere
    v: (N, 3) tangent vectors
    Returns: new points on sphere
    """
    norm_v = np.linalg.norm(v, axis=1, keepdims=True)
    # Avoid division by zero
    norm_v_safe = np.where(norm_v < 1e-12, 1e-12, norm_v)
    sin_norm = np.sin(norm_v_safe)
    cos_norm = np.cos(norm_v_safe)
    
    # Handle small norms
    small = norm_v < 1e-12
    sin_norm[small] = 0.0
    cos_norm[small] = 1.0
    
    # exp_x(v) = cos(||v||) x + sin(||v||) / ||v|| v
    # For small v, use Taylor expansion: x + v + 0.5 * (||v||^2) * x - ... but cos/sin formula is stable enough
    # Actually, for very small v, sin(t)/t ~ 1, so exp ~ x + v.
    # Let's use the formula carefully.
    factor = sin_norm / norm_v_safe
    factor[small] = 1.0
    
    new_x = cos_norm * x + factor * v
    # Re-normalize to stay on sphere (numerical stability)
    new_x /= np.linalg.norm(new_x, axis=1, keepdims=True)
    return new_x

def _fibonacci_sphere(n):
    """Generate a Fibonacci spiral distribution on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return x

def _random_sphere(n, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _repulsion_init(n, seed=0, steps=20):
    """
    Simple repulsion-based initialization.
    Start with random points, then iteratively move each point away from others.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    
    for _ in range(steps):
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(n, dtype=bool)
        safe_dist = np.where(mask, dist, 1.0)
        inv_dist_sq = np.zeros_like(safe_dist)
        inv_dist_sq[mask] = 1.0 / (safe_dist[mask] ** 2)
        
        force = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        # Move slightly in direction of force
        x += 0.01 * force
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        
    return x

def _riemannian_cg(x, max_iter=100, tol=1e-8, time_limit=5.0):
    """
    Riemannian Conjugate Gradient method on the sphere.
    """
    start_time = time.time()
    energy, grad_eucl = _log_energy_and_grad_sphere(x)
    grad_tan = _riemannian_grad(x, grad_eucl)
    
    # Check convergence
    if np.linalg.norm(grad_tan) < tol:
        return x, energy
    
    p = -grad_tan
    p /= np.linalg.norm(p)
    best_x = x.copy()
    best_energy = energy
    
    for k in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Line search: minimize f(exp_x(t*p)) for t in [0, 1]
        # Use simple backtracking
        t = 1.0
        c1 = 1e-4
        alpha = 1e-3
        
        for _ in range(20):
            x_new = _exp_map(x, t * p)
            e_new, grad_eucl_new = _log_energy_and_grad_sphere(x_new)
            
            # Armijo condition
            if e_new < energy + c1 * t * np.sum(grad_tan * p):
                break
            t *= alpha
            
        # Update
        x = x_new
        energy = e_new
        grad_eucl = grad_eucl_new
        grad_tan_new = _riemannian_grad(x, grad_eucl)
        
        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()
            
        # Check convergence
        if np.linalg.norm(grad_tan_new) < tol:
            break
            
        # Compute beta (Polak-Ribiere)
        beta = np.sum(grad_tan_new * (grad_tan_new - grad_tan)) / (np.sum(grad_tan * grad_tan) + 1e-12)
        beta = max(0.0, beta)
        
        # Update direction
        p_new = -grad_tan_new + beta * _exp_map(p, t * p)  # Parallel transport approx
        # Better parallel transport: p_transported = p + 2 * <p, x> * x
        # But p is tangent, so <p, x> = 0. So p_transported = p? No.
        # Parallel transport of tangent vector v at x to y=exp_x(t*v) is:
        # v_transport = v + (1 - cos(t*||v||)) * (v/||v||)^2 * x + sin(t*||v||) * ...
        # Simpler: use the fact that for sphere, parallel transport along geodesic is rotation.
        # For simplicity, let's just use the standard CG update in tangent space, ignoring transport for now (approximation)
        # Or use: p_new = -grad_tan_new + beta * p
        p_new = -grad_tan_new + beta * p
        
        # Ensure p_new is tangent
        p_new = _riemannian_grad(x, p_new)
        
        if np.linalg.norm(p_new) < 1e-12:
            p_new = -grad_tan_new
            p_new = _riemannian_grad(x, p_new)
            
        p = p_new
        grad_tan = grad_tan_new
        
    return best_x, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_energy = float('inf')
    best_x = None
    
    # Initial configurations
    inits = [
        _fibonacci_sphere(N),
        _random_sphere(N, seed=42),
        _random_sphere(N, seed=123),
        _repulsion_init(N, seed=0, steps=10),
        _repulsion_init(N, seed=1, steps=10),
    ]
    
    for init_idx, x_init in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        # Run Riemannian CG
        x_res, e_res = _riemannian_cg(x_init, max_iter=50, tol=1e-8, time_limit=15.0)
        
        if e_res < best_energy:
            best_energy = e_res
            best_x = x_res
            
    # If we have time, do a final polish with L-BFGS on the best result
    if best_x is not None and time.time() < deadline - 5:
        # Use L-BFGS on the flattened coordinates, but we need to handle the constraint.
        # Since we are already on the sphere, we can just use L-BFGS on the 3N coordinates
        # and hope it stays close, or project back.
        # Better: use the Riemannian gradient with L-BFGS-B in ambient space?
        # Let's stick to Riemannian CG for consistency, but maybe run it again with tighter tol.
        
        # Alternatively, use scipy minimize with a custom objective that projects to sphere.
        # But Riemannian CG is already good. Let's try a few more local perturbations.
        
        # Local perturbation + re-optimization
        rng = np.random.default_rng(999)
        for _ in range(5):
            if time.time() > deadline - 5:
                break
                
            # Perturb best_x slightly
            x_pert = best_x + 0.01 * rng.normal(size=(N, 3))
            x_pert /= np.linalg.norm(x_pert, axis=1, keepdims=True)
            
            x_res, e_res = _riemannian_cg(x_pert, max_iter=30, tol=1e-8, time_limit=5.0)
            
            if e_res < best_energy:
                best_energy = e_res
                best_x = x_res
                
    if best_x is None:
        best_x = inits[0]
        
    return best_x

# EVOLVE-BLOCK-END
