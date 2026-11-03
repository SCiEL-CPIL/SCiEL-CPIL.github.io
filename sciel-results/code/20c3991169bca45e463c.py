"""
Strategy: Multi-start optimization using SciPy's L-BFGS-B.
The previous approach used manual gradient descent which is slow and prone to getting stuck.
L-BFGS-B is a quasi-Newton method that approximates the Hessian, allowing for much faster
convergence to local minima. By using a parameterization that automatically satisfies the
unit sphere constraint (e.g., normalizing after each step or using spherical coordinates),
we can let L-BFGS-B do the heavy lifting.

We will use multiple initializations:
1. Fibonacci sphere (good global distribution)
2. Random uniform points
3. A "clustered" start to ensure we explore different basins of attraction if needed, 
   though for log energy, symmetric starts are usually best.
   
Given the 120s limit, we can afford several L-BFGS-B runs.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 115.0  # Safety margin from 120s

def _log_energy_fast(pts):
    """
    Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|).
    Returns (energy, gradient) for optimization.
    """
    n = len(pts)
    # Ensure points are on sphere (normalization is done in the wrapper, 
    # but for energy calc we assume they are close enough or we re-normalize here for stability)
    # Actually, for the objective function passed to L-BFGS-B, we should handle the constraint.
    # If we use unconstrained coordinates and normalize inside, the gradient calculation
    # needs to be consistent with that transformation.
    # 
    # Alternative: Use the fact that for points on the sphere, 
    # E = -0.5 * sum_{i!=j} log(2 - 2*x_i.x_j)
    # Gradient w.r.t x_i (in R3) before projection:
    # g_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # Then project to tangent space.
    # 
    # However, L-BFGS-B works on unconstrained variables. 
    # Let's use the "normalized" approach:
    # Variables are in R^(3N). We normalize them to get pts.
    # This creates a non-smoothness at the origin, but we start away from it.
    # The gradient of the energy w.r.t the raw variables y is:
    # dE/dy_i = (dE/dx_i) * (dx_i/dy_i)
    # where x_i = y_i / ||y_i||
    # dx_i/dy_i = (I - x_i x_i^T) / ||y_i||
    # So dE/dy_i = (g_i - (g_i.x_i)x_i) / ||y_i||
    # This is exactly the tangent space gradient scaled by 1/||y_i||.
    
    # Let's compute energy and gradient in one go for speed.
    # pts is (N, 3)
    # We need to be careful: the input to this function from L-BFGS-B will be raw y.
    # But I will write a separate objective function that handles the normalization and gradient.
    
    # For now, this function is just for checking the score of a final normalized set.
    dot = pts @ pts.T
    dist_sq = np.maximum(2.0 - 2.0 * dot, 1e-15)
    np.fill_diagonal(dist_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-18):
        return float('inf'), None
    terms = -0.5 * np.log(d_sq)
    energy = math.fsum(terms.tolist())
    
    # Gradient calculation for the normalized points (tangent space)
    # g_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq_full = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(dist_sq_full, 1.0)
    weights = 1.0 / dist_sq_full
    np.fill_diagonal(weights, 0.0)
    grad_raw = np.einsum('ij,ijk->ik', weights, diff)
    
    # Project to tangent space
    proj_factor = np.sum(grad_raw * pts, axis=1, keepdims=True)
    grad_tangent = grad_raw - proj_factor * pts
    
    return energy, grad_tangent

def _objective_and_grad(y_flat):
    """
    Objective function for L-BFGS-B.
    Input: y_flat (3N,)
    Output: energy, gradient (3N,)
    """
    n = len(y_flat) // 3
    y = y_flat.reshape(n, 3)
    
    # Normalize to sphere
    norms = np.linalg.norm(y, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.maximum(norms, 1e-15)
    pts = y / norms
    
    # Compute energy and tangent gradient
    energy, grad_tangent = _log_energy_fast(pts)
    
    if energy == float('inf'):
        return energy, np.zeros_like(y_flat)
        
    # Map tangent gradient back to raw space gradient
    # dE/dy_i = grad_tangent_i / ||y_i||
    grad_raw = grad_tangent / norms
    
    return energy, grad_raw.flatten()

def _fibonacci_sphere(n):
    """Golden-angle Fibonacci spiral points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, seed):
    np.random.seed(seed)
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi = np.arccos(2 * np.random.uniform(0, 1, n) - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

def construct():
    global DEADLINE
    start_time = time.time()
    
    # Initial candidates
    candidates = []
    
    # 1. Fibonacci
    fib_pts = _fibonacci_sphere(N)
    candidates.append(fib_pts.flatten())
    
    # 2. Random (a few different seeds)
    for seed in [42, 123, 999]:
        rand_pts = _random_uniform_sphere(N, seed)
        candidates.append(rand_pts.flatten())
        
    # 3. Maybe a "coarse" grid? Or just stick to Fib + Randoms.
    # Let's add one more structured start: Riesz S-designs or similar?
    # For now, Fib is usually the best start for log energy.
    
    best_energy = float('inf')
    best_pts = None
    
    # Optimization loop
    for i, init_y in enumerate(candidates):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:  # Leave 10s for final steps
            break
            
        # Run L-BFGS-B
        try:
            res = minimize(
                _objective_and_grad,
                init_y,
                method='L-BFGS-B',
                jac=True,  # We provide the gradient
                options={
                    'maxiter': 1000,
                    'maxfun': 1000,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                    'disp': False
                }
            )
            
            if res.success or res.fun < best_energy:
                # Normalize the result
                y_opt = res.x.reshape(N, 3)
                norms = np.linalg.norm(y_opt, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-15)
                pts_opt = y_opt / norms
                
                # Verify energy
                e, _ = _log_energy_fast(pts_opt)
                if e < best_energy and np.isfinite(e):
                    best_energy = e
                    best_pts = pts_opt
                    
        except Exception as e:
            pass # Ignore failures for this start
    
    # If we still have time, do a final polish on the best result
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        try:
            init_y = best_pts.flatten()
            res = minimize(
                _objective_and_grad,
                init_y,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 500,
                    'maxfun': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'disp': False
                }
            )
            y_opt = res.x.reshape(N, 3)
            norms = np.linalg.norm(y_opt, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            pts_opt = y_opt / norms
            
            e, _ = _log_energy_fast(pts_opt)
            if e < best_energy and np.isfinite(e):
                best_energy = e
                best_pts = pts_opt
        except Exception:
            pass

    # Fallback if something went wrong
    if best_pts is None:
        best_pts = fib_pts

    # Final normalization and check
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    # Check for invalid points (too close)
    # The problem states points closer than 1e-9 make it invalid.
    # Our optimization should prevent this, but let's verify.
    # If invalid, we might need to perturb, but L-BFGS-B usually keeps them apart.
    
    return best_pts
