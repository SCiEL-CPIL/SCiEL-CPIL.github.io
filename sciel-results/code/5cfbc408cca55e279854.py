"""
Optimized solution for minimizing logarithmic energy of 100 points on the sphere.
Strategy:
1. Generate diverse initializations (Fibonacci, Sobol, Random).
2. For each initialization, perform Riemannian gradient descent to converge to a local basin.
3. Polish the result using L-BFGS-B with tight tolerances.
4. Track the best energy found and return the corresponding points.
"""

import numpy as np
import math
import time
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
_DEADLINE = 145.0  # Leave 5s margin for safety
_START_TIME = None

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and its Euclidean gradient.
    Energy E = sum_{i<j} log(1/|x_i - x_j|)
    For unit vectors, |x_i - x_j|^2 = 2 - 2 x_i . x_j.
    E = -0.5 * sum_{i<j} log(2 - 2 x_i . x_j)
    
    Gradient w.r.t x_k:
    dE/dx_k = sum_{j != k} (x_j - x_k) / |x_k - x_j|^2
    """
    # pts has shape (N, 3)
    dots = pts @ pts.T
    # Clip to avoid log of negative or zero due to floating point errors
    dots = np.clip(dots, -1.0, 1.0 - 1e-15)
    
    sq_dist = 2.0 - 2.0 * dots
    np.fill_diagonal(sq_dist, 1.0) # Avoid division by zero in gradient calc
    
    # Energy
    iu = np.triu_indices(N, k=1)
    sq_dist_upper = sq_dist[iu]
    # Ensure no zero distances (should be handled by clipping, but just in case)
    sq_dist_upper = np.maximum(sq_dist_upper, 1e-15)
    energy = -0.5 * math.fsum(math.log(v) for v in sq_dist_upper)
    
    # Gradient
    # inv_sq_dist[k, j] = 1 / |x_k - x_j|^2 for k != j, 0 otherwise
    inv_sq_dist = 1.0 / sq_dist
    np.fill_diagonal(inv_sq_dist, 0.0)
    
    # grad_k = sum_{j != k} (x_j - x_k) / sq_dist_kj
    #        = sum_{j != k} x_j / sq_dist_kj - x_k * sum_{j != k} 1/sq_dist_kj
    #        = (inv_sq_dist @ pts)_k - x_k * (inv_sq_dist @ ones)_k
    
    row_sums = inv_sq_dist.sum(axis=1)
    grad = (inv_sq_dist @ pts) - pts * row_sums[:, np.newaxis]
    
    return energy, grad

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _tangent_project(x, grad):
    """Project Euclidean gradient onto the tangent space of the sphere at x."""
    # x is (N, 3), grad is (N, 3)
    # Projection of g onto T_x S^2 is g - (g . x) x
    dot = np.sum(grad * x, axis=1, keepdims=True)
    return grad - dot * x

def _riemannian_gradient_flow(pts, max_steps=200, tol=1e-10):
    """
    Perform Riemannian gradient descent with backtracking line search.
    """
    x = _project_to_sphere(pts)
    f, g = _energy_and_grad(x)
    g_tan = _tangent_project(x, g)
    
    for step in range(max_steps):
        # Check convergence
        g_norm = np.linalg.norm(g_tan)
        if g_norm < tol:
            break
        
        # Backtracking line search
        # Initial step size
        # Heuristic: inverse of Lipschitz constant estimate
        L = np.max(np.linalg.norm(np.linalg.inv(np.diag(1.0))) ) # Placeholder, we'll use standard backtracking
        alpha = 0.1
        c1 = 1e-4 # Armijo condition
        
        # Compute search direction
        d = -g_tan
        
        # Normalize direction to have unit norm for consistent step size interpretation
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-15:
            break
        d = d / d_norm
        
        while True:
            # Move in tangent space
            x_new_tan = x + alpha * d
            # Retract to sphere (simple normalization is a valid retraction for sphere)
            x_new = _project_to_sphere(x_new_tan)
            
            f_new, g_new = _energy_and_grad(x_new)
            
            # Check Armijo condition
            # f(x_new) <= f(x) + c1 * alpha * <g_tan, d>
            # Note: <g_tan, d> is negative since d = -g_tan / ||g_tan||
            # So RHS is f(x) - c1 * alpha * ||g_tan||
            
            if f_new <= f + c1 * alpha * (-np.sum(g_tan * d)):
                # Accept step
                x = x_new
                f = f_new
                g = g_new
                g_tan = _tangent_project(x, g)
                break
            else:
                alpha *= 0.5
                if alpha < 1e-12:
                    break
        
        if alpha < 1e-12:
            break
            
    return x, f

def _generate_fibonacci():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts

def _generate_sobol(seed=0):
    sampler = qmc.QMC(3, seed=seed)
    # Sobol sequences are in [0,1)^d. We need to map to sphere.
    # A simple mapping: use inverse CDF for uniform distribution on sphere.
    # Or just generate random points using the Sobol samples as seeds for a PRNG? 
    # Actually, qmc.QMC can sample from the unit cube. 
    # We can use the standard mapping:
    # z = 2*u1 - 1
    # theta = 2*pi*u2
    # r = sqrt(1-z^2)
    # x = r*cos(theta), y=r*sin(theta), z=z
    
    samples = sampler.random(N)
    u1 = samples[:, 0]
    u2 = samples[:, 1]
    
    z = 2.0 * u1 - 1.0
    theta = 2.0 * np.pi * u2
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    return _project_to_sphere(pts)

def _generate_random(seed=42):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    return _project_to_sphere(pts)

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_pts = None
    best_energy = float('inf')
    
    def _update_best(pts):
        nonlocal best_pts, best_energy
        pts = _project_to_sphere(pts)
        # Check validity: no points too close
        dots = pts @ pts.T
        np.fill_diagonal(dots, -2.0)
        max_dot = dots.max()
        if max_dot > 1.0 - 1e-9:
            return # Invalid
        e, _ = _energy_and_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()

    # List of initializations to try
    initializations = []
    
    # 1. Fibonacci Spiral
    initializations.append(_generate_fibonacci())
    
    # 2. Sobol Sequences (multiple seeds)
    for s in range(5):
        initializations.append(_generate_sobol(seed=s))
        
    # 3. Random Points (multiple seeds)
    for s in range(10):
        initializations.append(_generate_random(seed=s))

    # Optimization Loop
    for i, init_pts in enumerate(initializations):
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - 10.0:
            break
            
        # Step 1: Riemannian Gradient Descent
        try:
            pts_refined, e_refined = _riemannian_gradient_flow(init_pts, max_steps=100)
        except Exception:
            pts_refined = init_pts
            e_refined, _ = _energy_and_grad(pts_refined)
            
        # Step 2: L-BFGS-B Polish
        try:
            result = minimize(
                _energy_and_grad,
                pts_refined,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            _update_best(result.x)
        except Exception:
            # If L-BFGS fails, try the refined points from gradient flow
            _update_best(pts_refined)
            
    # If no valid points were found (should not happen), fallback to Fibonacci
    if best_pts is None:
        best_pts = _generate_fibonacci()
        # Force update even if energy is inf? No, just return it.
        _update_best(best_pts)
        
    # Final Polish: Run one more L-BFGS-B on the best point with even tighter tolerance
    # This is a "high-precision polish pass"
    if time.time() - _START_TIME < _DEADLINE - 5.0:
        try:
            result = minimize(
                _energy_and_grad,
                best_pts,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 100}
            )
            _update_best(result.x)
        except Exception:
            pass
            
    return best_pts

