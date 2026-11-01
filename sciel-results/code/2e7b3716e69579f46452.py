import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _energy_fast(points):
    """Compute logarithmic energy E = sum_{i<j} log(1/||x_i - x_j||) efficiently."""
    # Use squared distances to avoid sqrt where possible
    # d_ij^2 = |x_i|^2 + |x_j|^2 - 2*x_i.x_j = 2 - 2*x_i.x_j for unit vectors
    dot = points @ points.T  # (N, N)
    np.fill_diagonal(dot, 1.0)
    d_sq = 2.0 - 2.0 * dot
    np.maximum(d_sq, 1e-30, out=d_sq)
    # Only sum upper triangle
    iu = np.triu_indices(N, k=1)
    d_sq_upper = d_sq[iu]
    return -np.sum(np.log(np.sqrt(d_sq_upper)))

def _energy_and_grad(points):
    """
    Compute logarithmic energy and its gradient.
    E = sum_{i<j} log(1/||x_i - x_j||)
    dE/dx_i = -sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Gradient calculation
    # grad[i] = -sum_j (x_i - x_j) / dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    grad = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    # Energy calculation using squared distances (faster)
    iu = np.triu_indices(N, k=1)
    d_sq_upper = dist_sq[iu]
    d_sq_upper = np.maximum(d_sq_upper, 1e-30)
    energy = -0.5 * np.sum(np.log(d_sq_upper))
    
    return energy, grad

def _objective_and_grad(x):
    """
    Objective function and gradient for scipy.minimize.
    x is a flat array of shape (3*N,).
    We project to sphere to enforce constraint.
    """
    points = x.reshape(N, 3)
    
    # Project to sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points_sphere = points / norms
    
    # Calculate energy and gradient on sphere
    e, g = _energy_and_grad(points_sphere)
    
    # Add penalty for leaving the sphere
    lambda_pen = 100.0
    diff_pen = points - points_sphere
    penalty = 0.5 * lambda_pen * np.sum(diff_pen**2)
    grad_pen = lambda_pen * diff_pen
    
    total_grad = (g + grad_pen).flatten()
    
    return e + penalty, total_grad

def _fibonacci_spiral():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(seed=None):
    """Generate random points on sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _steepest_descent_sphere(points, max_iter=2000, tol=1e-14, deadline=None):
    """
    Steepest descent on the sphere with adaptive step size and backtracking.
    Uses tangent-space gradient projection.
    """
    p = points.copy()
    e, g = _energy_and_grad(p)
    best_e = e
    best_p = p.copy()
    
    step_size = 0.05
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Project gradient to tangent space
        g_tan = g - (g * p).sum(axis=1, keepdims=True) * p
        
        # Check convergence
        g_norm = np.linalg.norm(g_tan)
        if g_norm < tol:
            break
        
        # Backtracking line search (Armijo condition)
        step = step_size
        accepted = False
        c1 = 1e-4  # Armijo constant
        
        for _ in range(20):
            p_new = p - step * g_tan
            p_new = _project_to_sphere(p_new)
            e_new, _ = _energy_and_grad(p_new)
            
            if e_new < e - c1 * step * np.sum(g_tan**2):
                p = p_new
                e = e_new
                if e < best_e:
                    best_e = e
                    best_p = p.copy()
                _, g = _energy_and_grad(p)
                accepted = True
                break
            step *= 0.5
        
        if not accepted:
            # Try smaller initial step for next iteration
            step_size *= 0.5
            if step_size < 1e-10:
                break
        else:
            # Gradually increase step size if accepted
            step_size = min(step_size * 1.1, 0.1)
    
    return best_p, best_e

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # 1. Direct Initialization: Fibonacci Spiral
    points = _fibonacci_spiral()
    best_points = points.copy()
    
    # Initial energy calculation
    e_init, _ = _energy_and_grad(best_points)
    best_energy = e_init
    
    # 2. Optimization Phase with L-BFGS
    x0 = best_points.flatten()
    
    try:
        # First run: Standard L-BFGS
        res = minimize(_objective_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
        
        if hasattr(res, 'x') and res.x is not None:
            new_points = _project_to_sphere(res.x.reshape(N, 3))
            e_new, _ = _energy_and_grad(new_points)
            if e_new < best_energy:
                best_energy = e_new
                best_points = new_points
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT * 0.5:
            return best_points

        # Second run: Refine with tighter tolerances
        x1 = best_points.flatten()
        res2 = minimize(_objective_and_grad, x1, method='L-BFGS-B', jac=True,
                       options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 30})
        
        if hasattr(res2, 'x') and res2.x is not None:
            new_points = _project_to_sphere(res2.x.reshape(N, 3))
            e_new, _ = _energy_and_grad(new_points)
            if e_new < best_energy:
                best_energy = e_new
                best_points = new_points

    except Exception:
        pass

    # 3. High-precision polish with steepest descent
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        polished, e_polished = _steepest_descent_sphere(
            best_points, max_iter=5000, tol=1e-15, deadline=deadline - 3.0
        )
        if e_polished < best_energy:
            best_energy = e_polished
            best_points = polished

    # 4. Final validation and cleanup
    best_points = _project_to_sphere(best_points)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                rng = np.random.RandomState(i * 1000 + j)
                best_points[i] += rng.randn(3) * 1e-6
                best_points[i] = _project_to_sphere(best_points[i].reshape(1, 3))[0]
    
    return best_points
