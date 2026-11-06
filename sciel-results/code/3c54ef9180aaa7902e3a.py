"""Rewritten program for logarithmic energy minimization of 150 points on the sphere."""
import numpy as np
import math
import time

N = 150
TIME_LIMIT = 115.0  # safety margin before 120s limit


def _energy_and_grad(points):
    """Compute logarithmic energy and its gradient (tangent to sphere).
    
    Energy: sum_{i<j} log(1/|x_i - x_j|)
    Gradient w.r.t. x_i: sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(points)
    # Pairwise distances
    # points shape: (n, 3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j!=i} diff[i,j] / dist_sq[i,j]
    # diff[i,j] = x_i - x_j
    inv_dist_sq = 1.0 / dist_sq  # (n, n)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad shape: (n, 3)
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Project gradient to tangent space: grad_tangent = grad - (grad . x) * x
    dot = np.sum(grad * points, axis=1, keepdims=True)
    grad_tangent = grad - dot * points
    
    return energy, grad_tangent


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _lbfgsb_optimize(points, max_iter=5000, time_limit=110.0):
    """Optimize using L-BFGS-B on flattened coordinates with sphere constraint handled by projection."""
    from scipy.optimize import minimize
    
    x0 = points.ravel()
    n = len(points)
    start_time = time.time()
    
    def fun_and_grad(x):
        pts = x.reshape(n, 3)
        # Project to sphere
        pts = _project_to_sphere(pts)
        e, g = _energy_and_grad(pts)
        # The gradient in the flat space needs to account for projection,
        # but since we're using L-BFGS-B with projection, we just return the tangent gradient flattened
        return e, g.ravel()
    
    result = minimize(fun_and_grad, x0, method='L-BFGS-B', 
                      jac=True, options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10,
                                          'maxfun': 50000})
    
    elapsed = time.time() - start_time
    best_pts = _project_to_sphere(result.x.reshape(n, 3))
    return best_pts, result.fun, elapsed


def _gradient_descent(points, max_iter=20000, time_limit=110.0):
    """Gradient descent with adaptive learning rate."""
    n = len(points)
    start_time = time.time()
    best_energy, best_pts = _energy_and_grad(points)[0], points.copy()
    lr = 0.01
    min_lr = 1e-8
    patience = 0
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        energy, grad = _energy_and_grad(points)
        
        if energy < best_energy - 1e-15:
            best_energy = energy
            best_pts = points.copy()
            patience = 0
        else:
            patience += 1
        
        # Adaptive learning rate
        if patience > 100:
            lr *= 0.5
            if lr < min_lr:
                break
            patience = 0
        
        # Gradient step
        points = points - lr * grad
        points = _project_to_sphere(points)
        
        # Check convergence
        if np.linalg.norm(grad) < 1e-12:
            break
    
    return best_pts, best_energy, time.time() - start_time


def _polish(points, time_limit=10.0):
    """Fine-tune with very small learning rate gradient descent."""
    n = len(points)
    start_time = time.time()
    best_energy, best_pts = _energy_and_grad(points)[0], points.copy()
    lr = 0.001
    
    for it in range(5000):
        if time.time() - start_time > time_limit:
            break
        
        energy, grad = _energy_and_grad(points)
        
        if energy < best_energy - 1e-15:
            best_energy = energy
            best_pts = points.copy()
        
        points = points - lr * grad
        points = _project_to_sphere(points)
        
        if np.linalg.norm(grad) < 1e-14:
            break
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    
    # Generate initial points using Fibonacci spiral
    points = _fibonacci_sphere(N)
    
    # First optimization phase: L-BFGS-B
    try:
        points, energy, elapsed = _lbfgsb_optimize(points, max_iter=3000, time_limit=60.0)
    except Exception:
        points, energy, elapsed = _gradient_descent(points, max_iter=10000, time_limit=60.0)
    
    elapsed_total = time.time() - start_time
    
    # Second optimization phase: Gradient descent for further refinement
    remaining_time = TIME_LIMIT - elapsed_total - 15.0  # Leave 15s for polish
    if remaining_time > 5.0:
        try:
            points, energy2, elapsed2 = _gradient_descent(points, max_iter=10000, time_limit=remaining_time)
            points, energy = points, energy2
        except Exception:
            pass
    
    # Final polish phase
    elapsed_total = time.time() - start_time
    polish_time = min(10.0, TIME_LIMIT - elapsed_total - 2.0)
    if polish_time > 1.0:
        try:
            points, energy = _polish(points, time_limit=polish_time)
        except Exception:
            pass
    
    # Ensure valid output
    points = _project_to_sphere(points)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-9:
                # Perturb slightly
                points[j] += 1e-8 * np.random.randn(3)
                points[j] = points[j] / np.linalg.norm(points[j])
    
    return points
