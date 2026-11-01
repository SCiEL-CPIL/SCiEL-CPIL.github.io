"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize, Bounds

N = 150
DEADLINE = 115.0  # safety margin before 120s limit


def _log_energy_and_grad(points):
    """Compute logarithmic energy and its gradient for points on the sphere."""
    n = points.shape[0]
    # Compute all pairwise distances
    # points: (n, 3)
    # dist^2[i,j] = |p_i - p_j|^2
    sq_dists = np.sum(points**2, axis=1)
    sq_dists = sq_dists[:, None] + sq_dists[None, :] - 2.0 * points @ points.T
    np.maximum(sq_dists, 0.0, out=sq_dists)
    dists = np.sqrt(sq_dists)
    
    # Avoid log(0) for i==j
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    energy = -np.sum(log_dists)
    
    # Gradient: d/dp_i sum_{j!=i} log(1/|p_i - p_j|) = -sum_{j!=i} (p_i - p_j)/|p_i - p_j|^2
    # = sum_{j!=i} (p_j - p_i)/|p_i - p_j|^2
    # But we need to project to tangent space
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = dists[i, j]
                grad[i] += (points[j] - points[i]) / (d * d)
    
    # Project gradient to tangent space: g_tan = g - (g . p) * p
    for i in range(n):
        g_dot_p = np.dot(grad[i], points[i])
        grad[i] -= g_dot_p * points[i]
    
    return energy, grad


def _log_energy_fast(points):
    """Fast energy computation using vectorized operations."""
    n = points.shape[0]
    sq_dists = np.sum(points**2, axis=1)
    sq_dists = sq_dists[:, None] + sq_dists[None, :] - 2.0 * points @ points.T
    np.maximum(sq_dists, 0.0, out=sq_dists)
    dists = np.sqrt(sq_dists)
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dists[iu]))


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def construct():
    start_time = time.time()
    
    # Generate initial points using Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_energy = _log_energy_fast(points)
    
    # Use L-BFGS-B on the sphere via parametrization
    # We'll work in R^3 but project back to sphere after each step
    # Actually, let's use a different approach: work with angles or use scipy with constraints
    
    # Better approach: use the fact that the gradient on the sphere is the projection
    # We'll use a custom line search with retraction
    
    def objective_and_grad(x_flat):
        """Objective and gradient in flat R^{3N} space, with projection to sphere."""
        pts = x_flat.reshape(N, 3)
        pts = _normalize(pts)
        energy, grad = _log_energy_and_grad(pts)
        # The gradient in flat space needs to account for the normalization
        # For simplicity, we'll just use the projected gradient
        return energy, grad.flatten()
    
    # Use L-BFGS-B with multiple restarts from different initial conditions
    best_energy = _log_energy_fast(points)
    best_points = points.copy()
    
    # First, do a single optimization from the Fibonacci start
    x0 = points.flatten()
    
    def obj(x):
        pts = _normalize(x.reshape(N, 3))
        return _log_energy_fast(pts)
    
    def obj_grad(x):
        pts = _normalize(x.reshape(N, 3))
        e, g = _log_energy_and_grad(pts)
        return e, g.flatten()
    
    # Use L-BFGS-B
    try:
        result = minimize(
            obj_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 5000}
        )
        if result.success or result.fun < best_energy:
            best_points = _normalize(result.x.reshape(N, 3))
            best_energy = result.fun
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > DEADLINE - 30:
        return best_points
    
    # Try multiple random restarts with L-BFGS-B
    rng = np.random.RandomState(42)
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Generate random initial points
        pts = rng.randn(N, 3)
        pts = _normalize(pts)
        
        x0 = pts.flatten()
        
        try:
            result = minimize(
                obj_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 3000}
            )
            if result.fun < best_energy:
                best_points = _normalize(result.x.reshape(N, 3))
                best_energy = result.fun
        except Exception:
            pass
    
    # Final polish: do a few more iterations from the best point
    if time.time() - start_time < DEADLINE - 10:
        x0 = best_points.flatten()
        try:
            result = minimize(
                obj_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxfun': 10000}
            )
            if result.fun < best_energy:
                best_points = _normalize(result.x.reshape(N, 3))
                best_energy = result.fun
        except Exception:
            pass
    
    return best_points
