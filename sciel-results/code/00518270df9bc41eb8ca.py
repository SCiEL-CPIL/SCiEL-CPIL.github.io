import numpy as np
import time
from scipy.optimize import minimize
import math

N = 150

def _fibonacci_sphere(n, seed=0):
    """Generate n points on the unit sphere using Fibonacci spiral."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    # Add small random perturbation for diversity
    if seed > 0:
        i = i + rng.uniform(-0.1, 0.1, n)
        i = np.clip(i, 0.5, n - 0.5)
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=0):
    """Generate n random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _energy_and_gradient(X):
    """Compute logarithmic energy and its gradient for points on unit sphere.
    
    Energy: sum_{i<j} log(1/||x_i - x_j||)
    Gradient with respect to x_k: -sum_{j!=k} (x_k - x_j) / ||x_k - x_j||^2
    """
    n = X.shape[0]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-18)
    dist = np.sqrt(dist_sq)
    
    # Mask for i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    energy_val = -np.sum(np.log(dist[mask]))
    
    # Gradient
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy_val, grad

def _project_to_sphere(X):
    """Project points to unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _optimize_from_initial(X0, max_time=None):
    """Optimize starting from X0 using L-BFGS-B with penalty for sphere constraint."""
    X = _project_to_sphere(X0.copy())
    n = X.shape[0]
    
    def objective_and_grad(flat_x):
        X = flat_x.reshape(n, 3)
        e, g = _energy_and_gradient(X)
        # Penalty for being off the sphere
        norms = np.linalg.norm(X, axis=1)
        penalty = 1e6 * np.sum((norms - 1.0) ** 2)
        # Gradient of penalty
        grad_penalty = 2e6 * (norms - 1.0) * X / np.maximum(norms, 1e-15)
        return e + penalty, (g + grad_penalty).flatten()
    
    x0 = X.flatten()
    
    if max_time is not None:
        start = time.time()
        # Use L-BFGS-B with time check
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
    else:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
    
    X_opt = _project_to_sphere(result.x.reshape(n, 3))
    return X_opt

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    best_X = None
    best_energy = np.inf
    
    # Try multiple initializations
    initializations = []
    
    # Fibonacci spiral (standard)
    initializations.append(_fibonacci_sphere(N, seed=0))
    
    # Fibonacci with perturbations
    for seed in [1, 2, 3]:
        initializations.append(_fibonacci_sphere(N, seed=seed))
    
    # Random initializations
    for seed in [42, 123, 456, 789]:
        initializations.append(_random_sphere(N, seed=seed))
    
    for idx, X0 in enumerate(initializations):
        if time.time() > deadline - 30:
            break
        
        try:
            X_opt = _optimize_from_initial(X0)
            e, _ = _energy_and_gradient(X_opt)
            
            if e < best_energy:
                best_energy = e
                best_X = X_opt.copy()
        except Exception:
            continue
    
    # If we still have time, do a final polish on the best result
    if best_X is not None and time.time() < deadline - 10:
        try:
            X_polished = _optimize_from_initial(best_X)
            e, _ = _energy_and_gradient(X_polished)
            if e < best_energy:
                best_energy = e
                best_X = X_polished.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if best_X is None:
        best_X = _fibonacci_sphere(N, seed=0)
    
    # Final projection to ensure unit norm
    best_X = _project_to_sphere(best_X)
    
    return best_X
