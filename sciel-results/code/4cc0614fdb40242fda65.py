"""Coupled variational optimization: charged particles in tunable spherical harmonic potential."""
import numpy as np
from scipy.optimize import minimize
import math

N = 100


def _log_energy(points):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    d_sq = dists_sq[iu]
    d_sq = np.maximum(d_sq, 1e-30)
    d = np.sqrt(d_sq)
    return -np.sum(np.log(d))


def _log_energy_grad(points):
    """Vectorized gradient of log-energy. Returns (N,3) array."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    # grad[i] = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # = -sum_{j!=i} diff[i,j] / dists_sq[i,j]
    inv_d2 = np.zeros_like(dists_sq)
    mask = dists_sq > 1e-30
    inv_d2[mask] = 1.0 / dists_sq[mask]
    np.fill_diagonal(inv_d2, 0.0)
    grad = -np.sum(diff * inv_d2[:, np.newaxis], axis=1)
    return grad


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _fibonacci_initialization():
    """Fibonacci spiral on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_initialization(seed):
    """Random points on sphere."""
    np.random.seed(seed)
    pts = np.random.randn(N, 3)
    return _project_to_sphere(pts)


def _optimize_from_start(points0, max_iter=800, tol=1e-10):
    """Optimize from a given starting configuration using projected gradient descent."""
    points = _project_to_sphere(points0.copy())
    best_points = points.copy()
    best_energy = _log_energy(points)
    
    lr = 0.05
    patience = 0
    max_patience = 50
    
    for it in range(max_iter):
        # Compute energy and gradient
        energy = _log_energy(points)
        grad = _log_energy_grad(points)
        
        # Project gradient to tangent space
        grad_tangent = np.zeros_like(grad)
        for i in range(N):
            radial = np.dot(grad[i], points[i])
            grad_tangent[i] = grad[i] - radial * points[i]
        
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < tol:
            break
        
        # Adaptive step: normalize gradient for stable steps
        if grad_norm > 1e-12:
            direction = grad_tangent / grad_norm
        else:
            break
        
        # Try step with current learning rate
        step_size = lr
        for attempt in range(10):
            points_new = _project_to_sphere(points - step_size * grad_tangent)
            new_energy = _log_energy(points_new)
            
            if new_energy < energy - 1e-15:
                points = points_new
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_points = points.copy()
                    patience = 0
                else:
                    patience += 1
                break
            else:
                step_size *= 0.5
        
        # Update learning rate
        if patience < max_patience:
            lr = min(lr * 1.03, 0.5)
        else:
            lr = max(lr * 0.7, 0.001)
            patience = 0
        
        # If stuck, try a small random perturbation
        if patience >= max_patience and it < max_iter - 100:
            np.random.seed(it)
            perturbation = 0.01 * np.random.randn(N, 3)
            points_pert = _project_to_sphere(points + perturbation)
            pert_energy = _log_energy(points_pert)
            if pert_energy < energy:
                points = points_pert
                energy = pert_energy
                patience = 0
                lr = 0.05
    
    return best_points, best_energy


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    
    best_points = None
    best_energy = np.inf
    
    # Multiple restarts with different initializations
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_fibonacci_initialization())
    
    # 2. Fibonacci with small perturbation
    np.random.seed(42)
    fib_pert = _fibonacci_initialization() + 0.02 * np.random.randn(N, 3)
    initializations.append(_project_to_sphere(fib_pert))
    
    # 3. Random initialization
    initializations.append(_random_initialization(123))
    
    # 4. Another random initialization
    initializations.append(_random_initialization(456))
    
    # 5. Fibonacci with larger perturbation
    np.random.seed(789)
    fib_pert2 = _fibonacci_initialization() + 0.05 * np.random.randn(N, 3)
    initializations.append(_project_to_sphere(fib_pert2))
    
    for init_idx, init_pts in enumerate(initializations):
        # Allocate more iterations for better initializations
        max_iter = 1000 if init_idx < 2 else 800
        pts, energy = _optimize_from_start(init_pts, max_iter=max_iter)
        
        if energy < best_energy:
            best_energy = energy
            best_points = pts.copy()
    
    # Final polishing phase with very small steps
    if best_points is not None:
        points = best_points.copy()
        for it in range(300):
            energy = _log_energy(points)
            grad = _log_energy_grad(points)
            
            grad_tangent = np.zeros_like(grad)
            for i in range(N):
                radial = np.dot(grad[i], points[i])
                grad_tangent[i] = grad[i] - radial * points[i]
            
            grad_norm = np.linalg.norm(grad_tangent)
            if grad_norm < 1e-12:
                break
            
            # Very small steps for polishing
            step_size = 0.001
            for attempt in range(20):
                points_new = _project_to_sphere(points - step_size * grad_tangent)
                new_energy = _log_energy(points_new)
                
                if new_energy < energy - 1e-15:
                    points = points_new
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_points = points.copy()
                    break
                else:
                    step_size *= 0.5
        
        best_points = points
    
    # Ensure valid output
    result = _project_to_sphere(best_points)
    
    # Validate: no points too close, all finite
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(result[i] - result[j])
            if d < 1e-9:
                np.random.seed(i * 1000 + j)
                result[j] += 1e-8 * np.random.randn(3)
                result[j] = result[j] / np.linalg.norm(result[j])
    
    if not np.all(np.isfinite(result)):
        result = _fibonacci_initialization()
    
    return result
