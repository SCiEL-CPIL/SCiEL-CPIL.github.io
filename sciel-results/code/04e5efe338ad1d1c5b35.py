import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_and_grad_fast(points):
    """
    Compute logarithmic energy and gradient efficiently.
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    grad_i E = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    # Compute pairwise distances using broadcasting but more memory-efficient
    # diff[i,j] = x_i - x_j
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    
    # Energy: -sum_{i<j} log(dists[i,j])
    # Use upper triangle only
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dists[iu]))
    
    # Gradient: grad_i = -sum_{j!=i} diff[i,j] / dists[i,j]^2
    # diff[i,j] = x_i - x_j
    inv_dists_sq = 1.0 / dists_sq
    np.fill_diagonal(inv_dists_sq, 0.0)
    # grad_i = -sum_j diff[i,j] * inv_dists_sq[i,j]
    grads = -np.einsum('ijk,ij->ik', diff, inv_dists_sq)
    
    return energy, grads

def _project_to_sphere(points):
    """Project points to unit sphere."""
    norms = np.linalg.norm(points, axis=1)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return points / norms[:, np.newaxis]

def _tangent_gradient(points, grads):
    """Project gradient to tangent space of sphere at each point."""
    # Tangent space: remove radial component
    # For point x on unit sphere, tangent projection: g - (g·x)x
    dot = np.sum(grads * points, axis=1)
    return grads - dot[:, np.newaxis] * points

def _fibonacci_spiral(n=N):
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n=N, seed=None):
    """Generate random points on sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    v = rng.randn(n, 3)
    norms = np.linalg.norm(v, axis=1)
    return v / norms[:, np.newaxis]

def _gradient_descent(points, start_time, time_limit, lr_init=1e-3, max_iters=50000):
    """
    Gradient descent with momentum and adaptive learning rate.
    Returns best points found.
    """
    points = points.copy()
    best_points = points.copy()
    
    current_energy, _ = _energy_and_grad_fast(points)
    best_energy = current_energy
    
    momentum = 0.8
    v = np.zeros_like(points)
    lr = lr_init
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit - 10.0:
            break
        
        energy, grads = _energy_and_grad_fast(points)
        if not np.isfinite(energy):
            break
        
        # Project gradient to tangent space
        tgrads = _tangent_gradient(points, grads)
        
        # Update momentum
        v = momentum * v - lr * tgrads
        
        # Try update with backtracking line search
        best_lr = lr
        best_new_energy = energy
        best_new_points = points.copy()
        
        for backtrack in range(10):
            trial_points = points + v
            trial_points = _project_to_sphere(trial_points)
            trial_energy, _ = _energy_and_grad_fast(trial_points)
            
            if np.isfinite(trial_energy) and trial_energy < energy - 1e-10 * abs(energy):
                best_lr = lr
                best_new_energy = trial_energy
                best_new_points = trial_points
                break
            else:
                lr *= 0.5
        
        if best_new_energy < energy - 1e-12:
            points = best_new_points
            current_energy = best_new_energy
            lr = best_lr * 1.1  # Slightly increase if step worked
            lr = min(lr, 5e-2)
        else:
            lr *= 0.5
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_points = points.copy()
        
        # Convergence check
        if it > 20:
            grad_norm = np.linalg.norm(tgrads)
            if grad_norm < 1e-8:
                break
    
    return best_points, best_energy

def _lbfgs_polish(points, start_time, time_limit):
    """
    Final polish with L-BFGS-B on tangent space.
    """
    if time.time() - start_time > time_limit - 20.0:
        return points
    
    def objective_and_grad(flat_x):
        if time.time() - start_time > time_limit - 10.0:
            return 0.0, np.zeros_like(flat_x)
        pts = flat_x.reshape(N, 3)
        pts = _project_to_sphere(pts)
        e, g = _energy_and_grad_fast(pts)
        # Project gradient to tangent space
        tg = _tangent_gradient(pts, g)
        return e, tg.flatten()
    
    x0 = points.flatten()
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': 500,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50
            }
        )
        if result.success or result.fun < 0:
            new_points = _project_to_sphere(result.x.reshape(N, 3))
            new_energy, _ = _energy_and_grad_fast(new_points)
            old_energy, _ = _energy_and_grad_fast(points)
            if new_energy < old_energy:
                return new_points
    except Exception:
        pass
    
    return points

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    # Multiple starting points
    starts = []
    
    # Fibonacci spiral
    starts.append(_fibonacci_spiral())
    
    # Random starts with different seeds
    for seed in [42, 123, 456]:
        starts.append(_random_points(seed=seed))
    
    # Jittered Fibonacci
    fib = _fibonacci_spiral()
    jitter = np.random.RandomState(7).randn(N, 3) * 0.1
    starts.append(_project_to_sphere(fib + jitter))
    
    best_points = None
    best_energy = np.inf
    
    for idx, start in enumerate(starts):
        if time.time() - start_time > time_limit - 25.0:
            break
        
        # Allocate time budget for each start
        remaining = time_limit - 25.0 - (time.time() - start_time)
        if remaining < 10.0:
            break
        
        # Gradient descent phase
        gd_points, gd_energy = _gradient_descent(
            start, start_time, time_limit,
            lr_init=1e-3,
            max_iters=30000
        )
        
        # L-BFGS polish
        polished_points = _lbfgs_polish(gd_points, start_time, time_limit)
        
        final_energy, _ = _energy_and_grad_fast(polished_points)
        
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = polished_points
    
    if best_points is None:
        best_points = _fibonacci_spiral()
    
    return best_points
