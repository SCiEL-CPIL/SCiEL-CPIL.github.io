import numpy as np
import math
import time

N = 100

def _log_energy_and_grad(X):
    """
    Computes the logarithmic energy and its gradient.
    X: (N, 3) array of points on the sphere.
    Returns:
        energy: float
        grad: (N, 3) array
    """
    # Compute pairwise distances
    # X is (N, 3)
    # dist2[i, j] = ||X[i] - X[j]||^2
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2)
    
    # Avoid division by zero for i=j
    np.fill_diagonal(dist2, 1.0)
    
    dist = np.sqrt(dist2)
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # We sum over all i != j and divide by 2
    # log(dist) is -inf where dist=0, but we handled diagonal.
    # We need to be careful with very small distances.
    # The task says "Points closer than 1e-9 ... invalid".
    # So we assume dist > 1e-9.
    
    # Mask for i < j to avoid double counting and self-interaction
    # But for gradient, we need sum over all j != i.
    # grad_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    # Note: log(1/d) = -log(d).
    # d/dx_i (-log ||x_i - x_j||) = - (1/||x_i - x_j||) * (x_i - x_j)/||x_i - x_j|| = -(x_i - x_j)/||x_i - x_j||^2
    # So grad_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    
    # Vectorized gradient calculation
    # diff is (N, N, 3)
    # inv_dist2 is (N, N)
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad[i] = sum_j diff[i, j] * inv_dist2[i, j]
    # diff[i, j] = X[i] - X[j]
    grad = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    # Energy
    # Sum log(1/dist) for i < j
    # We can use the upper triangle
    iu = np.triu_indices(N, k=1)
    energy = np.sum(np.log(1.0 / dist[iu]))
    
    return energy, grad

def _project_to_sphere(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    return X / norms

def _optimize(X0, max_iter=2000, initial_step=1e-2):
    X = X0.copy()
    step_size = initial_step
    best_energy = _log_energy_and_grad(X)[0]
    best_X = X.copy()
    
    for it in range(max_iter):
        energy, grad = _log_energy_and_grad(X)
        
        # If we found a better energy, update best
        if energy < best_energy:
            best_energy = energy
            best_X = X.copy()
            
        # Gradient step
        X_new = X - step_size * grad
        X_new = _project_to_sphere(X_new)
        
        # Check if step improved energy
        energy_new, _ = _log_energy_and_grad(X_new)
        
        if energy_new < energy:
            X = X_new
            # Increase step size slightly if improvement is significant?
            # Or keep constant.
            # Let's try a simple adaptive step
            # step_size = min(step_size * 1.05, 1e-1)
        else:
            # Decrease step size
            step_size *= 0.5
            if step_size < 1e-10:
                break
                
    return best_X, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    best_X = None
    best_E = float('inf')
    
    # Initializations
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    initializations = [X_fib]
    
    # 2. Random Uniform
    for _ in range(5):
        X_rand = np.random.randn(N, 3)
        X_rand = _project_to_sphere(X_rand)
        initializations.append(X_rand)
        
    # 3. Sobol Sequence (if available)
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=3, scramble=True)
        # Generate enough points
        # Sobol generates in [0,1]^d
        # We need N points.
        # Use a larger sample and take first N? Or just generate N.
        # Scramble helps with uniformity.
        points = sampler.random(N)
        # Map to sphere
        z_s = 2.0 * points[:, 0] - 1.0
        r_s = np.sqrt(1.0 - z_s**2)
        phi_s = 2.0 * np.pi * points[:, 1]
        X_sob = np.column_stack([r_s * np.cos(phi_s), r_s * np.sin(phi_s), z_s])
        initializations.append(X_sob)
    except ImportError:
        pass
        
    # Run optimization for each initialization
    for X0 in initializations:
        if time.time() > deadline:
            break
            
        # Run a few variants of step size?
        # Just one robust run per init
        X_opt, E_opt = _optimize(X0, max_iter=3000, initial_step=5e-3)
        
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
            
    if best_X is None:
        return X_fib # Fallback
        
    return best_X
