"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the golden angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _energy_and_grad(x):
    """
    Compute the logarithmic energy and its gradient for points on the unit sphere.
    x: (N, 3) array of unit vectors.
    Energy: sum_{i<j} log(1 / ||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    Gradient (in ambient R^3): grad_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    Note: we project to the tangent space by subtracting (grad_i . x_i) * x_i.
    """
    n = x.shape[0]
    # Compute pairwise distances
    # dist[i,j] = ||x_i - x_j||
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Avoid division by zero on the diagonal
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of -log(dist_ij)
    # We want sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient in ambient space:
    # grad_ambient_i = sum_{j != i} (x_i - x_j) / dist_ij^2
    # Note: d/dx_i [ -log(||x_i - x_j||) ] = -(x_i - x_j)/||x_i - x_j||^2
    # Wait: energy = -sum log(dist). d/dx_i (-log dist_ij) = - (1/dist_ij) * (x_i - x_j)/dist_ij = -(x_i - x_j)/dist_ij^2
    # So grad_ambient_i = - sum_{j != i} (x_i - x_j) / dist_ij^2
    
    # inv_dist_sq: (N, N), diagonal set to 0
    inv_dist_sq = np.zeros_like(dist_sq)
    np.fill_diagonal(inv_dist_sq, 0.0)
    inv_dist_sq[dist_sq > 0] = 1.0 / dist_sq[dist_sq > 0]
    
    # grad_ambient[i] = - sum_j (x_i - x_j) * inv_dist_sq[i,j]
    # = - sum_j x_i * inv_dist_sq[i,j] + sum_j x_j * inv_dist_sq[i,j]
    # = -x_i * sum_j inv_dist_sq[i,j] + sum_j x_j * inv_dist_sq[i,j]
    
    row_sums = np.sum(inv_dist_sq, axis=1)  # (N,)
    # term1 = -x * row_sums[:, np.newaxis]
    # term2 = x.T @ inv_dist_sq  -> (3, N) @ (N, N) = (3, N), then transpose to (N, 3)
    term2 = (x.T @ inv_dist_sq).T  # (N, 3)
    grad_ambient = -x * row_sums[:, np.newaxis] + term2
    
    # Project to tangent space: grad_tangent = grad_ambient - (grad_ambient . x) * x
    dot_products = np.sum(grad_ambient * x, axis=1)  # (N,)
    grad_tangent = grad_ambient - dot_products[:, np.newaxis] * x
    
    return energy, grad_tangent

def _project_to_sphere(x):
    """Project points back onto the unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return x / norms

def _optimize(x0, max_iters=5000, time_limit=10.0, verbose=False):
    """
    Optimize the logarithmic energy on the sphere using projected gradient descent
    with adaptive step size (backtracking line search).
    """
    x = x0.copy()
    best_x = x.copy()
    best_energy = _energy_and_grad(x)[0]
    
    start_time = time.time()
    lr = 0.01  # initial learning rate
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        energy, grad = _energy_and_grad(x)
        
        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()
        
        # Backtracking line search
        # We move in the negative gradient direction (tangent)
        step = -lr * grad
        
        # Try to find a step that decreases energy
        alpha = 1.0
        for _ in range(20):
            x_trial = x + alpha * step
            x_trial = _project_to_sphere(x_trial)
            
            # Check for very close points
            diff = x_trial[:, np.newaxis, :] - x_trial[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            np.fill_diagonal(dist_sq, 1.0)
            if np.min(dist_sq) < 1e-18:
                alpha *= 0.5
                continue
            
            e_trial = _energy_and_grad(x_trial)[0]
            
            if e_trial < energy - 1e-10 * alpha * np.sum(grad * grad):
                break
            alpha *= 0.5
        
        x = x_trial
        x = _project_to_sphere(x)
        
        # Adapt learning rate
        if it % 100 == 0 and it > 0:
            # Simple adaptation: if progress is slow, reduce lr; if fast, keep
            pass
        
    return best_x, best_energy

def construct():
    """
    Construct 100 points on the unit sphere to minimize logarithmic energy.
    Uses multiple restarts from different initial configurations with gradient-based optimization.
    """
    total_time_limit = 110.0  # Safety margin before 120s kill
    start_time = time.time()
    
    # Initial configuration: Fibonacci sphere
    x_fib = _fibonacci_sphere(N)
    
    best_x = x_fib.copy()
    best_energy = _energy_and_grad(x_fib)[0]
    
    # Try optimizing from the Fibonacci start
    x_opt, e_opt = _optimize(x_fib, max_iters=5000, time_limit=30.0)
    if e_opt < best_energy:
        best_energy = e_opt
        best_x = x_opt.copy()
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > total_time_limit - 20.0:
        return _project_to_sphere(best_x)
    
    # Additional restarts with perturbed Fibonacci sphere
    np.random.seed(42)
    for restart in range(5):
        elapsed = time.time() - start_time
        if elapsed > total_time_limit - 15.0:
            break
        
        # Create a perturbed version of the Fibonacci sphere
        noise_scale = 0.1 * (restart + 1)
        x_pert = x_fib + noise_scale * np.random.randn(N, 3)
        x_pert = _project_to_sphere(x_pert)
        
        x_opt, e_opt = _optimize(x_pert, max_iters=3000, time_limit=15.0)
        if e_opt < best_energy:
            best_energy = e_opt
            best_x = x_opt.copy()
    
    # Final polish: more iterations on the best candidate
    elapsed = time.time() - start_time
    if elapsed < total_time_limit - 5.0:
        x_opt, e_opt = _optimize(best_x, max_iters=5000, time_limit=total_time_limit - 5.0 - elapsed)
        if e_opt < best_energy:
            best_energy = e_opt
            best_x = x_opt.copy()
    
    # Ensure valid output
    result = _project_to_sphere(best_x)
    # Verify no points are too close
    diff = result[:, np.newaxis, :] - result[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    if np.min(dist_sq) < 1e-18:
        # Fallback to Fibonacci sphere
        result = x_fib.copy()
    
    return result
# EVOLVE-BLOCK-END
