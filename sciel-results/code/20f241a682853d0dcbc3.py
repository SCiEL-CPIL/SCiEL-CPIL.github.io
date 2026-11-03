"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import time
import math
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _fibonacci_sphere(n, seed=0):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _log_energy_and_grad(points):
    """
    Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) and its gradient.
    points: (N, 3) array on unit sphere.
    Returns (energy, grad) where grad is the gradient w.r.t. points (N,3).
    """
    n = points.shape[0]
    # Compute all pairwise distances
    # diff[i,j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # log(1/d) = -log(d)
    # Energy = sum_{i<j} -log(dist[i,j])
    # Use upper triangle only
    iu = np.triu_indices(n, k=1)
    energy = np.sum(-np.log(dist[iu]))
    
    # Gradient: d/dx_i [ -log|x_i - x_j| ] = -(x_i - x_j) / |x_i - x_j|^2
    # For each point i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Note: we need to be careful with the factor
    # d/dx_i log(1/|x_i - x_j|) = d/dx_i (-log|x_i - x_j|) = -(x_i - x_j)/|x_i - x_j|^2
    # So grad_i = sum_{j != i} -(x_i - x_j)/|x_i - x_j|^2 = sum_{j != i} (x_j - x_i)/|x_i - x_j|^2
    
    # dist_sq = dist^2
    dist_sq = dist * dist
    np.fill_diagonal(dist_sq, 1.0)
    
    # We want grad[i] = sum_j (points[j] - points[i]) / dist_sq[i,j]
    # This is like: for each i, sum over j of (points[j] - points[i]) / dist_sq[i,j]
    # = sum_j points[j]/dist_sq[i,j] - points[i] * sum_j 1/dist_sq[i,j]
    
    # Let's compute it directly
    # inv_dist_sq = 1/dist_sq, with diagonal = 0
    inv_dist_sq = np.zeros_like(dist_sq)
    np.fill_diagonal(dist_sq, 1.0)  # ensure no div by zero
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = sum_j (points[j] - points[i]) * inv_dist_sq[i,j]
    # = (points @ inv_dist_sq) - points[i] * sum_j inv_dist_sq[i,j]
    # points @ inv_dist_sq: (n,3) @ (n,n) -> (n,3), where result[i] = sum_j points[j] * inv_dist_sq[j,i]
    # But inv_dist_sq is symmetric, so inv_dist_sq[j,i] = inv_dist_sq[i,j]
    # So result[i] = sum_j points[j] * inv_dist_sq[i,j]
    sum_points = points @ inv_dist_sq  # (n, 3)
    sum_inv = np.sum(inv_dist_sq, axis=1)  # (n,)
    grad = sum_points - points * sum_inv[:, np.newaxis]
    
    return energy, grad

def _project_to_sphere(points):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    best_points = _fibonacci_sphere(N)
    best_energy, _ = _log_energy_and_grad(best_points)
    
    # Multiple restarts with different initializations
    rng = np.random.default_rng(42)
    
    n_restarts = 0
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        if restart == 0:
            # First restart uses Fibonacci spiral
            x0 = _fibonacci_sphere(N)
        else:
            # Subsequent restarts use perturbed Fibonacci or random
            if restart % 2 == 0:
                # Perturbed Fibonacci
                x0 = _fibonacci_sphere(N)
                noise = rng.normal(0, 0.1, x0.shape)
                x0 = _project_to_sphere(x0 + noise)
            else:
                # Random points
                x0 = rng.normal(0, 1, (N, 3))
                x0 = _project_to_sphere(x0)
        
        n_restarts += 1
        
        try:
            # Use L-BFGS-B with the log energy
            result = minimize(
                _log_energy_and_grad,
                x0.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < best_energy:
                x_opt = _project_to_sphere(result.x.reshape(N, 3))
                e_opt, _ = _log_energy_and_grad(x_opt)
                
                if e_opt < best_energy:
                    best_energy = e_opt
                    best_points = x_opt
                    
        except Exception:
            continue
    
    # Final polish: high-precision refinement
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                _log_energy_and_grad,
                best_points.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.fun < best_energy:
                x_opt = _project_to_sphere(result.x.reshape(N, 3))
                e_opt, _ = _log_energy_and_grad(x_opt)
                if e_opt < best_energy:
                    best_energy = e_opt
                    best_points = x_opt
        except Exception:
            pass
    
    # Ensure output is valid
    norms = np.linalg.norm(best_points, axis=1)
    best_points = best_points / norms[:, np.newaxis]
    
    return best_points
# EVOLVE-BLOCK-END
