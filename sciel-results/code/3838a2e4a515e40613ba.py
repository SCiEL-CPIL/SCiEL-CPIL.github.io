"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _log_energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dist, 1.0)
    # We only want upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Clamp to avoid log of very small numbers
    d = np.maximum(d, 1e-15)
    # Energy is sum of log(1/d) = -sum log(d)
    # Use math.fsum for precision as per task description
    log_vals = np.log(d)
    return -math.fsum(list(log_vals))

def _gradient_log_energy(points):
    """Compute gradient of the logarithmic energy with respect to points.
    
    Energy E = sum_{i<j} -log(|x_i - x_j|)
    dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(points)
    grad = np.zeros_like(points)
    # Vectorized computation for speed
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Gradient term: (x_i - x_j) / |x_i - x_j|^2
    grad_terms = diff / dist_sq[:, :, np.newaxis]
    
    # Sum over j for each i
    grad = np.sum(grad_terms, axis=1)
    return grad

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return points / norms

def construct():
    # Start with a good initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize to be safe
    points = _project_to_sphere(points)
    
    # Use L-BFGS-B with constraint that points stay on sphere
    # We'll use a penalty method or reparameterize
    
    # Define objective and gradient in terms of flattened array
    def objective(x_flat):
        pts = x_flat.reshape(N, 3)
        pts = _project_to_sphere(pts)
        return _log_energy(pts)
    
    def gradient(x_flat):
        pts = x_flat.reshape(N, 3)
        pts = _project_to_sphere(pts)
        grad = _gradient_log_energy(pts)
        # Project gradient to tangent space: g - (g·x)x
        grad_proj = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        return grad_proj.flatten()
    
    # Use L-BFGS-B optimization
    x0 = points.flatten()
    
    # Stage 1: L-BFGS with moderate precision
    result1 = minimize(
        objective, 
        x0, 
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12}
    )
    
    points = _project_to_sphere(result1.x.reshape(N, 3))
    
    # Stage 2: Fine-tune with smaller tolerance
    x0 = points.flatten()
    result2 = minimize(
        objective, 
        x0, 
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-14}
    )
    
    points = _project_to_sphere(result2.x.reshape(N, 3))
    
    # Final refinement with very small gradient steps
    lr = 1e-8
    current_energy = _log_energy(points)
    for it in range(50):
        grad = _gradient_log_energy(points)
        # Project gradient to tangent space
        grad_proj = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        new_points = _project_to_sphere(points - lr * grad_proj)
        new_energy = _log_energy(new_points)
        if new_energy < current_energy:
            points = new_points
            current_energy = new_energy
        else:
            lr *= 0.5
            if lr < 1e-15:
                break
    
    # Final normalization
    points = _project_to_sphere(points)
    
    # Check for close points
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Perturb if necessary
        rng = np.random.default_rng(42)
        points = points + rng.normal(0, 1e-6, points.shape)
        points = _project_to_sphere(points)

    return points
# EVOLVE-BLOCK-END
