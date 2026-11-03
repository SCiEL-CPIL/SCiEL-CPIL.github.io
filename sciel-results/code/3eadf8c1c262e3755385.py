"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _energy_and_grad(params):
    """Compute logarithmic energy and gradient for 100 points on unit sphere.
    params: shape (100, 3) array of points (not necessarily normalized)
    """
    x = params / np.linalg.norm(params, axis=1, keepdims=True)
    # Compute pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    log_terms = np.log(dist[mask])
    energy = -np.sum(log_terms)
    
    # Gradient: dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # For the log energy: dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # But wait: E = sum_{i<j} log(1/|x_i-x_j|) = -sum_{i<j} log|xi-xj|
    # dE/dx_i = -sum_{j != i, j>i or j<i} (1/|xi-xj|) * (xi-xj)/|xi-xj|
    #          = -sum_{j != i} (xi-xj)/|xi-xj|^2
    #          = sum_{j != i} (xj-xi)/|xi-xj|^2
    
    grad = np.zeros_like(x)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = dist[i, j]
            if d < 1e-12:
                continue
            grad[i] += (x[j] - x[i]) / (d * d)
    
    # Project gradient to tangent space
    for i in range(N):
        grad[i] -= np.dot(grad[i], x[i]) * x[i]
    
    return energy, grad

def _energy_only(params):
    x = params / np.linalg.norm(params, axis=1, keepdims=True)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    return -np.sum(np.log(dist[mask]))

def _golden_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start from golden spiral
    x0 = _golden_spiral()
    best_params = x0.copy()
    best_energy = _energy_only(x0)
    
    # First optimization pass with BFGS
    res = minimize(_energy_and_grad, x0, method='BFGS', jac=True,
                   options={'maxiter': 200, 'gtol': 1e-10})
    if time.time() > deadline:
        return best_params
    
    if res.success or res.fun < best_energy:
        best_params = res.x.copy()
        best_energy = res.fun
    
    # Second pass with stricter tolerance
    if time.time() < deadline - 30:
        res2 = minimize(_energy_and_grad, best_params, method='BFGS', jac=True,
                        options={'maxiter': 500, 'gtol': 1e-14})
        if time.time() < deadline:
            if res2.success or res2.fun < best_energy:
                best_params = res2.x.copy()
                best_energy = res2.fun
    
    # Third pass - very tight tolerance for final polish
    if time.time() < deadline - 10:
        res3 = minimize(_energy_and_grad, best_params, method='BFGS', jac=True,
                        options={'maxiter': 200, 'gtol': 1e-16})
        if time.time() < deadline:
            if res3.success or res3.fun < best_energy:
                best_params = res3.x.copy()
                best_energy = res3.fun
    
    # Final normalization
    norms = np.linalg.norm(best_params, axis=1, keepdims=True)
    result = best_params / norms
    
    # Validate
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(result[i] - result[j])
            if d < 1e-9:
                # Perturb slightly
                result[i] += np.random.randn(3) * 1e-6
                result[i] /= np.linalg.norm(result[i])
    
    return result
# EVOLVE-BLOCK-END
