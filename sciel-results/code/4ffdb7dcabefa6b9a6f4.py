"""Optimized logarithmic energy minimization for 100 points on the unit sphere."""
import numpy as np
import time
from scipy.optimize import minimize
import math

N = 100

# EVOLVE-BLOCK-START
def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy and its gradient for points on the unit sphere.
    Energy = sum_{i<j} log(1 / ||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    Gradient is with respect to the 3D coordinates (will be projected to tangent space later).
    """
    n = pts.shape[0]
    # Compute pairwise distances
    # dist[i,j] = ||pts[i] - pts[j]||
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
    
    # Avoid log(0) - though points should be distinct
    dist = np.maximum(dist, 1e-15)
    
    # Energy: sum over i<j of log(1/dist) = -sum over i<j of log(dist)
    # Use upper triangle only
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dist[iu])
    energy = -np.sum(log_dists)
    
    # Gradient: d/dx_i of -log(||x_i - x_j||) = -(x_i - x_j) / ||x_i - x_j||^2
    # For each pair (i,j), gradient contribution to x_i is -(x_i - x_j)/dist^2
    # and to x_j is +(x_i - x_j)/dist^2
    # Sum over all pairs
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(i+1, n):
            d = dist[i, j]
            if d < 1e-15:
                continue
            diff_ij = pts[i] - pts[j]
            factor = 1.0 / (d * d)
            grad[i] -= factor * diff_ij
            grad[j] += factor * diff_ij
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to lie on the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _log_energy(pts):
    """Just the energy, no gradient."""
    e, _ = _log_energy_and_grad(pts)
    return e


def _fibonacci_spiral(n):
    """Generate n points on the unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _optimize_from_start(pts0, max_time):
    """
    Optimize from a starting configuration using L-BFGS-B with tangent-space projection.
    Uses a wrapper that projects gradients to the tangent space.
    """
    n = pts0.shape[0]
    # Flatten to 1D for scipy
    x0 = pts0.flatten()
    
    def objective_and_grad(x):
        pts = x.reshape(n, 3)
        pts = _project_to_sphere(pts)
        e, g = _log_energy_and_grad(pts)
        # Project gradient to tangent space: g_tangent = g - (g . x) * x
        g_proj = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        return e, g_proj.flatten()
    
    def objective(x):
        e, _ = objective_and_grad(x)
        return e
    
    def gradient(x):
        _, g = objective_and_grad(x)
        return g
    
    # Use L-BFGS-B
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        best_pts = _project_to_sphere(result.x.reshape(n, 3))
        best_energy = _log_energy(best_pts)
    except Exception:
        best_pts = pts0
        best_energy = _log_energy(pts0)
    
    return best_pts, best_energy


def _refine_high_precision(pts, max_time):
    """
    High-precision refinement pass with tighter tolerances.
    """
    n = pts.shape[0]
    x0 = pts.flatten()
    
    def objective_and_grad(x):
        pts = x.reshape(n, 3)
        pts = _project_to_sphere(pts)
        e, g = _log_energy_and_grad(pts)
        g_proj = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        return e, g_proj.flatten()
    
    def objective(x):
        e, _ = objective_and_grad(x)
        return e
    
    def gradient(x):
        _, g = objective_and_grad(x)
        return g
    
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 100}
        )
        refined_pts = _project_to_sphere(result.x.reshape(n, 3))
        refined_energy = _log_energy(refined_pts)
        
        # Keep the better one
        current_energy = _log_energy(pts)
        if refined_energy < current_energy:
            return refined_pts, refined_energy
        else:
            return pts, current_energy
    except Exception:
        return pts, _log_energy(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with Fibonacci spiral
    best_pts = _fibonacci_spiral(N)
    best_energy = _log_energy(best_pts)
    
    # Multi-start: try several different initializations
    rng = np.random.default_rng(42)
    
    # Generate multiple starting configurations
    starts = []
    starts.append(best_pts.copy())
    
    # Perturbed Fibonacci
    for _ in range(5):
        perturbation = 0.1 * rng.standard_normal((N, 3))
        perturbed = best_pts + perturbation
        perturbed = _project_to_sphere(perturbed)
        starts.append(perturbed)
    
    # Random starts
    for _ in range(5):
        random_pts = rng.standard_normal((N, 3))
        random_pts = _project_to_sphere(random_pts)
        starts.append(random_pts)
    
    # Also try a few more structured starts
    # Spherical coordinates with different patterns
    for k in range(3):
        i = np.arange(N)
        z = np.cos(np.pi * (i + 0.5) / N)
        phi = (2.0 * np.pi * i / N) + k * 0.1
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        starts.append(pts)
    
    # Optimize each start
    for idx, start_pts in enumerate(starts):
        if time.time() > deadline - 30:  # Need at least 30s left for refinement
            break
        
        pts, energy = _optimize_from_start(start_pts, deadline - time.time())
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts
    
    # Final high-precision refinement
    if time.time() < deadline - 10:
        best_pts, best_energy = _refine_high_precision(best_pts, deadline - time.time())
    
    # Ensure all points are on the unit sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    # Final check: ensure no points are too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge points apart
                best_pts[i] += 1e-8 * rng.standard_normal(3)
                best_pts[j] -= 1e-8 * rng.standard_normal(3)
                best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
