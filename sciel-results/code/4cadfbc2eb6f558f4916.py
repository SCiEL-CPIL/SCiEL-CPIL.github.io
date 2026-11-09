"""Optimized logarithmic energy minimization for 150 points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 115.0  # safety margin before 120s limit


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    pts: (N, 3) array of unit vectors
    Returns: (energy, grad) where grad is (N, 3) array
    """
    n = len(pts)
    # Compute pairwise distances
    # diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    # Avoid self-pairs (diagonal)
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(dist2)
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Clip to avoid log(0)
    d = np.maximum(d, 1e-15)
    energy = -np.sum(np.log(d))
    
    # Gradient: d/dp_i [sum_{j!=i} -log(|p_i - p_j|)]
    # = sum_{j!=i} (p_i - p_j) / |p_i - p_j|^2
    # Note: this is the gradient in R^3; for sphere-constrained optimization
    # we project, but L-BFGS-B works on the unconstrained R^3 and we re-normalize
    # Actually, let's use the approach where we optimize in R^3 and project.
    # The gradient of the energy with respect to p_i in R^3 is:
    # g_i = sum_{j!=i} (p_i - p_j) / |p_i - p_j|^2
    
    # Compute gradient efficiently
    # For each pair (i,j), contribution to grad[i] is (p_i - p_j)/dist^2
    # and to grad[j] is (p_j - p_i)/dist^2 = -(p_i - p_j)/dist^2
    
    # Build gradient
    grad = np.zeros_like(pts)
    # dist2[i,j] = |p_i - p_j|^2
    # We need (p_i - p_j) / dist2[i,j] for i != j
    # diff[i,j] = p_i - p_j
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad[i] = sum_j diff[i,j] * inv_dist2[i,j]
    # This is a matrix product: grad = diff @ inv_dist2 (but diff is 3D)
    # grad[i] = sum_j (p_i - p_j) / |p_i - p_j|^2
    # = sum_j diff[i,j] * inv_dist2[i,j]
    
    # Using einsum: grad[i,k] = sum_j diff[i,j,k] * inv_dist2[i,j]
    grad = np.einsum('ijk,ij->ik', diff, inv_dist2)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def construct():
    start_time = time.time()
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_energy = _energy_and_grad(pts)[0]
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 5:
        return best_pts
    
    # Optimization: minimize energy over R^3, project to sphere after each step
    # Use L-BFGS-B with a custom objective that projects
    
    def objective(x):
        pts_proj = _project_to_sphere(x.reshape(N, 3))
        e, _ = _energy_and_grad(pts_proj)
        return e
    
    def gradient(x):
        pts_proj = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(pts_proj)
        # Project gradient to tangent space of sphere
        # For point p on sphere, tangent gradient = g - (g·p) * p
        grad_tangent = g - np.sum(g * pts_proj, axis=1, keepdims=True) * pts_proj
        return grad_tangent.ravel()
    
    # Run L-BFGS-B
    try:
        result = minimize(
            objective,
            pts.ravel(),
            jac=gradient,
            method='L-BFGS-B',
            options={
                'maxiter': 200,
                'maxfun': 200,
                'ftol': 1e-12,
                'gtol': 1e-10,
                'maxls': 50
            }
        )
        
        if result.success or result.fun < best_energy:
            new_pts = _project_to_sphere(result.x.reshape(N, 3))
            new_energy = _energy_and_grad(new_pts)[0]
            if new_energy < best_energy:
                best_pts = new_pts
                best_energy = new_energy
    except Exception:
        pass
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 10:
        return best_pts
    
    # Refinement: run additional iterations with tighter tolerances
    try:
        result2 = minimize(
            objective,
            best_pts.ravel(),
            jac=gradient,
            method='L-BFGS-B',
            options={
                'maxiter': 100,
                'maxfun': 100,
                'ftol': 1e-14,
                'gtol': 1e-12,
                'maxls': 50
            }
        )
        
        if result2.success or result2.fun < best_energy:
            new_pts = _project_to_sphere(result2.x.reshape(N, 3))
            new_energy = _energy_and_grad(new_pts)[0]
            if new_energy < best_energy:
                best_pts = new_pts
                best_energy = new_energy
    except Exception:
        pass
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 5:
        return best_pts
    
    # Final polish: use scipy's minimize with even tighter settings
    try:
        result3 = minimize(
            objective,
            best_pts.ravel(),
            jac=gradient,
            method='L-BFGS-B',
            options={
                'maxiter': 50,
                'maxfun': 50,
                'ftol': 1e-15,
                'gtol': 1e-13,
                'maxls': 30
            }
        )
        
        if result3.success or result3.fun < best_energy:
            new_pts = _project_to_sphere(result3.x.reshape(N, 3))
            new_energy = _energy_and_grad(new_pts)[0]
            if new_energy < best_energy:
                best_pts = new_pts
                best_energy = new_energy
    except Exception:
        pass
    
    # Ensure all points are on unit sphere and non-degenerate
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return best_pts
