"""Optimize 150 points on unit sphere to minimize logarithmic energy."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    # Compute pairwise distances
    # diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Energy: -sum_{i<j} log(dists[i,j])
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dists[iu]))
    
    # Gradient: for each i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j != i} diff[i,j] / dists[i,j]^2
    # diff[i,j] / dists[i,j]^2, with diagonal excluded
    inv_dists_sq = np.zeros_like(dists)
    np.fill_diagonal(inv_dists_sq, 0.0)
    inv_dists_sq[iu] = 1.0 / (dists[iu] ** 2)
    inv_dists_sq[(iu[1], iu[0])] = 1.0 / (dists[iu] ** 2)
    
    # grad[i] = sum_j diff[i,j] * inv_dists_sq[i,j]
    grad = np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_sphere(n):
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    
    best_pts = _fibonacci_sphere(N)
    best_energy = _energy_and_grad(best_pts)[0]
    
    # Try multiple random initializations and optimize each
    np.random.seed(42)
    
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Generate random initial points
        if restart == 0:
            pts = _fibonacci_sphere(N)
        else:
            # Random points on sphere
            pts = np.random.randn(N, 3)
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Use L-BFGS-B with penalty for sphere constraint
        def objective(x):
            if time.time() > deadline - 2.0:
                return 1e10, None
            pts = x.reshape(N, 3)
            # Penalty for deviation from unit sphere
            norms = np.linalg.norm(pts, axis=1)
            penalty = 100.0 * np.sum((norms - 1.0) ** 2)
            energy, grad = _energy_and_grad(pts)
            return energy + penalty, None
        
        def gradient(x):
            if time.time() > deadline - 2.0:
                return np.zeros_like(x)
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            energy, grad = _energy_and_grad(pts)
            # Add gradient of penalty
            penalty_grad = 200.0 * (norms - 1.0) * pts / norms
            return (grad + penalty_grad).flatten()
        
        x0 = pts.flatten()
        
        try:
            result = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if result.success or result.fun < best_energy:
                new_pts = result.x.reshape(N, 3)
                new_pts = _project_to_sphere(new_pts)
                new_energy = _energy_and_grad(new_pts)[0]
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pts = new_pts
                    
        except Exception:
            continue
    
    # Final polish: high-precision optimization on best point
    if time.time() < deadline - 10.0:
        def polish_objective(x):
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1)
            penalty = 1000.0 * np.sum((norms - 1.0) ** 2)
            energy, grad = _energy_and_grad(pts)
            return energy + penalty, None
        
        def polish_gradient(x):
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            energy, grad = _energy_and_grad(pts)
            penalty_grad = 2000.0 * (norms - 1.0) * pts / norms
            return (grad + penalty_grad).flatten()
        
        x0 = best_pts.flatten()
        try:
            result = minimize(
                polish_objective,
                x0,
                jac=polish_gradient,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            new_pts = result.x.reshape(N, 3)
            new_pts = _project_to_sphere(new_pts)
            new_energy = _energy_and_grad(new_pts)[0]
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
        except Exception:
            pass
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Check for near-duplicate points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Perturb slightly
                best_pts[j] += 1e-8 * np.random.randn(3)
                best_pts[j] /= np.linalg.norm(best_pts[j])
    
    return best_pts
