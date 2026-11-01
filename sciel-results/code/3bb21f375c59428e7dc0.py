"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    # Compute pairwise distances
    # pts is (n, 3)
    # dist^2_{ij} = |x_i|^2 + |x_j|^2 - 2 x_i . x_j
    # Since points are on unit sphere, |x_i|^2 = 1
    # dist_{ij} = sqrt(2 - 2 * x_i . x_j)
    dots = np.dot(pts, pts.T)
    # dist^2 = 2 - 2*dots, but clip to avoid negative due to numerical issues
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)  # avoid log(0) on diagonal
    dist_sq = np.maximum(dist_sq, 1e-15)
    dist = np.sqrt(dist_sq)
    # log(1/d) = -log(d)
    log_inv_dist = -np.log(dist)
    np.fill_diagonal(log_inv_dist, 0.0)
    # Sum over i < j
    triu_indices = np.triu_indices(n, k=1)
    energy = math.fsum(log_inv_dist[triu_indices].tolist())
    return energy


def _project_to_sphere(pts):
    """Normalize each point to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def construct():
    """Construct 150 points on the unit sphere minimizing logarithmic energy."""
    # Start with a good initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Use L-BFGS-B to optimize the energy
    # We'll work in Cartesian coordinates and project to sphere after each step
    # Actually, let's use a simpler approach: optimize directly with constraints
    
    def energy_and_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        pts = pts / norms
        
        # Compute energy
        dots = np.dot(pts, pts.T)
        dist_sq = 2.0 - 2.0 * dots
        np.fill_diagonal(dist_sq, 1.0)
        dist_sq = np.maximum(dist_sq, 1e-15)
        dist = np.sqrt(dist_sq)
        log_inv_dist = -np.log(dist)
        np.fill_diagonal(log_inv_dist, 0.0)
        triu_indices = np.triu_indices(N, k=1)
        energy = np.sum(log_inv_dist[triu_indices])
        
        # Compute gradient
        # E = sum_{i<j} -log(|x_i - x_j|)
        # dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
        # Let's compute this efficiently
        grad = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                d = np.linalg.norm(diff)
                if d < 1e-15:
                    continue
                grad[i] += diff / (d * d)
        
        # Project gradient to tangent space: grad_tan = grad - (grad . x_i) * x_i
        grad_proj = np.zeros_like(grad)
        for i in range(N):
            g_dot_x = np.dot(grad[i], pts[i])
            grad_proj[i] = grad[i] - g_dot_x * pts[i]
        
        return energy, grad_proj.flatten()
    
    # Use L-BFGS-B with a wrapper that handles the projection
    x0 = pts.flatten()
    
    best_energy = _log_energy(pts)
    best_pts = pts.copy()
    
    # Try L-BFGS-B
    try:
        # We need a function that returns (f, g) for L-BFGS-B
        # But L-BFGS-B doesn't support constraints well for sphere
        # Let's use a different approach: optimize in spherical coordinates or use penalty
        
        # Alternative: use scipy's minimize with method 'L-BFGS-B' on the flat coordinates
        # and add a penalty for not being on the sphere, or just normalize after each step
        
        # Let's try a simpler gradient descent with projection
        lr = 0.01
        for step in range(5000):
            # Compute energy and gradient
            dots = np.dot(pts, pts.T)
            dist_sq = 2.0 - 2.0 * dots
            np.fill_diagonal(dist_sq, 1.0)
            dist_sq = np.maximum(dist_sq, 1e-15)
            dist = np.sqrt(dist_sq)
            log_inv_dist = -np.log(dist)
            np.fill_diagonal(log_inv_dist, 0.0)
            triu_indices = np.triu_indices(N, k=1)
            energy = np.sum(log_inv_dist[triu_indices])
            
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
            
            # Compute gradient
            grad = np.zeros_like(pts)
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    diff = pts[i] - pts[j]
                    d = np.linalg.norm(diff)
                    if d < 1e-15:
                        continue
                    grad[i] += diff / (d * d)
            
            # Project to tangent space
            for i in range(N):
                g_dot_x = np.dot(grad[i], pts[i])
                grad[i] = grad[i] - g_dot_x * pts[i]
            
            # Update
            pts = pts - lr * grad
            
            # Project back to sphere
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            pts = pts / norms
            
            # Reduce learning rate
            if step % 500 == 0:
                lr *= 0.5
            
            if lr < 1e-6:
                break
                
    except Exception:
        pass
    
    # Final check
    final_energy = _log_energy(pts)
    if final_energy < best_energy:
        best_pts = pts
    
    return _project_to_sphere(best_pts)

# EVOLVE-BLOCK-END
