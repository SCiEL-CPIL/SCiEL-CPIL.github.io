"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.optimize import minimize


def log_energy(pts):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = pts.shape[0]
    # Compute all pairwise distances
    # Use efficient computation
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    dists = dists[mask]
    
    # Avoid log(0) or negative values
    dists = np.maximum(dists, 1e-15)
    
    # Sum log(1/d) = -sum(log(d))
    # Use math.fsum for accuracy
    vals = -np.log(dists)
    return math.fsum(vals.tolist())


def log_energy_grad(pts):
    """Compute gradient of log energy w.r.t. points on the sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    
    dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = pts.shape[0]
    grad = np.zeros_like(pts)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pts[i] - pts[j]
            dist_sq = np.dot(d, d)
            if dist_sq < 1e-20:
                continue
            grad[i] += d / dist_sq
    
    return grad


def project_to_tangent(pt, grad):
    """Project gradient to tangent space of sphere at pt."""
    return grad - np.dot(grad, pt) * pt


def riemannian_gradient_descent(pts, max_iter=5000, lr_init=0.01):
    """Riemannian gradient descent on the sphere using exponential map."""
    pts = pts.copy()
    lr = lr_init
    best_energy = log_energy(pts)
    best_pts = pts.copy()
    
    for it in range(max_iter):
        # Compute gradient
        grad = log_energy_grad(pts)
        
        # Project to tangent space
        tangent_grad = np.array([project_to_tangent(pts[i], grad[i]) for i in range(len(pts))])
        
        # Adaptive step size: use backtracking if energy increases
        for attempt in range(20):
            # Exponential map: move along tangent direction
            # For unit sphere: exp_p(v) = p*cos(|v|) + (v/|v|)*sin(|v|)
            new_pts = np.zeros_like(pts)
            for i in range(len(pts)):
                v = lr * tangent_grad[i]
                norm_v = np.linalg.norm(v)
                if norm_v < 1e-12:
                    new_pts[i] = pts[i]
                else:
                    new_pts[i] = pts[i] * np.cos(norm_v) + (v / norm_v) * np.sin(norm_v)
            
            # Normalize to ensure we're on the sphere
            new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
            
            new_energy = log_energy(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts.copy()
            
            if new_energy < log_energy(pts) + 1e-12:
                pts = new_pts
                break
            else:
                lr *= 0.5
        
        if lr < 1e-10:
            break
        
        # Slightly increase learning rate if we've been stuck
        if it > 0 and it % 100 == 0:
            lr = min(lr * 1.1, lr_init)
    
    return best_pts


def construct():
    """Construct 150 points on the unit sphere minimizing logarithmic energy."""
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    # Riemannian gradient descent
    pts = riemannian_gradient_descent(pts, max_iter=3000, lr_init=0.005)
    
    # Final normalization
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    return pts
# EVOLVE-BLOCK-END
