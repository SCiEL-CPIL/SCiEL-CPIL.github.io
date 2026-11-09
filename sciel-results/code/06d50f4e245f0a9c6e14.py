"""Rewritten program: minimize logarithmic energy of 100 points on the unit sphere."""
import numpy as np
import math
import time

N = 100

def _log_energy(pts):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = pts.shape[0]
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Avoid log(0) and log of very small numbers
    # Only sum i<j
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    # Clamp to avoid log(0)
    d = np.maximum(d, 1e-15)
    return math.fsum(np.log(1.0 / d).tolist())

def _log_energy_and_grad(pts):
    """Compute energy and gradient (tangent) for points on unit sphere."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Energy
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    energy = math.fsum(np.log(1.0 / d).tolist())
    
    # Gradient: for each pair (i,j), contribution to grad_i is - (x_i - x_j) / (d * d) * (1/d) ... 
    # Actually, energy = sum_{i<j} -log(d_ij)
    # d/dx_i (-log(d_ij)) = -(1/d_ij) * (x_i - x_j)/d_ij = -(x_i - x_j)/d_ij^2
    # So grad_i = sum_{j!=i} (x_i - x_j)/d_ij^2 ... wait, let me redo.
    # E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # dE/dx_i = -sum_{j!=i} (1/d_ij) * d(d_ij)/dx_i
    # d(d_ij)/dx_i = (x_i - x_j)/d_ij
    # So dE/dx_i = -sum_{j!=i} (x_i - x_j)/d_ij^2
    # = sum_{j!=i} (x_j - x_i)/d_ij^2
    
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(i + 1, n):
            d_ij = dists[i, j]
            if d_ij < 1e-15:
                continue
            diff_ij = pts[j] - pts[i]
            term = diff_ij / (d_ij * d_ij)
            grad[i] += term
            grad[j] -= term
    
    # Project gradient to tangent space
    for i in range(n):
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _initial_fibonacci():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    pts = _initial_fibonacci()
    best_energy = _log_energy(pts)
    best_pts = pts.copy()
    
    # Local optimization using gradient descent on the sphere
    lr = 0.01
    min_lr = 1e-8
    patience = 50
    no_improve = 0
    
    while time.time() < deadline:
        energy, grad = _log_energy_and_grad(pts)
        
        if energy < best_energy - 1e-12:
            best_energy = energy
            best_pts = pts.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > patience:
            break
        
        # Gradient descent step
        pts_new = pts - lr * grad
        pts = _project_to_sphere(pts_new)
        
        # Reduce learning rate if no improvement
        if no_improve % 10 == 0 and no_improve > 0:
            lr *= 0.5
            if lr < min_lr:
                break
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
        
        # Random initial points
        pts_r = np.random.randn(N, 3)
        pts_r = _project_to_sphere(pts_r)
        
        # Quick local optimization
        lr_r = 0.01
        for step in range(2000):
            if time.time() > deadline - 2.0:
                break
            energy_r, grad_r = _log_energy_and_grad(pts_r)
            pts_r_new = pts_r - lr_r * grad_r
            pts_r = _project_to_sphere(pts_r_new)
            if step % 100 == 0:
                lr_r *= 0.95
        
        final_energy = _log_energy(pts_r)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = pts_r.copy()
    
    return best_pts
