"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize, differential_evolution

def _energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|xi-xj|)."""
    n = pts.shape[0]
    # Compute pairwise distances
    # Use broadcasting: dist[i,j] = |pts[i] - pts[j]|
    # For memory efficiency with N=150, this is 150x150 = 22500 floats, fine.
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    # Compute log(1/d) = -log(d) for all pairs i<j
    # Use upper triangle
    iu = np.triu_indices(n, 1)
    d = dist[iu]
    # Avoid log(0)
    d = np.maximum(d, 1e-15)
    return -np.sum(np.log(d))

def _gradient(pts):
    """Compute gradient of logarithmic energy w.r.t. point positions.
    
    E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)
    dE/dxi = -sum_{j!=i} (xi - xj) / |xi - xj|^2
    """
    n = pts.shape[0]
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = pts[i] - pts[j]
            dist_sq = np.dot(diff, diff)
            if dist_sq < 1e-20:
                continue
            dist = np.sqrt(dist_sq)
            grad[i] -= diff / (dist_sq)
    return grad

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin before 150s limit
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_energy = _energy(pts)
    
    # Gradient descent with adaptive step size on the sphere
    # Using projected gradient: move in direction of negative gradient, then project to sphere
    lr = 0.01
    min_lr = 1e-6
    max_iter = 5000
    stagnation = 0
    best_energy_in_window = best_energy
    window_size = 50
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute gradient
        g = _gradient(pts)
        
        # Take step
        pts_new = pts - lr * g
        
        # Project to sphere
        pts_new = _project_to_sphere(pts_new)
        
        # Check energy
        e_new = _energy(pts_new)
        
        if e_new < best_energy - 1e-12:
            best_pts = pts_new.copy()
            best_energy = e_new
            stagnation = 0
            # Increase learning rate slightly if improving
            lr = min(lr * 1.05, 0.05)
        else:
            pts = pts_new  # Keep the new position even if not improving, to explore
            stagnation += 1
            # Decrease learning rate
            lr *= 0.95
            if lr < min_lr:
                lr = min_lr
        
        # If stagnating too long, try a smaller perturbation
        if stagnation > window_size:
            # Add small random perturbation to escape local minima
            noise = 0.001 * np.random.randn(N, 3)
            pts_new = pts + noise
            pts_new = _project_to_sphere(pts_new)
            e_pert = _energy(pts_new)
            if e_pert < best_energy:
                pts = pts_new
                best_pts = pts_new.copy()
                best_energy = e_pert
                stagnation = 0
            else:
                pts = pts_new  # Keep perturbed position
            lr = min_lr * 2
            stagnation = 0
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        lr = 1e-5
        for iteration in range(2000):
            if time.time() > deadline:
                break
            g = _gradient(best_pts)
            best_pts = best_pts - lr * g
            best_pts = _project_to_sphere(best_pts)
            lr *= 0.99
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Check for very close points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Separate them
                best_pts[j] = best_pts[i] + 1e-6 * np.random.randn(3)
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts
# EVOLVE-BLOCK-END
