# EVOLVE-BLOCK-START
import numpy as np
from math import fsum, log, sqrt
import time
import scipy.optimize as opt

def _energy_from_angles(params):
    """
    params: array of shape (2*N,)
    [theta_0, phi_0, theta_1, phi_1, ...]
    """
    thetas = params[0::2]
    phis = params[1::2]
    
    # Convert to Cartesian
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)
    
    points = np.column_stack([x, y, z])
    
    # Compute pairwise distances
    # Use broadcasting for efficiency
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    dists = np.sqrt(dist_sq[mask])
    
    # Log energy
    # Avoid log(0)
    dists = np.maximum(dists, 1e-15)
    vals = -np.log(dists)
    
    return fsum(vals.tolist())

def _fibonacci_spiral_angles(N):
    """Generate Fibonacci spiral points in (theta, phi)."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    # theta = arccos(z)
    thetas = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    phis = phi % (2 * np.pi)
    return thetas, phis

def construct():
    best_energy = float('inf')
    best_params = None
    
    rng = np.random.default_rng(42)
    
    # Define bounds for L-BFGS-B
    # theta in [0, pi], phi in [0, 2*pi]
    bounds = [(0, np.pi), (0, 2*np.pi)] * N
    
    restarts = []
    
    # Restart 1: Fibonacci Spiral
    thetas, phis = _fibonacci_spiral_angles(N)
    restarts.append(np.column_stack([thetas, phis]).flatten())
    
    # Restart 2-6: Random
    for _ in range(5):
        thetas = rng.uniform(0, np.pi, N)
        phis = rng.uniform(0, 2*np.pi, N)
        restarts.append(np.column_stack([thetas, phis]).flatten())
        
    # Restart 7-10: Random with small noise around Fibonacci
    for _ in range(4):
        thetas, phis = _fibonacci_spiral_angles(N)
        thetas = np.clip(thetas + rng.standard_normal(N) * 0.1, 0, np.pi)
        phis = (phis + rng.standard_normal(N) * 0.1) % (2 * np.pi)
        restarts.append(np.column_stack([thetas, phis]).flatten())

    for params in restarts:
        res = opt.minimize(_energy_from_angles, params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.fun < best_energy:
            best_energy = res.fun
            best_params = res.x.copy()
            
    # Convert best params to points
    thetas = best_params[0::2]
    phis = best_params[1::2]
    
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)
    
    points = np.column_stack([x, y, z])
    
    # Normalize just in case of numerical drift
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    return points
# EVOLVE-BLOCK-END
