"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _energy_and_grad_spherical(tp):
    """
    Compute energy and gradient for points parameterized by (theta, phi).
    tp: (N, 2) array. tp[:, 0] = theta, tp[:, 1] = phi.
    """
    n = tp.shape[0]
    theta = tp[:, 0]
    phi = tp[:, 1]
    
    # Compute points on sphere
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)
    
    x = sin_t * cos_p
    y = sin_t * sin_p
    z = cos_t
    pts = np.column_stack([x, y, z])
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0) # Avoid log(0)
    
    # Energy
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dist[iu])
    energy = -np.sum(log_dists)
    
    # Gradient
    # du/dtheta and du/dphi
    du_dtheta = np.column_stack([cos_t * cos_p, cos_t * sin_p, -sin_t])
    du_dphi = np.column_stack([-sin_t * sin_p, sin_t * cos_p, np.zeros(n)])
    
    # dot_diff_dtheta[i, j] = (u_i - u_j) . du_dtheta[i]
    dot_diff_dtheta = np.sum(diff * du_dtheta[:, np.newaxis, :], axis=2)
    dot_diff_dphi = np.sum(diff * du_dphi[:, np.newaxis, :], axis=2)
    
    dist_sq = dist * dist
    np.fill_diagonal(dist_sq, 1.0)
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    grad_theta = -np.sum(dot_diff_dtheta * inv_dist_sq, axis=1)
    grad_phi = -np.sum(dot_diff_dphi * inv_dist_sq, axis=1)
    
    grad = np.column_stack([grad_theta, grad_phi])
    
    return energy, grad

def _fibonacci_sphere_spherical():
    """Generate initial (theta, phi) from Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    theta = np.arccos(z)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    phi = phi % (2.0 * np.pi)
    return np.column_stack([theta, phi])

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    tp_init = _fibonacci_sphere_spherical()
    
    try:
        res = minimize(
            _energy_and_grad_spherical,
            tp_init,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 20}
        )
        
        if time.time() > deadline:
             # Return initial if timed out during minimize? No, minimize is synchronous.
             # If it takes too long, we just return whatever we have or init.
             pass
        
        tp_opt = res.x
    except Exception:
        tp_opt = tp_init
        
    # Convert back to Cartesian
    theta = tp_opt[:, 0]
    phi = tp_opt[:, 1]
    pts = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    
    # Normalize just in case
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    return pts
# EVOLVE-BLOCK-END
