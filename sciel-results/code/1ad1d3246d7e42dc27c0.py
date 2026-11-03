"""Spherical covering with 50 points via symplectic potential minimization."""
import numpy as np
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _compute_forces(pts, power=4):
    """Compute repulsive forces between all pairs of points on the sphere."""
    N = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist_sq[dist_sq == 0] = 1e-10
    dist = np.sqrt(dist_sq)
    
    mask = ~np.eye(N, dtype=bool)
    dist_masked = np.where(mask, dist, 1.0)
    
    coeff = np.zeros_like(dist_sq)
    coeff[mask] = power / (dist_masked[mask] ** (power + 2))
    
    np.fill_diagonal(coeff, 0.0)
    
    forces = np.einsum('ij,ijk->ik', coeff, diff)
    
    return forces

def construct():
    """
    Use symplectic velocity Verlet integration to minimize a repulsive potential
    on the unit sphere.
    """
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    # Parameters
    dt = 0.01
    n_steps = 5000
    power = 4
    
    # Velocity initialization
    vel = np.zeros((N, 3))
    
    # Initial forces
    forces = _compute_forces(pts, power)
    
    for step in range(n_steps):
        # Velocity Verlet
        vel_half = vel + 0.5 * dt * forces
        pts_new = pts + dt * vel_half
        pts_new = _normalize(pts_new)
        
        forces_new = _compute_forces(pts_new, power)
        
        vel_new = vel_half + 0.5 * dt * forces_new
        
        # Damping
        vel = vel_new * 0.99
        pts = pts_new
        
    # Final normalization
    pts = _normalize(pts)
    
    return pts
# EVOLVE-BLOCK-END
