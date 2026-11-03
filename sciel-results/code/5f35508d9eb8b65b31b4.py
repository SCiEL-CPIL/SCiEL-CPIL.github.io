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


def _compute_covering_radius(pts):
    """Compute the covering radius in degrees for points on the unit sphere."""
    try:
        hull = ConvexHull(pts)
        max_radius = 0.0
        for simplex in hull.simplices:
            verts = pts[simplex]
            a, b, c = verts[0], verts[1], verts[2]
            
            edge1 = b - a
            edge2 = c - a
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            
            centroid = (a + b + c) / 3.0
            if np.dot(centroid, normal) < 0:
                normal = -normal
            
            d_a = np.arccos(np.clip(np.dot(normal, a), -1, 1))
            
            if d_a > max_radius:
                max_radius = d_a
        
        return np.degrees(max_radius)
    
    except Exception:
        n_theta = 50
        n_phi = 100
        thetas = np.linspace(0, np.pi, n_theta)
        phis = np.linspace(0, 2 * np.pi, n_phi)
        TH, PH = np.meshgrid(thetas, phis, indexing='ij')
        Y = np.zeros((n_theta, n_phi, 3))
        Y[:, :, 0] = np.sin(TH) * np.cos(PH)
        Y[:, :, 1] = np.sin(TH) * np.sin(PH)
        Y[:, :, 2] = np.cos(TH)
        Y = Y.reshape(-1, 3)
        
        dots = Y @ pts.T
        min_dots = np.max(dots, axis=1)
        max_angular = np.arccos(np.clip(min_dots, -1, 1))
        return np.degrees(np.max(max_angular))


def construct():
    """
    Use symplectic velocity Verlet integration to minimize a repulsive potential
    on the unit sphere, aiming to find a good covering configuration.
    """
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    vel = np.zeros((N, 3))
    
    # Parameters
    power = 4
    dt = 0.01
    n_steps = 50000
    gamma = 0.05
    
    # Precompute mask for i != j
    mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask, False)
    
    for step in range(n_steps):
        # Compute forces
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist2 = np.sum(diff**2, axis=2)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            factor = 1.0 / (dist2 ** ((power + 2) / 2.0))
        
        factor[~mask] = 0.0
        
        forces = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        # Constraint force
        r_i = np.linalg.norm(pts, axis=1)
        r_i_safe = np.where(r_i > 1e-10, r_i, 1.0)
        F_con = -2.0 * (r_i - 1.0) * (pts / r_i_safe[:, np.newaxis])
        
        forces += F_con
        
        # Velocity Verlet
        vel_half = vel + 0.5 * dt * forces
        pts_new = pts + dt * vel_half
        pts_new = _normalize(pts_new)
        
        # Recompute forces for new points
        diff_new = pts_new[:, np.newaxis, :] - pts_new[np.newaxis, :, :]
        dist2_new = np.sum(diff_new**2, axis=2)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            factor_new = 1.0 / (dist2_new ** ((power + 2) / 2.0))
        
        factor_new[~mask] = 0.0
        forces_new = np.sum(factor_new[:, :, np.newaxis] * diff_new, axis=1)
        
        r_i_new = np.linalg.norm(pts_new, axis=1)
        r_i_new_safe = np.where(r_i_new > 1e-10, r_i_new, 1.0)
        F_con_new = -2.0 * (r_i_new - 1.0) * (pts_new / r_i_new_safe[:, np.newaxis])
        forces_new += F_con_new
        
        vel_new = vel_half + 0.5 * dt * forces_new
        vel = vel_new * (1.0 - gamma * dt)
        pts = pts_new
        
    return pts
# EVOLVE-BLOCK-END
