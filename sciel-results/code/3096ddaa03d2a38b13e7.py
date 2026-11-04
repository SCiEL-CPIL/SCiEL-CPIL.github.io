# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _compute_covering_radius(pts):
    """
    Estimate covering radius using a high-resolution grid.
    This is an approximation but sufficient for guiding the search if exact hull is too slow.
    """
    n_theta = 180
    n_phi = 360
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi)
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    Y = np.zeros((n_theta, n_phi, 3))
    Y[:, :, 0] = np.sin(TH) * np.cos(PH)
    Y[:, :, 1] = np.sin(TH) * np.sin(PH)
    Y[:, :, 2] = np.cos(TH)
    Y = Y.reshape(-1, 3)
    
    # Compute max dot product (min angular distance) for each grid point
    dots = Y @ pts.T
    min_dots = np.max(dots, axis=1)
    max_angular = np.arccos(np.clip(min_dots, -1, 1))
    return np.degrees(np.max(max_angular))


def _local_optimize(pts, steps=1000, lr=0.01):
    """
    Perform a simple gradient descent on the tangent space to minimize the max nearest neighbor distance.
    This acts as a local polish.
    """
    N = pts.shape[0]
    pts = pts.copy()
    
    for _ in range(steps):
        # Compute nearest neighbor distances
        # dist2[i, j] = |p_i - p_j|^2
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist2, np.inf)
        
        # Find nearest neighbor for each point
        min_dist2 = np.min(dist2, axis=1)
        min_dist = np.sqrt(min_dist2)
        
        # We want to increase the minimum distance for the "worst" points.
        # Gradient of -min_dist with respect to p_i.
        # This is complex. Instead, we use a repulsive force from the nearest neighbor.
        # Force on i is proportional to (p_i - p_j) / dist^2 * weight
        # Weight is higher if dist is small.
        
        nearest_idx = np.argmin(dist2, axis=1)
        
        forces = np.zeros_like(pts)
        for i in range(N):
            j = nearest_idx[i]
            d = dist2[i, j]
            if d < 1e-10: continue
            # Repel from nearest neighbor
            # Stronger repulsion if distance is small
            weight = 1.0 / (d + 1e-6)
            forces[i] += weight * (pts[i] - pts[j])
            
        # Project to tangent plane
        dot_fp = np.sum(forces * pts, axis=1, keepdims=True)
        forces -= dot_fp * pts
        
        # Update
        pts = pts + lr * forces
        pts = _normalize(pts)
        
        # Decay learning rate
        lr *= 0.99
        
    return pts


def construct():
    """
    Minimize a dual-potential energy on the unit sphere using Velocity Verlet.
    Potential: E = sum_{i<j} (A/r^2 + B/r^12)
    This combines long-range spreading with short-range anti-clustering.
    """
    N = 50
    
    # 1. Initialization: Random uniform points or Fibonacci
    # Fibonacci is a good deterministic start
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    vel = np.zeros((N, 3))
    
    # Simulation parameters
    max_steps = 20000
    A = 1.0   # Long range coefficient
    B = 1.0   # Short range coefficient
    
    # Pre-compute constants for potential derivative
    # V = A/r^2 + B/r^12
    # F = -grad V = 2A/r^3 * r_vec + 12B/r^13 * r_vec
    # Note: r_vec = p_i - p_j, r = |r_vec|
    # Force on i from j: (2A/r^3 + 12B/r^13) * (p_i - p_j)
    
    for step in range(max_steps):
        t = step / max_steps
        
        # Adaptive step size: larger initially, smaller later
        # Use exponential decay
        dt = 0.05 * np.exp(-2.0 * t) + 0.001
        damping = 0.95 - 0.05 * t
        
        # --- Force Calculation ---
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist2, 1.0) # Avoid div by zero, will be masked
        
        dist = np.sqrt(dist2)
        
        # Force magnitude factor: 2A/r^3 + 12B/r^13
        # r^3 = dist2^1.5
        # r^13 = dist2^6.5
        inv_dist2 = 1.0 / dist2
        inv_dist3 = inv_dist2 * np.sqrt(inv_dist2)
        inv_dist13 = inv_dist3 * (inv_dist2 ** 5)
        
        f_mag = 2.0 * A * inv_dist3 + 12.0 * B * inv_dist13
        np.fill_diagonal(f_mag, 0.0)
        
        forces = np.sum(f_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Project to tangent plane
        dot_fp = np.sum(forces * pts, axis=1, keepdims=True)
        forces -= dot_fp * pts
        
        # --- Velocity Verlet Update ---
        # v_half = v + 0.5 * dt * F
        vel_half = vel + 0.5 * dt * forces
        
        # Project vel_half to tangent plane
        dot_vp = np.sum(vel_half * pts, axis=1, keepdims=True)
        vel_half -= dot_vp * pts
        
        # x_new = x + dt * v_half
        pts_new = pts + dt * vel_half
        pts_new = _normalize(pts_new)
        
        # Recalculate forces at new position
        diff_new = pts_new[:, np.newaxis, :] - pts_new[np.newaxis, :, :]
        dist2_new = np.sum(diff_new ** 2, axis=2)
        np.fill_diagonal(dist2_new, 1.0)
        
        dist_new = np.sqrt(dist2_new)
        inv_dist2_new = 1.0 / dist2_new
        inv_dist3_new = inv_dist2_new * np.sqrt(inv_dist2_new)
        inv_dist13_new = inv_dist3_new * (inv_dist2_new ** 5)
        
        f_mag_new = 2.0 * A * inv_dist3_new + 12.0 * B * inv_dist13_new
        np.fill_diagonal(f_mag_new, 0.0)
        
        forces_new = np.sum(f_mag_new[:, :, np.newaxis] * diff_new, axis=1)
        
        # Project new forces to tangent plane
        dot_fp_new = np.sum(forces_new * pts_new, axis=1, keepdims=True)
        forces_new -= dot_fp_new * pts_new
        
        # v_new = v_half + 0.5 * dt * F_new
        vel_new = vel_half + 0.5 * dt * forces_new
        
        # Apply damping
        vel_new *= damping
        
        # Project vel_new to tangent plane at new position
        dot_vp_new = np.sum(vel_new * pts_new, axis=1, keepdims=True)
        vel_new -= dot_vp_new * pts_new
        
        # Update state
        pts = pts_new
        vel = vel_new
        
    # Final local optimization to refine the covering radius
    pts = _local_optimize(pts, steps=2000, lr=0.05)
    
    return _normalize(pts)
# EVOLVE-BLOCK-END
