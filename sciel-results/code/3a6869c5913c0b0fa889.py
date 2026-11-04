# EVOLVE-BLOCK-START
import numpy as np


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def construct():
    """
    Minimize a repulsive potential on the unit sphere using symplectic velocity Verlet integration.
    Uses inverse-square law for better covering properties.
    """
    N = 50
    
    # 1. Initialization: Fibonacci sphere for good initial distribution
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    vel = np.zeros((N, 3))
    
    # Simulation parameters optimized for covering
    max_steps = 80000
    power = 2  # Inverse-square law: smoother, better for covering than higher powers
    
    # Precompute constants
    k_power = power
    
    for step in range(max_steps):
        # Progress 0 to 1
        t = step / max_steps
        
        # Adaptive step size with cooling schedule
        # Start with larger steps, decay to small steps for fine convergence
        dt = 0.005 * (1.0 - t) ** 3 + 0.00005
        
        # Damping: start with high damping, reduce over time
        damping = 0.95 - 0.05 * t
        
        # --- Vectorized Force Calculation ---
        # diff[i, j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        
        # dist2[i, j] = |pts[i] - pts[j]|^2
        dist2 = np.sum(diff ** 2, axis=2)
        
        # Avoid division by zero on diagonal
        np.fill_diagonal(dist2, 1.0)
        
        # Force: F_i = sum_{j != i} k * (pi - pj) / |pi - pj|^(power+1)
        # For V = sum 1/r^k, grad = -k * r_vec / r^(k+1)
        # Force = -grad = k * r_vec / r^(k+1)
        inv_dist = 1.0 / np.sqrt(dist2)
        f_mag = k_power * inv_dist ** (k_power + 1)
        
        # Zero out self-forces
        np.fill_diagonal(f_mag, 0.0)
        
        # Total force on each point i: sum_j f_mag[i,j] * diff[i,j]
        forces = np.sum(f_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Project forces onto the tangent plane of the sphere at each point
        # F_tangent = F - (F . p) p
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
        
        # Normalize to unit sphere
        pts_new = _normalize(pts_new)
        
        # Recalculate forces at new position
        diff_new = pts_new[:, np.newaxis, :] - pts_new[np.newaxis, :, :]
        dist2_new = np.sum(diff_new ** 2, axis=2)
        np.fill_diagonal(dist2_new, 1.0)
        inv_dist_new = 1.0 / np.sqrt(dist2_new)
        f_mag_new = k_power * inv_dist_new ** (k_power + 1)
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
    
    # Final normalization
    pts = _normalize(pts)
    
    return pts
# EVOLVE-BLOCK-END
