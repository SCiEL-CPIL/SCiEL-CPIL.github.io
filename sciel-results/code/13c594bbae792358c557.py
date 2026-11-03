# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _compute_covering_radius(pts):
    """Compute the covering radius in degrees for points on the unit sphere."""
    # Use a high-resolution grid for accurate estimation
    n_theta = 200
    n_phi = 400
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


def construct():
    """
    Minimize a repulsive potential on the unit sphere using symplectic integration
    with simulated annealing to escape local minima.
    """
    N = 50
    
    # 1. Initialization: Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    vel = np.zeros((N, 3))
    
    # Simulation parameters
    max_steps = 30000
    power = 8  # Sharper repulsion for better uniformity
    
    start_time = time.time()
    timeout = 110.0  # Leave 10s margin
    
    # Annealing parameters
    # T starts high to allow large moves, decays to 0
    T_start = 0.05
    T_end = 0.001
    cooling_rate = (T_end / T_start) ** (1.0 / max_steps)
    
    for step in range(max_steps):
        if time.time() - start_time > timeout:
            break
            
        # Progress 0 to 1
        t = step / max_steps
        
        # Temperature for noise
        T = T_start * (cooling_rate ** step)
        
        # Adaptive step size: start large, decay
        # Linear decay is often more stable for this
        dt = 0.005 * (1.0 - 0.5 * t) + 0.0005
        damping = 0.95 - 0.05 * t
        
        # --- Vectorized Force Calculation ---
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        
        # Avoid division by zero on diagonal
        np.fill_diagonal(dist2, 1.0)
        
        # Force magnitude: proportional to 1/dist^(power+1)
        inv_dist = 1.0 / np.sqrt(dist2)
        f_mag = inv_dist ** (power + 1)
        
        # Zero out self-forces
        np.fill_diagonal(f_mag, 0.0)
        
        # Total force on each point i: sum_j f_mag[i,j] * diff[i,j]
        forces = np.sum(f_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Project forces onto the tangent plane of the sphere at each point
        dot_fp = np.sum(forces * pts, axis=1, keepdims=True)
        forces -= dot_fp * pts
        
        # --- Velocity Verlet Update ---
        # v_half = v + 0.5 * dt * F
        vel_half = vel + 0.5 * dt * forces
        
        # Project vel_half to tangent plane
        dot_vp = np.sum(vel_half * pts, axis=1, keepdims=True)
        vel_half -= dot_vp * pts
        
        # Add thermal noise (Simulated Annealing)
        # Noise is added to velocity, scaled by sqrt(T)
        if T > 1e-5:
            noise = np.random.normal(0, np.sqrt(T), size=(N, 3))
            # Project noise to tangent plane
            dot_vn = np.sum(noise * pts, axis=1, keepdims=True)
            noise -= dot_vn * pts
            vel_half += noise
        
        # x_new = x + dt * v_half
        pts_new = pts + dt * vel_half
        
        # Normalize to unit sphere
        pts_new = _normalize(pts_new)
        
        # Recalculate forces at new position
        diff_new = pts_new[:, np.newaxis, :] - pts_new[np.newaxis, :, :]
        dist2_new = np.sum(diff_new ** 2, axis=2)
        np.fill_diagonal(dist2_new, 1.0)
        inv_dist_new = 1.0 / np.sqrt(dist2_new)
        f_mag_new = inv_dist_new ** (power + 1)
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
