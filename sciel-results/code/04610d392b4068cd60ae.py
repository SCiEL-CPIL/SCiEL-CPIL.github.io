# EVOLVE-BLOCK-START
import numpy as np


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _compute_covering_radius(pts):
    """Compute the covering radius in degrees for points on the unit sphere."""
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
    
    dots = Y @ pts.T
    min_dots = np.max(dots, axis=1)
    max_angular = np.arccos(np.clip(min_dots, -1, 1))
    return np.degrees(np.max(max_angular))


def construct():
    """
    Minimize a repulsive potential on the unit sphere using velocity Verlet integration.
    
    The potential is V = sum_{i<j} 1/|p_i - p_j|^4, which strongly repels points
    at short range, encouraging even distribution. An attractive force toward the
    origin keeps points on the unit sphere.
    """
    N = 50
    
    # Initialize with Fibonacci lattice for good initial distribution
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    phi = golden_angle * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    vel = np.zeros((N, 3))
    
    # Simulation parameters
    max_steps = 100000
    power = 4  # Inverse power law exponent
    
    # Precompute constants
    inv_power = 1.0 / power
    
    for step in range(max_steps):
        # Cooling schedule: dt starts large, decays
        t = step / max_steps
        dt = 0.005 * (1.0 - t) ** 1.5 + 0.00005
        damping = 0.995 - 0.005 * t
        
        # --- Compute forces at current positions ---
        # diff[i, j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        
        # Avoid division by zero on diagonal
        np.fill_diagonal(dist2, 1.0)
        
        # Force magnitude: F = -grad(V) where V = sum 1/r^power
        # d/dr (1/r^power) = -power * r^(-power-1)
        # Force on i from j: power * r^(-power-1) * (p_i - p_j)
        inv_dist = 1.0 / np.sqrt(dist2)
        f_mag = power * inv_dist ** (power + 1)
        
        # Zero out self-forces
        np.fill_diagonal(f_mag, 0.0)
        
        # Total force on each point i: sum_j f_mag[i,j] * diff[i,j]
        forces = np.sum(f_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Add attractive force toward origin to keep points on sphere
        # F_attract = -k * p (toward center)
        k_attract = 0.01
        forces -= k_attract * pts
        
        # Project forces onto tangent plane of sphere at each point
        # F_tangent = F - (F . p) p
        dot_fp = np.sum(forces * pts, axis=1, keepdims=True)
        forces -= dot_fp * pts
        
        # --- Velocity Verlet Step 1: v_half = v + 0.5 * dt * F ---
        vel_half = vel + 0.5 * dt * forces
        
        # Project velocity to tangent plane
        dot_vp = np.sum(vel_half * pts, axis=1, keepdims=True)
        vel_half -= dot_vp * pts
        
        # --- Velocity Verlet Step 2: x_new = x + dt * v_half ---
        pts_new = pts + dt * vel_half
        
        # Normalize to unit sphere
        pts_new = _normalize(pts_new)
        
        # --- Compute forces at new positions ---
        diff_new = pts_new[:, np.newaxis, :] - pts_new[np.newaxis, :, :]
        dist2_new = np.sum(diff_new ** 2, axis=2)
        np.fill_diagonal(dist2_new, 1.0)
        inv_dist_new = 1.0 / np.sqrt(dist2_new)
        f_mag_new = power * inv_dist_new ** (power + 1)
        np.fill_diagonal(f_mag_new, 0.0)
        forces_new = np.sum(f_mag_new[:, :, np.newaxis] * diff_new, axis=1)
        forces_new -= k_attract * pts_new
        
        # Project new forces to tangent plane
        dot_fp_new = np.sum(forces_new * pts_new, axis=1, keepdims=True)
        forces_new -= dot_fp_new * pts_new
        
        # --- Velocity Verlet Step 3: v_new = v_half + 0.5 * dt * F_new ---
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
