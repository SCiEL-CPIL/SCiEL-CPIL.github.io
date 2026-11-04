# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _estimate_covering_radius(pts):
    """Fast grid-based estimate of covering radius."""
    n_theta = 100
    n_phi = 200
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

def _run_simulation(seed):
    """Run a single symplectic integration simulation."""
    N = 50
    rng = np.random.default_rng(seed)
    
    # Random initialization on sphere
    pts = rng.normal(size=(N, 3))
    pts = _normalize(pts)
    
    vel = np.zeros((N, 3))
    
    max_steps = 20000
    power = 2  # Coulomb potential
    
    for step in range(max_steps):
        t = step / max_steps
        
        # Cooling schedule: dt starts large, ends small
        # Damping starts low, ends high
        dt = 0.05 * (1.0 - t) ** 3 + 0.001
        damping = 0.95 + 0.04 * t  # 0.95 to 0.99
        
        # Force calculation
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist2, 1.0)
        
        # Force: -grad(1/r^2) = 2/r^3 * r_vec
        # f_mag = 2 / dist2^(1.5)
        inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))
        f_mag = 2.0 * inv_dist3
        np.fill_diagonal(f_mag, 0.0)
        
        forces = np.sum(f_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Project to tangent
        dot_fp = np.sum(forces * pts, axis=1, keepdims=True)
        forces -= dot_fp * pts
        
        # Velocity Verlet
        vel_half = vel + 0.5 * dt * forces
        dot_vp = np.sum(vel_half * pts, axis=1, keepdims=True)
        vel_half -= dot_vp * pts
        
        pts_new = pts + dt * vel_half
        pts_new = _normalize(pts_new)
        
        # Recalculate forces at new position
        diff_new = pts_new[:, np.newaxis, :] - pts_new[np.newaxis, :, :]
        dist2_new = np.sum(diff_new ** 2, axis=2)
        np.fill_diagonal(dist2_new, 1.0)
        inv_dist3_new = 1.0 / (dist2_new * np.sqrt(dist2_new))
        f_mag_new = 2.0 * inv_dist3_new
        np.fill_diagonal(f_mag_new, 0.0)
        
        forces_new = np.sum(f_mag_new[:, :, np.newaxis] * diff_new, axis=1)
        dot_fp_new = np.sum(forces_new * pts_new, axis=1, keepdims=True)
        forces_new -= dot_fp_new * pts_new
        
        vel_new = vel_half + 0.5 * dt * forces_new
        vel_new *= damping
        
        dot_vp_new = np.sum(vel_new * pts_new, axis=1, keepdims=True)
        vel_new -= dot_vp_new * pts_new
        
        pts = pts_new
        vel = vel_new
        
    return _normalize(pts)

def construct():
    """
    Minimize covering radius by running multiple symplectic simulations 
    with different random initializations and selecting the best result.
    """
    N = 50
    best_pts = None
    best_score = np.inf
    
    num_runs = 10
    
    for i in range(num_runs):
        pts = _run_simulation(i)
        score = _estimate_covering_radius(pts)
        if score < best_score:
            best_score = score
            best_pts = pts
            
    return best_pts
# EVOLVE-BLOCK-END
