# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _compute_covering_radius_grid(pts, n_theta=180, n_phi=360):
    """Compute the covering radius in degrees for points on the unit sphere using a grid."""
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


def _simulate(pts_init, power=6, max_steps=20000):
    """
    Run symplectic integration to minimize the potential.
    """
    N = len(pts_init)
    pts = pts_init.copy()
    vel = np.zeros((N, 3))
    
    # Pre-compute constants
    # We use a cooling schedule
    # dt starts large and decays
    
    for step in range(max_steps):
        t = step / max_steps
        # Cooling: dt decreases, damping increases
        dt = 0.005 * (1.0 - t) ** 1.5 + 0.0001
        damping = 0.98 + 0.01 * t  # Damping increases to settle down
        
        # --- Force Calculation ---
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist2, 1.0)
        
        # Potential V = sum 1/r^power
        # Force F = -grad V = power * r_vec / r^(power+2)
        # We want to minimize V, so we move in direction of -grad V
        # grad (1/r^k) = -k r_vec / r^(k+2)
        # So -grad (1/r^k) = k r_vec / r^(k+2)
        
        inv_dist = 1.0 / np.sqrt(dist2)
        # f_mag = power * inv_dist^(power+1)
        f_mag = power * (inv_dist ** (power + 1))
        np.fill_diagonal(f_mag, 0.0)
        
        forces = np.sum(f_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Project forces onto tangent plane
        dot_fp = np.sum(forces * pts, axis=1, keepdims=True)
        forces -= dot_fp * pts
        
        # --- Velocity Verlet Update ---
        # v_half = v + 0.5 * dt * F
        vel_half = vel + 0.5 * dt * forces
        
        # Project velocity to tangent plane
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
        f_mag_new = power * (inv_dist_new ** (power + 1))
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
        
    return _normalize(pts)


def construct():
    """
    Minimize a repulsive potential on the unit sphere using symplectic integration.
    Uses multiple restarts to find a better local minimum.
    """
    N = 50
    best_pts = None
    best_score = float('inf')
    
    # Define several initial configurations
    inits = []
    
    # 1. Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(fib_pts)
    
    # 2. Random points (fixed seed for reproducibility in construct, but different from fib)
    rng = np.random.default_rng(42)
    for _ in range(3):
        # Generate random points on sphere
        vecs = rng.normal(size=(N, 3))
        rand_pts = _normalize(vecs)
        # Add slight perturbation to fib to escape local minima? Or just use random.
        # Let's use random starts.
        inits.append(rand_pts)
        
    # 3. Perturbed Fibonacci
    fib_pert = fib_pts.copy()
    noise = rng.normal(scale=0.1, size=(N, 3))
    fib_pert += noise
    fib_pert = _normalize(fib_pert)
    inits.append(fib_pert)

    # Run simulation for each init
    for init in inits:
        # Use power=6 for stronger repulsion which tends to equalize gaps better
        # Reduce steps to allow multiple runs within time limit
        # 5 inits * 15000 steps = 75000 steps total.
        # Each step is ~ O(N^2). 50^2 = 2500.
        # 75000 * 2500 = 1.875e8 operations. This should be < 120s in NumPy.
        pts = _simulate(init, power=6, max_steps=15000)
        
        # Evaluate score
        # Use a reasonably accurate grid estimate to select the best
        score = _compute_covering_radius_grid(pts, n_theta=120, n_phi=240)
        
        if score < best_score:
            best_score = score
            best_pts = pts
            
    # Final refinement on the best candidate
    # Run a shorter, finer-tuned simulation on the best candidate
    if best_pts is not None:
        best_pts = _simulate(best_pts, power=6, max_steps=5000)
        # Final normalization
        best_pts = _normalize(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
