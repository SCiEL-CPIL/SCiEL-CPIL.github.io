import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12
_DEADLINE = 145.0

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(rng):
    """Generate random initial points."""
    pts = rng.standard_normal((N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _get_lattice(M=5000):
    """Generate a high-resolution spherical lattice (Fibonacci) for spring targets."""
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_hull_radius(pts):
    """Compute the true covering radius using convex hull."""
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.pi
            
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return np.max(radii)
    except Exception:
        return np.pi

def _simulate(pts_init, lattice, dt=0.005, steps=5000, k_spring=10.0, k_repel=50.0, sigma=0.2):
    """
    Rigid-body dynamics simulation.
    - Springs pull each point to its nearest lattice point.
    - Repulsion pushes points apart if they are too close.
    - Velocity Verlet integration with damping.
    """
    pts = pts_init.copy()
    v = np.zeros_like(pts)
    
    # Pre-compute nearest lattice points for each initial point
    # We will update these assignments dynamically if points move significantly,
    # but for stability, we can just compute forces to the *current* nearest lattice point.
    # For speed, we might just use a fixed assignment from start, or recompute every N steps.
    
    best_radius = _compute_hull_radius(pts)
    
    for step in range(steps):
        # 1. Compute forces
        forces = np.zeros_like(pts)
        
        # Spring force: F = -k * (p - target)
        # Find nearest lattice point for each p
        # This is O(N*M). With N=130, M=5000, this is 650k ops per step.
        # 5000 steps * 650k = 3.25e9 ops. Might be too slow in Python loops.
        # Vectorized: (N,1,3) - (1,M,3) -> (N,M,3). Norm -> (N,M). Min -> (N,).
        # Memory: 130 * 5000 * 3 * 8 bytes = 15.6 MB. Acceptable.
        
        diff = pts[:, np.newaxis, :] - lattice[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        nearest_idx = np.argmin(dists_sq, axis=1)
        nearest_pts = lattice[nearest_idx]
        
        # Spring force
        forces += -k_spring * (pts - nearest_pts)
        
        # Repulsion force: Soft sphere potential
        # F_ij = k_repel * (sigma - r_ij) * (p_i - p_j) / r_ij if r_ij < sigma
        # Vectorized repulsion is O(N^2). N=130 -> 16900 pairs. Fast.
        diff_rep = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_rep = np.linalg.norm(diff_rep, axis=2)
        np.fill_diagonal(dists_rep, np.inf)
        
        # Mask for close pairs
        mask = dists_rep < sigma
        # Avoid division by zero
        safe_dists = np.where(dists_rep > _EPS, dists_rep, 1.0)
        
        # Force magnitude
        mag = k_repel * (sigma - dists_rep) / safe_dists_rep
        mag = np.where(mask, mag, 0.0)
        
        # Force vector: sum over j of mag_ij * (p_i - p_j) / r_ij
        # F_i = sum_j mag_ij * diff_ij
        forces_rep = np.sum(mag[:, :, np.newaxis] * diff_rep, axis=1)
        forces += forces_rep
        
        # 2. Velocity Verlet Integration
        # v_{n+1/2} = v_n + 0.5 * dt * F_n / m
        # x_{n+1} = x_n + dt * v_{n+1/2}
        # (Compute F_{n+1} at x_{n+1})
        # v_{n+1} = v_{n+1/2} + 0.5 * dt * F_{n+1} / m
        
        # We already have F_n (forces)
        v += 0.5 * dt * forces
        pts_new = pts + dt * v
        
        # Normalize to sphere
        pts_new /= np.linalg.norm(pts_new, axis=1, keepdims=True)
        
        # Damping: v *= 0.99
        v *= 0.95
        
        # Check improvement every 100 steps
        if step % 100 == 0:
            r = _compute_hull_radius(pts_new)
            if r < best_radius:
                best_radius = r
                pts = pts_new.copy()
            else:
                # If it got worse, maybe revert or just continue?
                # In dynamics, we usually just let it run.
                # But if it diverges, we should stop.
                if r > np.deg2rad(30):
                    break
            pts = pts_new
        
        pts = pts_new

    # Final normalization
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Initial setup
    lattice = _get_lattice(5000)
    rng = np.random.default_rng(42)
    
    best_pts = _initial_fib()
    best_r = _compute_hull_radius(best_pts)
    
    # Run simulation with different parameters
    # We can try multiple runs with different dt, k, sigma
    
    params_list = [
        {'dt': 0.005, 'steps': 3000, 'k_spring': 5.0, 'k_repel': 20.0, 'sigma': 0.3},
        {'dt': 0.003, 'steps': 5000, 'k_spring': 10.0, 'k_repel': 50.0, 'sigma': 0.2},
        {'dt': 0.001, 'steps': 10000, 'k_spring': 20.0, 'k_repel': 100.0, 'sigma': 0.15},
    ]
    
    for p in params_list:
        if time.time() > deadline - 10.0:
            break
            
        # Start from best current
        pts_init = best_pts.copy()
        # Add small noise to escape local minima
        pts_init += rng.normal(0, 0.01, (N, 3))
        pts_init /= np.linalg.norm(pts_init, axis=1, keepdims=True)
        
        new_pts, new_r = _simulate(
            pts_init, lattice, 
            dt=p['dt'], steps=p['steps'], 
            k_spring=p['k_spring'], k_repel=p['k_repel'], sigma=p['sigma']
        )
        
        if new_r < best_r:
            best_r = new_r
            best_pts = new_pts
        
    # Final polish: One more run with very small steps from best
    if time.time() < deadline - 5.0:
        pts_init = best_pts.copy()
        new_pts, new_r = _simulate(
            pts_init, lattice, 
            dt=0.001, steps=5000, 
            k_spring=50.0, k_repel=200.0, sigma=0.1
        )
        if new_r < best_r:
            best_r = new_r
            best_pts = new_pts

    # Final normalization and validity check
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
