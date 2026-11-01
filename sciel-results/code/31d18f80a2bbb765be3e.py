"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 115.0  # safety margin before 120s kill


def _score(pts):
    """Compute the packing radius for given centers."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    # Boundary constraint
    dist_origin = np.linalg.norm(pts, axis=1)
    if np.any(dist_origin >= 1.0):
        return 0.0
    boundary_r = 1.0 - dist_origin
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return float(min(min_pair, np.min(boundary_r)))


def _hex_lattice_init():
    """Generate initial positions from hexagonal lattice."""
    pts = []
    spacing = 0.2
    for j in range(-8, 9):
        for i in range(-8, 9):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y < 0.85:
                pts.append([x, y])
    pts = np.array(pts)
    # Select N points closest to origin
    r = np.linalg.norm(pts, axis=1)
    idx = np.argsort(r)[:N]
    return pts[idx]


def _ring_init():
    """Initialize with concentric rings."""
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # Ring 1: 6 circles
    r1 = 0.2
    for k in range(6):
        a = 2 * np.pi * k / 6
        pts.append([r1 * np.cos(a), r1 * np.sin(a)])
    # Ring 2: 12 circles
    r2 = 0.4
    for k in range(12):
        a = 2 * np.pi * k / 12 + np.pi / 12
        pts.append([r2 * np.cos(a), r2 * np.sin(a)])
    # Ring 3: remaining 31 circles
    r3 = 0.6
    remaining = N - len(pts)
    for k in range(remaining):
        a = 2 * np.pi * k / remaining
        pts.append([r3 * np.cos(a), r3 * np.sin(a)])
    return np.array(pts[:N])


def _random_init(seed):
    """Random initialization inside unit disk."""
    rng = np.random.RandomState(seed)
    pts = []
    for _ in range(N):
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            if x * x + y * y < 0.95:
                pts.append([x, y])
                break
    return np.array(pts)


def _force_directed(pts, max_iters=500, dt=0.001, k_rep=10.0, k_bound=5.0):
    """Force-directed relaxation with velocity Verlet."""
    pts = pts.astype(np.float64).copy()
    vel = np.zeros_like(pts)
    best_pts = pts.copy()
    best_r = _score(pts)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5.0:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        for i in range(N):
            # Boundary force
            r_i = np.linalg.norm(pts[i])
            if r_i > 0.9:
                forces[i] -= k_bound * (r_i - 0.9) * pts[i] / max(r_i, 1e-10)
            
            # Repulsive forces
            for j in range(i + 1, N):
                diff = pts[i] - pts[j]
                d = np.linalg.norm(diff)
                if d < 0.3:
                    f = k_rep * (0.3 - d) / max(d, 1e-10)
                    forces[i] += f * diff
                    forces[j] -= f * diff
        
        # Velocity Verlet integration
        vel += forces * dt
        vel *= 0.95  # damping
        pts += vel * dt
        
        # Clamp to unit disk
        r = np.linalg.norm(pts, axis=1)
        mask = r > 0.98
        if np.any(mask):
            pts[mask] *= 0.98 / r[mask]
        
        r_val = _score(pts)
        if r_val > best_r:
            best_r = r_val
            best_pts = pts.copy()
    
    return best_pts, best_r


def _lbfgs_refine(pts, maxiter=200):
    """L-BFGS-B refinement of the packing."""
    def objective(x):
        p = x.reshape(N, 2)
        # Penalty for boundary violation
        r_orig = np.linalg.norm(p, axis=1)
        boundary_penalty = np.sum(np.maximum(0, r_orig - 0.99) ** 2)
        
        # Pairwise distance objective (maximize min distance)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        
        return -min_d + 10.0 * boundary_penalty
    
    x0 = pts.reshape(-1)
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': maxiter, 'ftol': 1e-10})
    return result.x.reshape(N, 2)


def construct():
    """Main function: find optimal packing of 50 equal circles in unit disk."""
    global _DEADLINE
    _DEADLINE = time.time() + 115.0
    
    best_pts = None
    best_r = 0.0
    
    # Strategy 1: Hex lattice + force directed
    if time.time() < _DEADLINE - 30:
        pts_hex = _hex_lattice_init()
        pts_fd, r_fd = _force_directed(pts_hex, max_iters=300)
        if r_fd > best_r:
            best_r = r_fd
            best_pts = pts_fd.copy()
    
    # Strategy 2: Ring init + force directed
    if time.time() < _DEADLINE - 30:
        pts_ring = _ring_init()
        pts_fd2, r_fd2 = _force_directed(pts_ring, max_iters=300)
        if r_fd2 > best_r:
            best_r = r_fd2
            best_pts = pts_fd2.copy()
    
    # Strategy 3: Random inits with force directed
    for seed in [42, 123, 456]:
        if time.time() > _DEADLINE - 20:
            break
        pts_rand = _random_init(seed)
        pts_fd3, r_fd3 = _force_directed(pts_rand, max_iters=200)
        if r_fd3 > best_r:
            best_r = r_fd3
            best_pts = pts_fd3.copy()
    
    # Final L-BFGS refinement
    if best_pts is not None and time.time() < _DEADLINE - 10:
        pts_refined = _lbfgs_refine(best_pts, maxiter=100)
        r_refined = _score(pts_refined)
        if r_refined > best_r:
            best_r = r_refined
            best_pts = pts_refined
    
    if best_pts is None:
        best_pts = _hex_lattice_init()
    
    return best_pts
# EVOLVE-BLOCK-END
