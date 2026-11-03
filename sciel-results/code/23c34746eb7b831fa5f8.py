"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so
from scipy.spatial import cKDTree
import math

_DEADLINE = 150.0
_start_time = time.time()


def _time_left():
    return _DEADLINE - 15.0 - (time.time() - _start_time)


def _compute_radius(pts):
    """Compute the maximum radius for a set of points."""
    n = len(pts)
    if n < 2:
        return 0.0
    # Distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(pts, axis=1)
    # Pairwise distances
    tree = cKDTree(pts)
    pairs = tree.query_pairs(1e-10)
    if len(pairs) == 0:
        return min(dist_boundary.min(), 0.05)
    dists = []
    for i, j in pairs:
        d = np.linalg.norm(pts[i] - pts[j])
        dists.append(d)
    if len(dists) == 0:
        return min(dist_boundary.min(), 0.05)
    return min(dist_boundary.min(), min(dists) / 2.0)


def _objective(pts_flat, n):
    """Objective to minimize: negative of the minimum radius (with penalties)."""
    pts = pts_flat.reshape(-1, 2)
    # Boundary penalty
    dist_boundary = 1.0 - np.linalg.norm(pts, axis=1)
    min_boundary = dist_boundary.min()
    if min_boundary < 0:
        return -min_boundary * 100.0 + 100.0
    
    # Pairwise distances
    tree = cKDTree(pts)
    pairs = tree.query_pairs(1e-10)
    if len(pairs) == 0:
        return -min_boundary
    
    dists = np.array([np.linalg.norm(pts[i] - pts[j]) for i, j in pairs])
    min_dist = dists.min()
    
    return -min(min_boundary, min_dist / 2.0)


def _soft_min_objective(pts_flat, n, temperature):
    """Smooth approximation of the minimum using soft-min."""
    pts = pts_flat.reshape(-1, 2)
    dist_boundary = 1.0 - np.linalg.norm(pts, axis=1)
    
    tree = cKDTree(pts)
    pairs = tree.query_pairs(1e-10)
    
    if len(pairs) == 0:
        # Only boundary constraint
        vals = dist_boundary
    else:
        dists = np.array([np.linalg.norm(pts[i] - pts[j]) for i, j in pairs])
        vals = np.concatenate([dist_boundary, dists / 2.0])
    
    # Soft min
    if temperature <= 0:
        return -vals.min()
    else:
        min_val = vals.min()
        return min_val - temperature * np.log(np.sum(np.exp((vals - min_val) / temperature)))


def _relaxation(pts, n, max_iters=2000, dt=0.001):
    """Force-based relaxation with Hertzian overlap repulsion."""
    pts = pts.copy()
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        forces = np.zeros_like(pts)
        tree = cKDTree(pts)
        
        # Boundary forces
        dist_boundary = 1.0 - np.linalg.norm(pts, axis=1)
        for i in range(n):
            if dist_boundary[i] < 0.1:
                forces[i] += -pts[i] / max(np.linalg.norm(pts[i]), 1e-10) * (0.1 - dist_boundary[i]) * 10.0
        
        # Pairwise repulsion (Hertzian: F ~ overlap^2)
        pairs = tree.query_pairs(0.15)
        for i, j in pairs:
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 0.15:
                overlap = 0.15 - dist
                force_mag = overlap * overlap * 50.0
                if dist > 1e-10:
                    forces[i] += d / dist * force_mag
                    forces[j] -= d / dist * force_mag
        
        pts += dt * forces
        # Project back inside
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 0.999
        if np.any(mask):
            pts[mask] *= 0.999 / norms[mask]
    
    return pts


def _lbfgs_optimize(pts, n, max_iters=500, temperature=0.01):
    """L-BFGS-B optimization with soft-min objective."""
    x0 = pts.flatten()
    
    def obj(x):
        return _soft_min_objective(x, n, temperature)
    
    try:
        result = so.minimize(
            obj, x0, method='L-BFGS-B',
            options={'maxiter': max_iters, 'gtol': 1e-8, 'maxls': 20}
        )
        return result.x.reshape(-1, 2)
    except Exception:
        return pts


def _hex_init(n):
    """Hexagonal lattice initialization."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= n:
            return points[np.argsort(radius, kind="stable")[:n]]
        spacing *= 0.98


def _random_init(n, seed=42):
    """Random initialization inside unit disk."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n):
        while True:
            r = rng.random()
            theta = rng.uniform(0, 2 * np.pi)
            x, y = r * math.cos(theta), r * math.sin(theta)
            if x*x + y*y < 0.81:  # Stay well inside
                pts.append([x, y])
                break
    return np.array(pts)


def _concentric_rings_init(n):
    """Concentric rings initialization."""
    pts = []
    # Center point
    pts.append([0.0, 0.0])
    remaining = n - 1
    ring = 1
    while remaining > 0:
        # Number of points in this ring
        circumference = 2 * np.pi * ring * 0.03
        count = min(remaining, max(6, int(circumference / 0.06)))
        for k in range(count):
            theta = 2 * np.pi * k / count
            r = ring * 0.03
            pts.append([r * math.cos(theta), r * math.sin(theta)])
        remaining -= count
        ring += 1
    return np.array(pts[:n])


def construct():
    global _start_time
    _start_time = time.time()
    
    best_pts = None
    best_radius = 0.0
    
    # Multiple initialization strategies
    inits = []
    
    # 1. Hexagonal lattice
    hex_pts = _hex_init(N)
    inits.append(hex_pts)
    
    # 2. Concentric rings
    ring_pts = _concentric_rings_init(N)
    inits.append(ring_pts)
    
    # 3. Random with different seeds
    for seed in [1, 7, 42]:
        rand_pts = _random_init(N, seed)
        inits.append(rand_pts)
    
    # Process each initialization
    for init_idx, pts in enumerate(inits):
        if _time_left() < 20.0:
            break
        
        # Relaxation phase
        relaxed = _relaxation(pts, N, max_iters=1500, dt=0.002)
        
        # L-BFGS with progressively tighter temperatures
        temps = [0.05, 0.02, 0.01, 0.005, 0.001]
        current_pts = relaxed
        for temp in temps:
            if _time_left() < 10.0:
                break
            current_pts = _lbfgs_optimize(current_pts, N, max_iters=300, temperature=temp)
        
        # Final relaxation
        current_pts = _relaxation(current_pts, N, max_iters=500, dt=0.0005)
        
        # Compute radius
        r = _compute_radius(current_pts)
        
        if r > best_radius:
            best_radius = r
            best_pts = current_pts.copy()
        
        # Check if we've hit the known best
        if best_radius > 0.0531:
            break
    
    if best_pts is None:
        return _hex_init(N)
    
    return best_pts
# EVOLVE-BLOCK-END
