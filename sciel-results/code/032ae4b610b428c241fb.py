"""Improved packing: dense hexagonal init + repulsive-force refinement with wall constraint."""
import numpy as np
from scipy.spatial import cKDTree
import time

N = 300

def _compute_score(points):
    """Compute the raw score: min(min_pair_dist/2, min_i(1-|c_i|))."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check all inside unit circle
    dists = np.hypot(points[:, 0], points[:, 1])
    if np.any(dists >= 1.0):
        return 0.0
    # Check no coincident centers
    if np.any(dists < 1e-12):
        # Allow center at origin but check pair distances
        pass
    # Pairwise distances
    tree = cKDTree(points)
    pairs = tree.query_pairs(2.0)  # all pairs within 2.0 (will be all relevant)
    if len(pairs) == 0:
        min_pair_dist = np.inf
    else:
        pair_dists = []
        for i, j in pairs:
            d = np.hypot(points[i, 0] - points[j, 0], points[i, 1] - points[j, 1])
            pair_dists.append(d)
        min_pair_dist = min(pair_dists) if pair_dists else np.inf
    min_wall = 1.0 - np.max(dists)
    return min(min_pair_dist / 2.0, min_wall)


def _hex_lattice_init(n, seed=42):
    """Generate n points on a dense hexagonal lattice inside unit disk, with small random perturbation."""
    rng = np.random.RandomState(seed)
    # Generate hex lattice points
    pts = []
    # Use a fine hex lattice
    spacing = 0.05
    dx = spacing
    dy = spacing * np.sqrt(3.0) / 2.0
    # Cover the unit disk
    for j in range(-20, 21):
        for i in range(-20, 21):
            x = i * dx + (j % 2) * dx / 2.0
            y = j * dy
            if x * x + y * y <= 1.0:
                pts.append([x, y])
    pts = np.array(pts)
    # If we have more than n, randomly select n; if fewer, add random points
    if len(pts) >= n:
        # Select n points that are well-spread: use a greedy farthest-point sampling
        # For speed, just random sample with replacement avoided
        indices = rng.choice(len(pts), size=n, replace=False)
        pts = pts[indices]
    else:
        # Pad with random points
        extra = rng.uniform(-0.9, 0.9, size=(n - len(pts), 2))
        pts = np.vstack([pts, extra])
    # Add small random perturbation to break symmetry
    pts += rng.uniform(-0.005, 0.005, size=pts.shape)
    # Ensure inside unit circle
    dists = np.hypot(pts[:, 0], pts[:, 1])
    mask = dists < 1.0
    if not np.all(mask):
        pts[~mask] = pts[~mask] / dists[~mask, None] * 0.95
    return pts


def _repulsive_force_refine(points, n_steps=5000, dt=0.0001, k_repulse=1.0, k_wall=10.0, cooling=0.999):
    """Refine points using repulsive forces between circles and wall repulsion."""
    pts = points.copy().astype(float)
    n = len(pts)
    current_dt = dt
    
    for step in range(n_steps):
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsive forces (use KDTree for efficiency)
        tree = cKDTree(pts)
        pairs = tree.query_pairs(0.3)  # Only consider pairs that might interact
        
        for i, j in pairs:
            diff = pts[i] - pts[j]
            d = np.hypot(diff[0], diff[1])
            if d < 1e-10:
                # Random small push
                forces[i] += rng_uniform_small(2)
                forces[j] -= rng_uniform_small(2)
            else:
                # Repulsive force: strong when close
                # Use 1/d^2 style but capped
                force_mag = k_repulse / (d * d + 1e-6)
                force_vec = diff / d * force_mag
                forces[i] += force_vec
                forces[j] -= force_vec
        
        # Wall repulsion: push points inward if near boundary
        dists = np.hypot(pts[:, 0], pts[:, 1])
        for i in range(n):
            if dists[i] > 0.95:
                # Push inward
                inward = -pts[i] / (dists[i] + 1e-10)
                force_mag = k_wall * (dists[i] - 0.95)
                forces[i] += inward * force_mag
        
        # Update positions
        pts += forces * current_dt
        
        # Project back inside unit circle if any went outside
        dists = np.hypot(pts[:, 0], pts[:, 1])
        outside = dists > 1.0
        if np.any(outside):
            pts[outside] = pts[outside] / dists[outside, None]
        
        # Cool down
        current_dt *= cooling
    
    return pts


def rng_uniform_small(size):
    return np.random.uniform(-0.001, 0.001, size=size)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration
    points = _hex_lattice_init(N, seed=42)
    best_score = _compute_score(points)
    best_points = points.copy()
    
    # Multiple refinement passes with different parameters
    for trial in range(10):
        if time.time() > deadline:
            break
        
        # Try different random seeds for initialization
        if trial == 0:
            pts = _hex_lattice_init(N, seed=42)
        else:
            pts = _hex_lattice_init(N, seed=42 + trial * 100)
        
        # Refine
        n_steps = 3000
        pts = _repulsive_force_refine(pts, n_steps=n_steps, dt=0.0002, k_repulse=1.0, k_wall=10.0, cooling=0.9995)
        
        # Check score
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
        
        # If time allows, do a second refinement pass on best
        if time.time() < deadline - 30.0:
            pts2 = best_points.copy()
            pts2 = _repulsive_force_refine(pts2, n_steps=2000, dt=0.0001, k_repulse=0.5, k_wall=5.0, cooling=0.9998)
            score2 = _compute_score(pts2)
            if score2 > best_score:
                best_score = score2
                best_points = pts2.copy()
    
    return best_points
