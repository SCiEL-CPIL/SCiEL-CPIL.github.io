"""Improved packer for 300 equal circles in a unit circle."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 300


def _score(pts):
    """Compute the objective: min(min_pair_dist/2, min(1-|c_i|))."""
    n = len(pts)
    if n == 0:
        return 0.0
    # Distance to boundary
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    boundary = np.min(1.0 - radii)
    if boundary <= 0:
        return 0.0
    # Pairwise distances via KD-tree
    tree = cKDTree(pts)
    # We need the minimum pairwise distance. Use query_pairs for efficiency.
    pairs = tree.query_pairs(r=1.0)  # all pairs within distance 1
    if len(pairs) == 0:
        return boundary
    # Compute distances for all pairs within 1.0
    min_dist = np.inf
    for i, j in pairs:
        d = np.sqrt(np.sum((pts[i] - pts[j]) ** 2))
        if d < min_dist:
            min_dist = d
    return min(min_dist / 2.0, boundary)


def _force_relaxation(pts, max_iters=2000, dt_init=0.01, dt_decay=0.999):
    """
    Repulsion-based relaxation to maximize minimum distance.
    Uses a soft repulsion force that pushes points apart.
    """
    n = len(pts)
    pts = pts.copy()
    dt = dt_init
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for it in range(max_iters):
        # Build KD-tree for neighbor queries
        tree = cKDTree(pts)
        # Query all pairs within a reasonable radius
        # Use a radius that's a bit larger than current min dist to avoid O(N^2)
        # But for N=300, even O(N^2) is manageable if vectorized
        # Let's do full pairwise for simplicity and speed with N=300
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 2)
        dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (N, N)
        np.fill_diagonal(dists, np.inf)
        
        # Repulsion force: F_ij = (1/d^2) * (p_i - p_j) for d < threshold
        # Use a smooth force: F = (p_i - p_j) / max(d, eps)^2
        # Only apply for pairs closer than some cutoff
        cutoff = 0.2
        mask = dists < cutoff
        # Force on each point
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(n):
                if i != j and mask[i, j]:
                    d = dists[i, j]
                    # Repulsive force away from j
                    f = diff[i, j] / (d ** 2 + 1e-10)
                    forces[i] += f
        
        # Boundary force: push points inward if too close to boundary
        radii = np.sqrt(np.sum(pts ** 2, axis=1))
        boundary_margin = 1.0 - radii
        # If a point is too close to boundary, push it inward
        for i in range(n):
            if boundary_margin[i] < 0.05:
                # Push toward center
                norm = radii[i]
                if norm > 1e-10:
                    forces[i] -= pts[i] / norm * (0.05 - boundary_margin[i]) * 10
        
        # Update positions
        pts += dt * forces
        # Clamp to unit circle
        new_radii = np.sqrt(np.sum(pts ** 2, axis=1))
        too_far = new_radii > 0.99
        if np.any(too_far):
            pts[too_far] *= 0.99 / new_radii[too_far]
        
        # Decay step size
        dt *= dt_decay
        
        # Check score periodically
        if it % 10 == 0:
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
    
    return best_pts, best_score


def _hex_init():
    """Generate a hexagonal lattice initial configuration."""
    points = []
    spacing = 0.12
    k = 10
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.sqrt(x * x + y * y) <= 0.95:
                points.append([x, y])
    points = np.array(points)
    # Select N points closest to center
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    idx = np.argsort(radii, kind='stable')[:N]
    return points[idx]


def _fibonacci_init():
    """Fibonacci spiral initial configuration."""
    n = N
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    angles = np.arange(n) * golden_angle
    radii = np.sqrt(np.arange(n) / n) * 0.95
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])


def _random_init():
    """Random initial configuration inside the unit circle."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N)) * 0.95
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    # Try multiple initializations and pick the best
    best_pts = None
    best_score = -np.inf
    
    # Initializations to try
    inits = [
        ("hex", _hex_init),
        ("fib", _fibonacci_init),
    ]
    
    # Add some random inits
    np.random.seed(42)
    for _ in range(3):
        inits.append(("rand", _random_init))
    
    for name, init_func in inits:
        if time.time() > deadline - 20.0:
            break
        
        pts = init_func()
        score = _score(pts)
        
        # Relax
        pts_relaxed, score_relaxed = _force_relaxation(pts, max_iters=1500, dt_init=0.005, dt_decay=0.9995)
        
        if score_relaxed > best_score:
            best_score = score_relaxed
            best_pts = pts_relaxed.copy()
    
    # Final polish: local search with smaller steps
    if best_pts is not None and time.time() < deadline - 5.0:
        # Try a few more relaxation passes with smaller steps
        for _ in range(3):
            if time.time() > deadline - 3.0:
                break
            pts_polished, score_polished = _force_relaxation(
                best_pts, max_iters=500, dt_init=0.001, dt_decay=0.999
            )
            if score_polished > best_score:
                best_score = score_polished
                best_pts = pts_polished.copy()
    
    if best_pts is None:
        # Fallback
        best_pts = _hex_init()
    
    return best_pts
