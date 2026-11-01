"""Improved packing: multi-start local optimization with binary search on radius."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def _score(points):
    """Return the packing radius (min of half-pair-distances and boundary distances)."""
    n = len(points)
    if n < 2:
        return 0.0
    # Boundary constraint
    boundary = 1.0 - np.linalg.norm(points, axis=1)
    r_bound = np.min(boundary)
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    return min(r_bound, r_pair)


def _hex_init(n):
    """Generate n points on a hexagonal lattice inside the unit disk."""
    pts = []
    # Try various spacings to get roughly n points
    for spacing in np.linspace(0.15, 0.35, 50):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= n:
            # Sort by radius to prefer central points
            idx = np.argsort(radius[mask])
            return points[idx[:n]].copy()
    # Fallback: random
    return _random_init(n)


def _ring_init(n):
    """Generate n points in concentric rings inside the unit disk."""
    pts = []
    # First point at center
    pts.append([0.0, 0.0])
    remaining = n - 1
    ring = 1
    while remaining > 0:
        # Number of points on this ring
        count = min(remaining, 6 * ring)
        r = 0.85 * ring / (ring + 1)  # keep inside
        if r >= 1.0:
            r = 0.95
        for i in range(count):
            theta = 2.0 * np.pi * i / count + (np.pi / count if ring % 2 else 0)
            pts.append([r * np.cos(theta), r * np.sin(theta)])
        remaining -= count
        ring += 1
    return np.array(pts[:n])


def _random_init(n, seed=0):
    """Generate n random points inside the unit disk."""
    rng = np.random.RandomState(seed)
    pts = []
    for _ in range(n):
        while True:
            x, y = rng.uniform(-1, 1), rng.uniform(-1, 1)
            if x * x + y * y <= 1.0:
                pts.append([x, y])
                break
    return np.array(pts)


def _local_optimize(points, max_iter=2000, tol=1e-10, step_init=0.05):
    """Local optimization using a repulsive force model."""
    n = len(points)
    pts = points.copy()
    step = step_init
    best_r = _score(pts)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        # Compute forces
        forces = np.zeros_like(pts)
        for i in range(n):
            # Boundary repulsion
            dist_center = np.linalg.norm(pts[i])
            if dist_center > 0.99:
                forces[i] -= 0.5 * pts[i] / max(dist_center, 1e-10)
            
            for j in range(n):
                if i == j:
                    continue
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    # Random nudge
                    forces[i] += 0.01 * np.random.randn(2)
                    continue
                if dist < 2.0 * best_r:
                    # Repulsive force
                    forces[i] += d / dist * (2.0 * best_r - dist)
        
        # Update positions
        pts += step * forces
        # Clamp to unit disk
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 0.999
        if np.any(mask):
            pts[mask] *= 0.999 / norms[mask]
        
        new_r = _score(pts)
        if new_r > best_r:
            best_r = new_r
            step *= 0.98
        else:
            step *= 0.9
            # Revert
            pts = pts - step * forces / (0.9 - 0.98) if step > 0 else pts
        
        if step < 1e-8:
            break
    
    return pts, best_r


def _refine_with_binary_search(points, r_low, r_high, max_iter=50):
    """Binary search on radius with constraint checking."""
    n = len(points)
    best_pts = points.copy()
    best_r = r_low
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        r_mid = (r_low + r_high) / 2.0
        # Check if packing with radius r_mid is feasible
        # We do a simple local optimization targeting radius r_mid
        pts = points.copy()
        for it in range(50):
            forces = np.zeros_like(pts)
            for i in range(n):
                dist_center = np.linalg.norm(pts[i])
                if dist_center > 1.0 - r_mid:
                    forces[i] -= 0.3 * pts[i] / max(dist_center, 1e-10)
                for j in range(n):
                    if i == j:
                        continue
                    d = pts[i] - pts[j]
                    dist = np.linalg.norm(d)
                    if dist < 2.0 * r_mid:
                        forces[i] += d / max(dist, 1e-10) * (2.0 * r_mid - dist)
            pts += 0.01 * forces
            norms = np.linalg.norm(pts, axis=1)
            mask = norms > 1.0 - r_mid
            if np.any(mask):
                pts[mask] *= (1.0 - r_mid) / norms[mask]
        
        r_actual = _score(pts)
        if r_actual >= r_mid - 1e-9:
            r_low = r_mid
            best_pts = pts.copy()
            best_r = max(best_r, r_actual)
        else:
            r_high = r_mid
    
    return best_pts, best_r


def construct():
    start_time = time.time()
    
    # Try multiple initializations
    inits = [
        ("hex", _hex_init(N)),
        ("ring", _ring_init(N)),
        ("rand0", _random_init(N, seed=42)),
        ("rand1", _random_init(N, seed=123)),
        ("rand2", _random_init(N, seed=999)),
    ]
    
    best_pts = None
    best_r = 0.0
    
    for name, init_pts in inits:
        if time.time() > _DEADLINE - 10:
            break
        # Local optimization
        pts, r = _local_optimize(init_pts, max_iter=1500, step_init=0.03)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback to hex
        best_pts = _hex_init(N)
        best_r = _score(best_pts)
    
    # Refine with binary search
    if time.time() < _DEADLINE - 10:
        best_pts, best_r = _refine_with_binary_search(best_pts, best_r, min(best_r + 0.05, 0.15), max_iter=30)
    
    # Final polish
    if time.time() < _DEADLINE - 5:
        best_pts, r2 = _local_optimize(best_pts, max_iter=500, step_init=0.005)
        if r2 > best_r:
            best_r = r2
            best_pts = best_pts
    
    # Ensure validity: clamp to disk and check
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_pts[mask] *= 0.999 / norms[mask]
    
    return best_pts
# EVOLVE-BLOCK-END
