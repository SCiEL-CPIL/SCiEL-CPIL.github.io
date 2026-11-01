"""Rewritten program for the 50-circle packing task."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # safety margin before 150s kill


def _score(pts):
    """Compute the packing score (largest common radius)."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Distance to boundary
    r_boundary = 1.0 - np.linalg.norm(pts, axis=1)
    if np.any(r_boundary < -1e-12):
        return 0.0
    # Pairwise distances
    diffs = pts[:, None, :] - pts[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_pair, np.min(r_boundary))


def _neg_score(pts_flat):
    """Negative score for minimization."""
    pts = pts_flat.reshape(N, 2)
    return -_score(pts)


def _grad_neg_score(pts_flat):
    """Numerical gradient of negative score."""
    pts = pts_flat.reshape(N, 2)
    f0 = _score(pts)
    eps = 1e-7
    grad = np.zeros_like(pts_flat)
    for i in range(N * 2):
        old = pts_flat[i]
        pts_flat[i] = old + eps
        f1 = _score(pts_flat.reshape(N, 2))
        pts_flat[i] = old
        grad[i] = (f1 - f0) / eps  # This is gradient of score; we want negative
    return -grad  # gradient of negative score


def _ring_init(n_rings, r_inner=0.0):
    """Generate points on concentric rings."""
    pts = []
    if r_inner < 0.05:
        pts.append([0.0, 0.0])
        remaining = N - len(pts)
    else:
        remaining = N
    if remaining <= 0:
        return np.array(pts[:N])
    
    # Distribute remaining points on rings
    # Try different ring configurations
    best_pts = None
    best_score = -1.0
    
    for n_rings in range(1, 8):
        # Distribute points among rings
        per_ring = [remaining // n_rings] * n_rings
        for i in range(remaining % n_rings):
            per_ring[i] += 1
        
        # Remove empty rings
        per_ring = [c for c in per_ring if c > 0]
        n_actual = len(per_ring)
        
        radii = np.linspace(0.15, 0.92, n_actual)
        pts = []
        for idx, (count, r) in enumerate(zip(per_ring, radii)):
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            # Offset alternating rings
            if idx % 2 == 1:
                angles += np.pi / count
            for a in angles:
                pts.append([r * np.cos(a), r * np.sin(a)])
        
        if len(pts) == N:
            pts = np.array(pts)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
    
    if best_pts is None:
        # Fallback: simple grid
        best_pts = _hex_init()
    
    return best_pts


def _hex_init():
    """Hexagonal lattice initialization."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98


def _local_optimize(pts, max_iter=200, tol=1e-10):
    """Local optimization using gradient ascent on score."""
    pts = pts.copy()
    flat = pts.flatten()
    
    for it in range(max_iter):
        f0 = _score(pts)
        grad = _grad_neg_score(flat)
        pts_new = pts - 0.01 * grad.reshape(N, 2)
        # Project back: keep within unit circle
        norms = np.linalg.norm(pts_new, axis=1)
        mask = norms > 0.999
        if np.any(mask):
            pts_new[mask] *= 0.999 / norms[mask]
        
        f1 = _score(pts_new)
        if f1 >= f0 - tol:
            pts = pts_new
        else:
            # Try smaller step
            pts_new = pts - 0.001 * grad.reshape(N, 2)
            norms = np.linalg.norm(pts_new, axis=1)
            mask = norms > 0.999
            if np.any(mask):
                pts_new[mask] *= 0.999 / norms[mask]
            f1 = _score(pts_new)
            if f1 >= f0 - tol:
                pts = pts_new
            else:
                break
    return pts


def construct():
    start_time = time.time()
    
    # Generate multiple initial configurations
    inits = []
    
    # 1. Hexagonal lattice
    inits.append(_hex_init())
    
    # 2. Ring-based inits with different parameters
    for r_inner in [0.0, 0.1, 0.2, 0.3]:
        for n_rings in [3, 4, 5, 6]:
            pts = _ring_init(n_rings, r_inner)
            inits.append(pts)
    
    # 3. Random inits
    rng = np.random.RandomState(42)
    for _ in range(10):
        angles = rng.uniform(0, 2 * np.pi, N)
        radii = rng.uniform(0, 0.95, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts)
    
    # 4. Structured grid
    side = int(np.ceil(np.sqrt(N)))
    grid_x = np.linspace(-0.8, 0.8, side)
    grid_y = np.linspace(-0.8, 0.8, side)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
    if len(grid_pts) >= N:
        inits.append(grid_pts[:N])
    
    # Optimize each initialization
    best_pts = None
    best_score = -1.0
    
    for init in inits:
        if time.time() - start_time > DEADLINE:
            break
        
        # Quick local optimization
        opt_pts = _local_optimize(init, max_iter=100)
        s = _score(opt_pts)
        if s > best_score:
            best_score = s
            best_pts = opt_pts.copy()
    
    # Final polish on best result
    if best_pts is not None and time.time() - start_time < DEADLINE - 10:
        best_pts = _local_optimize(best_pts, max_iter=500, tol=1e-12)
    
    if best_pts is None:
        best_pts = _hex_init()
    
    return best_pts
