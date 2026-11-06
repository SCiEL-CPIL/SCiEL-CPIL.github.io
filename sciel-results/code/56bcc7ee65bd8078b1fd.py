"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_score(pts):
    """Compute the packing score: min(min pair dist/2, min(1 - |p|))."""
    if len(pts) != N:
        return -1.0
    if not np.all(np.isfinite(pts)):
        return -1.0
    # Check container
    norms = np.sqrt(np.sum(pts**2, axis=1))
    if np.any(norms >= 1.0):
        return -1.0
    min_boundary = 1.0 - np.max(norms)
    # Compute pairwise distances
    # Use a more memory-efficient approach for 300 points
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    # Actually, let's just do the full NxN but it's 300x300 = 90000, fine
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_pair, min_boundary)


def _fibonacci_disk_init(n, seed=0):
    """Generate n points using Fibonacci disk sampling."""
    rng = np.random.RandomState(seed)
    # Use the Fibonacci sphere projection to disk, or just use the known formula
    # Actually, let's use a simpler approach: random points in disk, then optimize
    # Or use the golden angle spiral
    i = np.arange(n)
    # Golden angle
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = i * golden_angle
    # Radius: use sqrt of uniform distribution for uniform area density
    r = np.sqrt((i + 0.5) / n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.column_stack([x, y])
    # Add small random perturbation to break symmetry
    pts += rng.normal(0, 0.01, size=pts.shape)
    # Project back inside unit circle
    norms = np.sqrt(np.sum(pts**2, axis=1))
    mask = norms > 1.0
    if np.any(mask):
        pts[mask] /= norms[mask, np.newaxis] * 1.001
    return pts


def _force_relaxation(pts, max_iters=5000, dt_init=0.01, dt_decay=0.999, deadline=None):
    """
    Force-directed relaxation to maximize minimum distance.
    Uses repulsive forces between all pairs and attractive force toward center.
    """
    pts = pts.copy()
    n = len(pts)
    dt = dt_init
    best_pts = pts.copy()
    best_score = _compute_score(pts)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1.0)  # Avoid division by zero
        
        # Repulsive force: F = 1/d^2 in direction away from other point
        # This is a common choice for packing problems
        inv_d2 = 1.0 / (dists**2)
        np.fill_diagonal(inv_d2, 0.0)
        
        # Force from repulsion
        forces = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
        
        # Boundary constraint: push points inward if too close to edge
        norms = np.sqrt(np.sum(pts**2, axis=1))
        boundary_force = np.zeros_like(pts)
        too_close = norms > 0.95
        if np.any(too_close):
            # Push inward proportionally to how close to edge
            strength = (norms[too_close] - 0.95) * 10.0
            boundary_force[too_close] = -pts[too_close] / norms[too_close, np.newaxis] * strength[:, np.newaxis]
        
        # Update positions
        pts = pts + dt * (forces + boundary_force)
        
        # Project back into unit disk
        norms = np.sqrt(np.sum(pts**2, axis=1))
        too_far = norms > 1.0
        if np.any(too_far):
            pts[too_far] /= norms[too_far, np.newaxis]
        
        # Decay step size
        dt *= dt_decay
        
        # Check score
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    return best_pts, best_score


def construct():
    """Construct 300 circle centers in unit disk maximizing common radius."""
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    best_pts = None
    best_score = -1.0
    
    # Try multiple random initializations with different seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        
        # Initialize with Fibonacci disk
        pts = _fibonacci_disk_init(N, seed=seed)
        
        # Run force relaxation
        pts_opt, score = _force_relaxation(pts, max_iters=3000, dt_init=0.01, 
                                           dt_decay=0.9995, deadline=deadline)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt
    
    # If we haven't found anything valid, fall back to a simple grid
    if best_pts is None or best_score <= 0:
        # Simple hexagonal grid fallback
        spacing = 0.1
        pts_list = []
        for j in range(-20, 21):
            for i in range(-20, 21):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3) / 2
                if x*x + y*y < 1.0:
                    pts_list.append([x, y])
        pts = np.array(pts_list[:N])
        best_pts = pts
    
    return best_pts
# EVOLVE-BLOCK-END
