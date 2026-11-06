"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree

def _hex_init(n):
    """Hexagonal lattice initialization clipped to unit disk."""
    # Start with a reasonable spacing
    # For n circles in unit disk, approximate spacing
    area = np.pi
    r_est = np.sqrt(area / n) * 0.9
    spacing = 2 * r_est * 1.1  # slightly larger to ensure we get enough points
    
    best_pts = None
    for attempt in range(50):
        s = spacing * (1.0 - attempt * 0.01)
        if s < 0.01:
            break
        # Generate hex lattice points
        k = int(2.0 / s) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = i * s + (j % 2) * s * 0.5
                y = j * s * np.sqrt(3) / 2
                if x*x + y*y <= (1.0 - s/2)**2:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= n:
            # Select n closest to origin
            dists = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(dists)[:n]
            best_pts = pts[idx]
            break
    
    if best_pts is None:
        # Fallback: random points in disk
        best_pts = np.random.uniform(-0.9, 0.9, (n, 2))
        mask = np.hypot(best_pts[:, 0], best_pts[:, 1]) <= 0.9
        while mask.sum() < n:
            extra = np.random.uniform(-0.9, 0.9, (n - mask.sum(), 2))
            mask_extra = np.hypot(extra[:, 0], extra[:, 1]) <= 0.9
            extra = extra[mask_extra]
            best_pts = np.vstack([best_pts[mask], extra])
            mask = np.ones(len(best_pts), dtype=bool)
        best_pts = best_pts[:n]
    
    return best_pts


def _compute_score(pts):
    """Compute the packing score: min of all pairwise distances/2 and boundary distances."""
    # Pairwise distances
    from scipy.spatial.distance import pdist
    min_pair = pdist(pts, 'euclidean').min() / 2.0
    
    # Boundary distances
    radii = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = (1.0 - radii).min()
    
    return min(min_pair, min_boundary)


def _relax(pts, max_iter=5000, tol=1e-10):
    """Gradient-based relaxation to maximize the minimum distance."""
    n = len(pts)
    pts = pts.copy().astype(np.float64)
    
    # Use a smooth approximation of the min function
    # We'll use a repulsion force approach
    dt = 0.001
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute forces: repulsion between all pairs + attraction to boundary
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion (only for close pairs to save time)
        tree = cKDTree(pts)
        # Find close pairs
        pairs = tree.query_pairs(r=0.2)  # Only consider pairs within 0.2
        
        for i, j in pairs:
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.array([1e-10, 0.0])
                dist = 1e-10
            # Repulsive force, stronger when closer
            f = d / (dist ** 2)
            forces[i] += f
            forces[j] -= f
        
        # Boundary constraint: push points inward if too close to boundary
        radii = np.hypot(pts[:, 0], pts[:, 1])
        for i in range(n):
            if radii[i] > 0.85:  # Close to boundary
                # Push toward center
                forces[i] -= pts[i] * 0.5
        
        # Normalize forces
        force_mags = np.linalg.norm(forces, axis=1)
        max_fm = force_mags.max()
        if max_fm > 1e-15:
            forces /= max_fm
        
        # Update positions
        new_pts = pts + dt * forces
        
        # Project back into disk
        new_radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
        for i in range(n):
            if new_radii[i] > 0.999:
                scale = 0.999 / new_radii[i]
                new_pts[i] *= scale
        
        # Check if score improved
        new_score = _compute_score(new_pts)
        if new_score > best_score - tol:
            pts = new_pts
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
        
        # Adaptive step size
        if new_score > best_score:
            dt = min(dt * 1.05, 0.01)
        else:
            dt *= 0.9
        
        if dt < 1e-8:
            break
    
    return best_pts, best_score


def _polish(pts, max_iter=2000):
    """Final high-precision polish pass."""
    n = len(pts)
    pts = pts.copy().astype(np.float64)
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    dt = 0.0005
    start_time = time.time()
    deadline = start_time + 20.0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        tree = cKDTree(pts)
        pairs = tree.query_pairs(r=0.15)
        
        for i, j in pairs:
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                d = np.array([1e-12, 0.0])
                dist = 1e-12
            f = d / (dist ** 2)
            forces[i] += f
            forces[j] -= f
        
        # Boundary constraint
        radii = np.hypot(pts[:, 0], pts[:, 1])
        for i in range(n):
            if radii[i] > 0.9:
                forces[i] -= pts[i] * 0.3
        
        force_mags = np.linalg.norm(forces, axis=1)
        max_fm = force_mags.max()
        if max_fm > 1e-15:
            forces /= max_fm
        
        new_pts = pts + dt * forces
        
        # Project back into disk
        new_radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
        for i in range(n):
            if new_radii[i] > 0.9999:
                scale = 0.9999 / new_radii[i]
                new_pts[i] *= scale
        
        new_score = _compute_score(new_pts)
        if new_score > best_score - 1e-12:
            pts = new_pts
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
        
        if new_score > best_score:
            dt = min(dt * 1.02, 0.002)
        else:
            dt *= 0.95
        
        if dt < 1e-10:
            break
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    total_deadline = start_time + 120.0
    
    # Initialize with hex lattice
    pts = _hex_init(N)
    
    # Relaxation phase
    pts, score = _relax(pts, max_iter=8000, tol=1e-11)
    
    # Polish phase
    if time.time() < total_deadline - 25.0:
        pts, score = _polish(pts, max_iter=3000)
    
    # Final validation: ensure all points are strictly inside
    radii = np.hypot(pts[:, 0], pts[:, 1])
    pts[radii > 0.99999] *= 0.99999 / radii[radii > 0.99999]
    
    return pts
# EVOLVE-BLOCK-END
