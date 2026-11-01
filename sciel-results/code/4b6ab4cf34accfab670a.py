"""Improved packing for 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def _score(pts):
    """Compute the radius for a given set of points."""
    if pts.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check bounds
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    # Compute min distance between any two points
    tree = cKDTree(pts)
    pairs = tree.query_pairs(1.0)
    if len(pairs) == 0:
        # No pairs within 1.0, so min dist is large, but we need actual min
        # Just compute all pairwise distances if few points, but N=200 so 19900 pairs
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
    else:
        min_pair_dist = np.inf
        for i, j in pairs:
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_pair_dist:
                min_pair_dist = d
    # Compute min distance to boundary
    dists_to_boundary = np.minimum(
        np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
        np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
    )
    min_boundary_dist = np.min(dists_to_boundary)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r


def _grid_init():
    """Create a grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    # Use a slightly denser grid to pick N points
    # For N=200, m=15 gives 225 points
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # Take first N points
    return grid[:N].copy()


def _jitter_init(scale=0.001, seed=42):
    """Add small random jitter to grid."""
    rng = np.random.default_rng(seed)
    pts = _grid_init()
    pts += rng.normal(0, scale, size=pts.shape)
    # Clip to bounds
    pts = np.clip(pts, -0.49, 0.49)
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with grid
    best_pts = _grid_init()
    best_score = _score(best_pts)
    
    # Try a few different initializations
    inits = [best_pts]
    for seed in range(10):
        pts = _jitter_init(scale=0.001, seed=seed)
        inits.append(pts)
    
    # Local optimization using gradient-free method (Nelder-Mead is slow for 400 dims)
    # Instead, use a simple iterative improvement:
    # For each point, try to move it to increase the minimum distance
    
    # Better approach: use scipy's minimize with a smooth approximation
    # But 400 dimensions is large. Let's try a simpler local search.
    
    # Actually, let's use a force-directed approach:
    # Repulsive forces between all pairs, attractive to grid positions
    
    for init_idx, pts0 in enumerate(inits):
        if time.time() > deadline:
            break
        
        pts = pts0.copy()
        current_score = _score(pts)
        
        # Force-directed relaxation
        # Parameters
        repulsion_strength = 1.0
        attraction_strength = 0.1
        learning_rate = 0.001
        max_iters = 500
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute forces
            forces = np.zeros_like(pts)
            
            # Repulsion between all pairs
            # For efficiency, use KDTree to find nearby pairs
            tree = cKDTree(pts)
            # Use a cutoff based on current radius * 2
            cutoff = max(0.02, current_score * 2.5)
            pairs = tree.query_pairs(cutoff)
            
            for i, j in pairs:
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-10:
                    # Force magnitude proportional to 1/dist^2
                    force = repulsion_strength / (dist * dist)
                    forces[i] += force * diff / dist
                    forces[j] -= force * diff / dist
            
            # Attraction to grid positions
            grid = _grid_init()
            for i in range(N):
                forces[i] += attraction_strength * (grid[i] - pts[i])
            
            # Update positions
            pts += learning_rate * forces
            
            # Keep points in bounds
            pts = np.clip(pts, -0.49, 0.49)
            
            # Check improvement
            new_score = _score(pts)
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
            else:
                # Reduce learning rate if no improvement
                learning_rate *= 0.95
                if learning_rate < 1e-6:
                    break
        
        # Final polish: try to increase the minimum distance by moving the most constrained points
        if time.time() > deadline:
            break
        
        # Identify the pair that determines the minimum distance
        tree = cKDTree(pts)
        pairs = tree.query_pairs(1.0)
        min_dist = np.inf
        min_pair = None
        for i, j in pairs:
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_dist:
                min_dist = d
                min_pair = (i, j)
        
        if min_pair is not None:
            i, j = min_pair
            # Try moving these two points apart
            diff = pts[i] - pts[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                direction = diff / norm
                # Move i in direction, j in opposite direction
                for step in [0.001, 0.0005, 0.0001]:
                    pts_try = pts.copy()
                    pts_try[i] += step * direction
                    pts_try[j] -= step * direction
                    pts_try = np.clip(pts_try, -0.49, 0.49)
                    try_score = _score(pts_try)
                    if try_score > current_score:
                        pts = pts_try
                        current_score = try_score
                        if current_score > best_score:
                            best_score = current_score
                            best_pts = pts.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
