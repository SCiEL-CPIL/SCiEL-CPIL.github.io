"""Improved packer: KD-tree accelerated local search with time guard."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 300
SEED = 12345
T_START = time.time()
DEADLINE = 110.0  # safety margin before the 120 s wall-clock limit


def _min_dist(points):
    """Efficiently compute the minimum pairwise distance."""
    tree = cKDTree(points)
    # query_knn returns distances to k nearest neighbors (excluding self if k>1)
    # We want the min distance to any other point, so k=2 (self + nearest)
    # Actually, cKDTree.query with k=2 gives self and nearest.
    # But it's faster to just query k=2 and take the min of the second column?
    # No, query(k=2) returns array of shape (N, 2). Col 0 is 0, Col 1 is dist to nearest.
    # We need the global min of Col 1.
    dists, _ = tree.query(points, k=2)
    return float(np.min(dists[:, 1]))


def _score(points):
    """Return the feasible common radius (0.0 if invalid)."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # distance to container boundary
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - norm
    if np.min(r_bound) < 0.0:
        return 0.0
    # minimum pairwise half-distance
    min_pair_dist = _min_dist(points)
    r_pair = 0.5 * min_pair_dist
    r = min(r_bound.min(), r_pair)
    return float(r) if r > 0.0 else 0.0


def _lattice_seed(spacing):
    """Generate a hexagonal lattice and pick N points inside the disk."""
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        keep = rad <= 1.0 - spacing / 2.0
        pts = pts[keep]
        if len(pts) >= N:
            # prefer inner points (smaller radius) to maximise margin
            order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
            return pts[order[:N]].copy()
        spacing *= 0.97


def _local_search(points, t_budget, rng):
    """Fast local search (hill climbing) using KD-tree for distance checks."""
    start = time.time()
    best = points.copy()
    best_r = _score(best)
    cur = points.copy()
    cur_r = best_r
    
    # Initial step size based on typical spacing
    # Approximate radius is ~0.05, so spacing ~0.1
    step = 0.01
    
    # Keep track of recent points for collision checking if we wanted to be smarter,
    # but for now we just re-evaluate the whole min dist. 
    # To speed up, we can check only the moved point's neighbors.
    
    tree = cKDTree(cur)
    
    while time.time() - start < t_budget:
        # Propose move
        idx = rng.integers(N)
        ang = rng.random() * 2.0 * np.pi
        mag = step * (0.5 + rng.random()) # random magnitude in [0.5*step, 1.5*step]
        dx = mag * np.cos(ang)
        dy = mag * np.sin(ang)
        
        old_pt = cur[idx].copy()
        new_pt = old_pt + np.array([dx, dy])
        
        # Check boundary
        if np.hypot(new_pt[0], new_pt[1]) > 1.0:
            continue
            
        # Check collision with neighbors
        # We need to ensure dist(new_pt, cur[j]) > 2*r_cur for all j != idx
        # Actually, we want to maximize r. 
        # Strategy: Move a point that is part of a "tight" cluster to relax it.
        # Simple hill climbing: Accept if min_dist increases.
        
        # Efficient check:
        # 1. Remove idx from tree (or just query excluding idx)
        # 2. Query new_pt against tree.
        # cKDTree.query(new_pt, k=1) gives nearest neighbor.
        # If nearest neighbor is idx itself (dist 0), we need k=2.
        # Since we are moving idx, we should query against the OTHER points.
        
        # Rebuilding tree every step is O(N log N). N=300 is small.
        # Let's try rebuilding every step first. If too slow, optimize.
        
        cur[idx] = new_pt
        new_min_dist = _min_dist(cur)
        new_r = 0.5 * new_min_dist
        
        # Also check boundary constraint for the new point
        new_r_bound = 1.0 - np.hypot(new_pt[0], new_pt[1])
        new_r = min(new_r, new_r_bound)
        
        if new_r > cur_r:
            cur_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = cur.copy()
            # Decrease step size slightly on success? Or keep it.
            step *= 0.99
        else:
            # Revert
            cur[idx] = old_pt
            # Increase step size slightly on failure? Or keep it.
            step *= 1.01
            if step > 0.1: step = 0.1
            if step < 0.001: step = 0.001

    return best, best_r


def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # initial hex-lattice seed
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    pts = _lattice_seed(spacing)
    best = pts.copy()
    best_r = _score(best)

    # Multi-start local search
    while time.time() - t0 < DEADLINE:
        # Perturb best solution
        jitter = 0.02 * rng.standard_normal((N, 2))
        cand = best + jitter
        
        # Project back into unit circle if needed
        rad = np.linalg.norm(cand, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            cand[mask] = cand[mask] / rad[mask, None] * (1.0 - 1e-6)

        t_left = DEADLINE - (time.time() - t0)
        if t_left < 5.0:
            break
        t_alloc = min(t_left - 2.0, 10.0)
        
        cand, r = _local_search(cand, t_alloc, rng)
        if r > best_r:
            best_r = r
            best = cand

    # Final safety
    final_r = _score(best)
    if final_r <= 0.0:
        best = _lattice_seed(2.0 / (np.sqrt(N) + 1.0))

    return best
