"""Pack 200 equal circles in a unit square to maximize common radius."""
import numpy as np
from scipy.optimize import minimize
import time

N = 200
_START_TIME = time.time()
_TIME_LIMIT = 135  # seconds, leave margin before 150s kill


def _compute_radius(pts):
    """Compute the achievable radius for a set of points."""
    if len(pts) < N:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    if np.any(~np.isfinite(pts)):
        return 0.0
    # Check inside square
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return 0.0
    # Min distance to wall
    wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    # Min pairwise distance
    from scipy.spatial.distance import cdist
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    r = min(wall, min_pair / 2.0)
    return r if r > 0 else 0.0

def _force_directed_layout(pts, n_iters=2000, dt=0.01, repulsion_strength=1.0):
    """
    Use repulsive forces to spread out points.
    This is much faster than scipy.optimize for this problem size.
    """
    pts = pts.copy().astype(np.float64)
    for it in range(n_iters):
        # Compute all pairwise differences
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        # Compute squared distances
        dist_sq = np.sum(diff ** 2, axis=-1)
        # Avoid division by zero
        np.fill_diagonal(dist_sq, 1.0)
        
        # Compute repulsive forces
        dist = np.sqrt(dist_sq)
        force_mag = repulsion_strength / (dist_sq * dist)
        # Mask diagonal
        force_mag[np.diag_indices_from(force_mag)] = 0.0
        
        # Sum forces for each point
        forces = np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Update positions
        pts += dt * forces
        
        # Project back into square [-0.5, 0.5]
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
            
    return pts

def _push_apart(pts, n_iters=500, strength=0.005):
    """
    Iteratively push apart the closest pairs of points.
    """
    pts = pts.copy().astype(np.float64)
    for it in range(n_iters):
        from scipy.spatial.distance import cdist
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        min_idx = np.unravel_index(np.argmin(D), D.shape)
        i, j = min_idx
        min_dist = D[i, j]
        if min_dist < 1e-12:
            pts[j] += np.random.randn(2) * 0.01
            continue
        direction = pts[j] - pts[i]
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
             direction = np.random.randn(2)
             norm = 1.0
        direction /= norm
        move = strength * direction
        new_i = pts[i] - move
        new_j = pts[j] + move
        if -0.5 <= new_i[0] <= 0.5 and -0.5 <= new_i[1] <= 0.5 and \
           -0.5 <= new_j[0] <= 0.5 and -0.5 <= new_j[1] <= 0.5:
            pts[i] = new_i
            pts[j] = new_j
        else:
            for frac in [0.5, 0.25, 0.1]:
                move = strength * frac * direction
                new_i = pts[i] - move
                new_j = pts[j] + move
                if -0.5 <= new_i[0] <= 0.5 and -0.5 <= new_i[1] <= 0.5 and \
                   -0.5 <= new_j[0] <= 0.5 and -0.5 <= new_j[1] <= 0.5:
                    pts[i] = new_i
                    pts[j] = new_j
                    break
    return pts

def _best_result():
    """Fallback: 10x20 rectangular grid."""
    rows, cols = 20, 10
    dx = 1.0 / cols
    dy = 1.0 / rows
    xs = -0.5 + (np.arange(cols) + 0.5) * dx
    ys = -0.5 + (np.arange(rows) + 0.5) * dy
    grid = np.array([(x, y) for y in ys for x in xs])
    return grid[:N]

def construct():
    t0 = time.time()
    best_r = 0.0
    best_pts = _best_result()
    best_r = _compute_radius(best_pts)
    
    # Strategy 1: Force Directed Layout from Grid
    if time.time() - t0 < _TIME_LIMIT - 20:
        try:
            for rows, cols in [(20, 10), (14, 15), (16, 13), (13, 16), (10, 20)]:
                if time.time() - t0 > _TIME_LIMIT - 20:
                    break
                dx = 1.0 / cols
                dy = 1.0 / rows
                xs = -0.5 + (np.arange(cols) + 0.5) * dx
                ys = -0.5 + (np.arange(rows) + 0.5) * dy
                grid = np.array([(x, y) for y in ys for x in xs])[:N]
                
                np.random.seed(42)
                grid_jittered = grid + np.random.randn(N, 2) * 0.01
                grid_jittered = np.clip(grid_jittered, -0.5, 0.5)
                
                pts_fd = _force_directed_layout(grid_jittered, n_iters=3000, dt=0.005, repulsion_strength=0.5)
                r_fd = _compute_radius(pts_fd)
                
                if r_fd > best_r:
                    best_r = r_fd
                    best_pts = pts_fd
        except Exception:
            pass

    # Strategy 2: Push Apart Local Search
    if best_r > 0 and time.time() - t0 < _TIME_LIMIT - 10:
        try:
            pts_pushed = _push_apart(best_pts, n_iters=500, strength=0.005)
            r_pushed = _compute_radius(pts_pushed)
            if r_pushed > best_r:
                best_r = r_pushed
                best_pts = pts_pushed
        except Exception:
            pass

    # Strategy 3: Second pass of Push Apart
    if best_r > 0 and time.time() - t0 < _TIME_LIMIT - 5:
        try:
            pts_pushed2 = _push_apart(best_pts, n_iters=200, strength=0.002)
            r_pushed2 = _compute_radius(pts_pushed2)
            if r_pushed2 > best_r:
                best_r = r_pushed2
                best_pts = pts_pushed2
        except Exception:
            pass

    if best_r <= 0:
        return _best_result()
    return best_pts
