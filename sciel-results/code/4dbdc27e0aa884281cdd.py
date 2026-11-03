"""Pack 200 equal circles in the unit square using greedy hex-grid search plus local refinement."""
import numpy as np
import time

N = 200
DEADLINE = 115.0  # wall-clock safety margin


def _score(pts):
    """Compute the common radius for a set of centers in [-0.5, 0.5]^2."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Distance to boundary
    min_coord = pts.min(axis=0)
    max_coord = pts.max(axis=0)
    r_bound = min(0.5 - max_coord[0], max_coord[0] + 0.5,
                  0.5 - max_coord[1], max_coord[1] + 0.5)
    if r_bound < 0:
        return 0.0
    # Pairwise distances (vectorized, chunked to limit memory)
    chunk = 50
    min_pair = float('inf')
    for i in range(0, N, chunk):
        for j in range(i + 1, N, chunk):
            sub_i = pts[i:i + chunk]
            sub_j = pts[j:j + chunk]
            # For the diagonal chunk where i==j, handle self-distances
            if i == j:
                diff = sub_i[:, np.newaxis, :] - sub_i[np.newaxis, :, :]
                d = np.sqrt(np.sum(diff ** 2, axis=-1))
                np.fill_diagonal(d, np.inf)
                min_pair = min(min_pair, d.min())
            else:
                diff = sub_i[:, np.newaxis, :] - sub_j[np.newaxis, :, :]
                d = np.sqrt(np.sum(diff ** 2, axis=-1))
                min_pair = min(min_pair, d.min())
    return min(r_bound, min_pair / 2.0)


def _greedy_hex_grid(n_trials=30, grid_n=40):
    """Greedy placement on a hexagonal grid with random offsets."""
    best_pts = None
    best_r = 0.0
    # Hex grid spacing
    # We'll place circles one at a time, choosing the grid position (with offset)
    # that maximizes the minimum distance to existing circles and boundary.
    
    # Precompute hex grid base positions
    # For a hex grid with spacing s in a unit square
    dx = 1.0 / grid_n  # approximate spacing
    # Hex grid: rows offset by dx/2
    y_positions = np.arange(0, grid_n + 2) * (dx * np.sqrt(3) / 2)
    x_positions = np.arange(0, grid_n + 2) * dx
    
    # Generate candidate positions (hex grid)
    candidates = []
    for yi, y in enumerate(y_positions):
        x_offset = (dx / 2.0) if yi % 2 else 0.0
        for x in x_positions:
            px = x + x_offset
            py = y
            if -0.5 <= px <= 0.5 and -0.5 <= py <= 0.5:
                candidates.append((px - 0.5, py - 0.5))  # shift to [-0.5, 0.5]
    
    candidates = np.array(candidates)
    
    for trial in range(n_trials):
        # Random offset
        ox = np.random.uniform(-dx / 4, dx / 4)
        oy = np.random.uniform(-dx / 4, dx / 4)
        offset_pts = candidates + np.array([ox, oy])
        
        # Filter to valid positions
        mask = np.all(np.abs(offset_pts) <= 0.5, axis=1)
        offset_pts = offset_pts[mask]
        
        if len(offset_pts) < N:
            continue
        
        # Greedy placement
        placed = []
        remaining = offset_pts.copy()
        
        for _ in range(N):
            # For each candidate, compute min distance to placed circles and boundary
            best_idx = -1
            best_min_dist = -1
            
            # Boundary distance for each candidate
            r_bound = np.minimum(
                np.minimum(0.5 - remaining[:, 0], remaining[:, 0] + 0.5),
                np.minimum(0.5 - remaining[:, 1], remaining[:, 1] + 0.5)
            )
            
            # Distance to placed circles
            if len(placed) == 0:
                min_pair = np.inf
                # Just pick the one with max boundary distance
                score_arr = r_bound.copy()
            else:
                placed_arr = np.array(placed)
                # Compute min distance from each candidate to any placed circle
                diff = remaining[:, np.newaxis, :] - placed_arr[np.newaxis, :, :]
                d = np.sqrt(np.sum(diff ** 2, axis=-1))
                min_pair = d.min(axis=1)
                score_arr = np.minimum(r_bound, min_pair / 2.0)
            
            best_idx = np.argmax(score_arr)
            best_min_dist = score_arr[best_idx]
            
            if best_min_dist <= 0:
                break
            
            placed.append(remaining[best_idx])
            remaining = np.delete(remaining, best_idx, axis=0)
        
        if len(placed) < N:
            continue
        
        pts = np.array(placed)
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    return best_pts, best_r


def _local_refinement(pts, max_iters=200):
    """Simple local refinement: move each point slightly toward its nearest neighbor direction."""
    if pts is None:
        return pts, 0.0
    
    best_r = _score(pts)
    pts = pts.copy()
    
    for it in range(max_iters):
        improved = False
        for i in range(N):
            orig = pts[i].copy()
            # Try small perturbations
            for dx in [-0.001, 0.001]:
                for dy in [-0.001, 0.001]:
                    pts[i] = orig + np.array([dx, dy])
                    if np.any(np.abs(pts[i]) > 0.5):
                        continue
                    r = _score(pts)
                    if r > best_r + 1e-12:
                        best_r = r
                        improved = True
                        break
                    pts[i] = orig
                if improved:
                    break
            if not improved:
                pts[i] = orig
        if not improved:
            break
    
    return pts, best_r


def construct():
    t0 = time.time()
    
    # Start with greedy hex grid search
    pts, r = _greedy_hex_grid(n_trials=20, grid_n=35)
    
    if time.time() - t0 > DEADLINE - 30:
        return pts if pts is not None else np.random.uniform(-0.45, 0.45, (N, 2))
    
    # Try a few more trials if time permits
    if time.time() - t0 < DEADLINE - 60:
        pts2, r2 = _greedy_hex_grid(n_trials=15, grid_n=40)
        if pts2 is not None and r2 > r:
            pts, r = pts2, r2
    
    if time.time() - t0 > DEADLINE - 20:
        return pts
    
    # Local refinement
    pts, r = _local_refinement(pts, max_iters=100)
    
    return pts
