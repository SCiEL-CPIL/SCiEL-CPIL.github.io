"""Pack 200 equal circles in a unit square using discrete greedy selection on a fine grid."""
import numpy as np
import time

N = 200
_START_TIME = time.time()
_TIME_LIMIT = 110  # seconds, leave margin before 120s kill

def _get_radius(pts):
    """Calculate the maximum possible radius for given centers."""
    if len(pts) == 0:
        return 0.0
    
    # Distance to walls
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])
    
    wall_dist = min(min_x + 0.5, 0.5 - max_x, min_y + 0.5, 0.5 - max_y)
    
    # Min pairwise distance
    # Using cdist for efficiency
    from scipy.spatial.distance import cdist
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair_dist = np.min(D)
    
    return min(wall_dist, min_pair_dist / 2.0)

def _greedy_pack(grid_pts, N):
    """
    Greedy selection of N points from grid_pts to maximize min distance.
    grid_pts: (M, 2) array of candidate points.
    Returns: (N, 2) array of selected points.
    """
    M = len(grid_pts)
    if M == 0:
        return np.array([])
    
    # Initialize min distances to walls for all candidates
    # dist to left = x + 0.5, right = 0.5 - x, etc.
    dist_walls = np.min(np.stack([
        grid_pts[:, 0] + 0.5,
        0.5 - grid_pts[:, 0],
        grid_pts[:, 1] + 0.5,
        0.5 - grid_pts[:, 1]
    ], axis=1), axis=1)
    
    current_min_dists = dist_walls.copy()
    
    selected = []
    available_mask = np.ones(M, dtype=bool)
    
    for _ in range(N):
        if not np.any(available_mask):
            break
            
        # Find the candidate with the maximum current_min_dist
        # We only consider available candidates
        # To handle this efficiently, we can set unavailable ones to -inf
        dists_to_consider = np.where(available_mask, current_min_dists, -np.inf)
        
        best_idx = np.argmax(dists_to_consider)
        
        if dists_to_consider[best_idx] <= 0:
             # No valid point left
             break
             
        selected.append(best_idx)
        
        # Mark as unavailable
        available_mask[best_idx] = False
        
        # Update min distances for all remaining available candidates
        # Distance from new point to all candidates
        new_point = grid_pts[best_idx]
        diffs = grid_pts - new_point
        dists_to_new = np.linalg.norm(diffs, axis=1)
        
        # Update current_min_dists
        # We only care about available ones, but vectorized update is fine
        current_min_dists = np.minimum(current_min_dists, dists_to_new)
        
        # Optional: Prune candidates that are now too close to be useful?
        # Since we take argmax, and we want to maximize the min distance,
        # any candidate with a new min_dist < current_best_possible could be pruned,
        # but determining "current_best_possible" is tricky.
        # Given M is not huge, we skip pruning for simplicity and speed of implementation.
        
    return grid_pts[selected]

def _generate_hex_grid(spacing, offset_x=0.0, offset_y=0.0):
    """Generate a hexagonal grid of points inside the square."""
    pts = []
    # Hex grid parameters
    # Row height = spacing * sqrt(3)/2
    # Col width = spacing
    # Odd rows shifted by spacing/2
    
    row_h = spacing * np.sqrt(3) / 2
    col_w = spacing
    
    # Determine range
    # y from -0.5 to 0.5
    # x from -0.5 to 0.5
    
    y = -0.5 + offset_y
    row_idx = 0
    while y <= 0.5:
        shift = (col_w / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.5 + offset_x + shift
        while x <= 0.5:
            # Check if inside square (with small margin to avoid boundary issues in eval)
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append((x, y))
            x += col_w
        y += row_h
        row_idx += 1
        
    if not pts:
        return np.array([])
    return np.array(pts)

def construct():
    t0 = time.time()
    
    best_r = 0.0
    best_pts = None
    
    # Parameters for grid generation
    # We want a fine grid.
    # If spacing is too large, we miss optimal positions.
    # If spacing is too small, M grows, and greedy gets slower.
    # M ~ Area / (spacing^2 * sqrt(3)/2) ~ 1 / (0.866 * s^2)
    # If s = 0.01, M ~ 1/0.0000866 ~ 11500.
    # If s = 0.005, M ~ 46000.
    # Greedy cost: N * M vector ops.
    # 200 * 46000 = 9.2M ops. Very fast.
    
    base_spacing = 0.015 # Start with a reasonable spacing
    # We will try multiple offsets to break symmetry
    
    num_trials = 50
    # We can also vary spacing slightly
    
    for trial in range(num_trials):
        if time.time() - t0 > _TIME_LIMIT:
            break
            
        # Random small offset to break grid symmetry
        # Offset should be small, within half spacing
        off_x = np.random.uniform(-base_spacing/2, base_spacing/2)
        off_y = np.random.uniform(-base_spacing/2, base_spacing/2)
        
        # Occasionally try a finer grid
        if trial % 5 == 0:
            spacing = base_spacing * 0.8
        else:
            spacing = base_spacing
            
        grid_pts = _generate_hex_grid(spacing, off_x, off_y)
        
        if len(grid_pts) < N:
            continue
            
        # Run greedy packing
        selected_pts = _greedy_pack(grid_pts, N)
        
        if len(selected_pts) < N:
            continue
            
        # Calculate radius
        r = _get_radius(selected_pts)
        
        if r > best_r:
            best_r = r
            best_pts = selected_pts.copy()
            
        # If we are getting very close to the known best, we might want to
        # do a local refinement, but the discrete method is quite strong.
        # Let's just keep trying different grids.
        
    # Fallback if something went wrong
    if best_pts is None:
        # Simple grid fallback
        rows, cols = 14, 15
        dx = 1.0 / cols
        dy = 1.0 / rows
        xs = -0.5 + (np.arange(cols) + 0.5) * dx
        ys = -0.5 + (np.arange(rows) + 0.5) * dy
        grid = np.array([(x, y) for y in ys for x in xs])
        best_pts = grid[:N]
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
