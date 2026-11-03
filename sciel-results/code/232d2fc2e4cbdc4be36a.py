"""Pack 200 equal circles in a unit square using force-directed optimization with cooling."""
import numpy as np
import time
from scipy.spatial import KDTree

N = 200
_TIME_LIMIT = 135  # seconds, leave margin before 150s kill


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
    tree = KDTree(pts)
    pairs = tree.query_pairs(r=1.0)  # pairs within distance 1
    if len(pairs) == 0:
        min_pair_dist = np.inf
    else:
        min_pair_dist = np.inf
        for i, j in pairs:
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_pair_dist:
                min_pair_dist = d
        if min_pair_dist == np.inf:
            # Fallback: compute all pairs if query_pairs didn't find any close ones
            # This shouldn't happen for N=200 in unit square
            from scipy.spatial.distance import cdist
            D = cdist(pts, pts)
            np.fill_diagonal(D, np.inf)
            min_pair_dist = np.min(D)

    return min(wall_dist, min_pair_dist / 2.0)


def _compute_forces(pts, r_target, repulsion_strength=10.0, wall_strength=5.0):
    """
    Compute forces on each point.
    - Repulsive force between pairs closer than 2*r_target
    - Attractive force to keep points within bounds (soft walls)
    - Also a small attractive force to maintain spacing (prevents collapse)
    """
    n = len(pts)
    forces = np.zeros_like(pts)
    
    # Build KDTree for neighbor queries
    tree = KDTree(pts)
    
    # For each point, find neighbors within some radius
    # Use a reasonable cutoff
    cutoff = max(2 * r_target, 0.05)
    
    for i in range(n):
        # Wall forces: push points away from boundaries
        # Left wall
        if pts[i, 0] < -0.5 + r_target:
            forces[i, 0] += wall_strength * (r_target - (pts[i, 0] + 0.5))
        # Right wall
        if pts[i, 0] > 0.5 - r_target:
            forces[i, 0] -= wall_strength * (pts[i, 0] - (0.5 - r_target))
        # Bottom wall
        if pts[i, 1] < -0.5 + r_target:
            forces[i, 1] += wall_strength * (r_target - (pts[i, 1] + 0.5))
        # Top wall
        if pts[i, 1] > 0.5 - r_target:
            forces[i, 1] -= wall_strength * (pts[i, 1] - (0.5 - r_target))
        
        # Repulsive forces from neighbors
        neighbors = tree.query_ball_point(pts[i], cutoff)
        for j in neighbors:
            if i == j:
                continue
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Random nudge for coincident points
                forces[i] += np.random.randn(2) * 0.01
                continue
            if dist < 2 * r_target:
                # Repulsive force: stronger when closer
                overlap = 2 * r_target - dist
                force_mag = repulsion_strength * overlap / dist
                forces[i] += force_mag * diff
    
    return forces


def _optimize_from_initial(pts, r_target, max_iters=5000, learning_rate=0.001, cooling=0.999):
    """
    Force-directed optimization to maximize the minimum distance.
    We try to push all pairs to be at least 2*r_target apart.
    """
    pts = pts.copy()
    current_lr = learning_rate
    
    for it in range(max_iters):
        # Check time
        if time.time() > _TIME_LIMIT - 5:
            break
        
        # Compute forces
        forces = _compute_forces(pts, r_target)
        
        # Update positions
        pts += current_lr * forces
        
        # Project back to square (hard constraint)
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        
        # Cool down
        current_lr *= cooling
        
        # Early stopping if forces are very small
        if np.max(np.abs(forces)) < 1e-6:
            break
    
    return pts


def _hex_grid_initial(spacing=0.08, offset_x=0.0, offset_y=0.0):
    """Generate a hexagonal grid of points inside the square."""
    pts = []
    row_h = spacing * np.sqrt(3) / 2
    col_w = spacing
    
    y = -0.5 + offset_y
    row_idx = 0
    while y <= 0.5:
        shift = (col_w / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.5 + offset_x + shift
        while x <= 0.5:
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append((x, y))
            x += col_w
        y += row_h
        row_idx += 1
        
    if not pts:
        return np.array([])
    return np.array(pts)


def _select_best_subset(pts, N):
    """
    From a larger set of points, select N points that maximize the minimum distance.
    Use greedy selection.
    """
    if len(pts) <= N:
        return pts
    
    selected = []
    available = set(range(len(pts)))
    
    for _ in range(N):
        if not available:
            break
        
        # Find the point that maximizes the minimum distance to already selected points
        best_idx = None
        best_min_dist = -1
        
        for idx in available:
            if not selected:
                min_dist = np.inf
            else:
                min_dist = np.min([np.linalg.norm(pts[idx] - pts[s]) for s in selected])
            
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        
        selected.append(best_idx)
        available.discard(best_idx)
    
    return pts[selected]


def construct():
    t0 = time.time()
    
    best_r = 0.0
    best_pts = None
    
    # Strategy: 
    # 1. Generate several initial configurations (hex grids with different offsets/spacings)
    # 2. For each, run force-directed optimization
    # 3. Keep the best result
    
    np.random.seed(42)
    
    # Different initial configurations to try
    configs = []
    
    # Hex grids with various parameters
    for spacing in [0.06, 0.07, 0.08, 0.09, 0.10]:
        for _ in range(3):
            off_x = np.random.uniform(-spacing/2, spacing/2)
            off_y = np.random.uniform(-spacing/2, spacing/2)
            configs.append(('hex', spacing, off_x, off_y))
    
    # Random configurations
    for _ in range(5):
        configs.append(('random', None, None, None))
    
    # Grid configurations
    for rows in [10, 12, 14, 15]:
        for cols in [10, 12, 14, 15]:
            if rows * cols >= N:
                configs.append(('grid', rows, cols, None))
    
    # Shuffle configs
    np.random.shuffle(configs)
    
    for config in configs:
        if time.time() - t0 > _TIME_LIMIT - 10:
            break
        
        cfg_type = config[0]
        
        if cfg_type == 'hex':
            spacing = config[1]
            off_x = config[2]
            off_y = config[3]
            pts = _hex_grid_initial(spacing, off_x, off_y)
        elif cfg_type == 'random':
            pts = np.random.uniform(-0.5, 0.5, size=(N, 2))
        elif cfg_type == 'grid':
            rows = config[1]
            cols = config[2]
            dx = 1.0 / cols
            dy = 1.0 / rows
            xs = -0.5 + (np.arange(cols) + 0.5) * dx
            ys = -0.5 + (np.arange(rows) + 0.5) * dy
            grid = np.array([(x, y) for y in ys for x in xs])
            # Select N points from grid
            if len(grid) >= N:
                # Use greedy selection
                pts = _select_best_subset(grid, N)
            else:
                continue
        else:
            continue
        
        if len(pts) < N:
            continue
        
        # If we have more than N points, select the best N
        if len(pts) > N:
            pts = _select_best_subset(pts, N)
        
        # Compute initial radius
        r_init = _get_radius(pts)
        
        # Try to improve with force-directed optimization
        # Target radius slightly above current
        r_target = r_init * 1.05
        
        # Run optimization
        optimized_pts = _optimize_from_initial(pts, r_target, max_iters=3000, learning_rate=0.0005, cooling=0.9995)
        
        # Compute final radius
        r_final = _get_radius(optimized_pts)
        
        # Also try with the original points (in case optimization made it worse)
        r_orig = r_init
        
        r_best = max(r_final, r_orig)
        pts_best = optimized_pts if r_final >= r_orig else pts
        
        if r_best > best_r:
            best_r = r_best
            best_pts = pts_best.copy()
        
        # If we're doing well, try more iterations
        if best_r > 0.036:
            # Try more aggressive optimization
            r_target2 = best_r * 1.02
            optimized_pts2 = _optimize_from_initial(best_pts, r_target2, max_iters=2000, learning_rate=0.0003, cooling=0.9998)
            r_final2 = _get_radius(optimized_pts2)
            if r_final2 > best_r:
                best_r = r_final2
                best_pts = optimized_pts2.copy()
    
    # Fallback if something went wrong
    if best_pts is None:
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
