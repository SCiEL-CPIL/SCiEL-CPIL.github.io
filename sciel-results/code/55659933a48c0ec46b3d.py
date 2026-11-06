"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(points):
    """Calculate the radius for a given set of points."""
    if len(points) < 2:
        return 0.5
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = dists.min()
    
    # Wall distances
    wall_dists = np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1]))
    min_wall_dist = wall_dists.min()
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r


def _step(points, alpha=0.01):
    """Perform one gradient ascent step on the minimum radius."""
    N = len(points)
    if N < 2:
        return points, 0.0

    # Compute distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = dists.min()
    
    # Wall distances
    wall_dists = np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1]))
    min_wall_dist = wall_dists.min()
    
    pair_r = min_pair_dist / 2.0
    wall_r = min_wall_dist
    
    r = min(pair_r, wall_r)
    grad = np.zeros_like(points)
    
    thresh = 1e-8
    
    if pair_r < wall_r:
        # Bottleneck is a pair
        # Find all pairs with dist/2 close to pair_r
        # dists[i,j] / 2 <= pair_r + thresh
        active_mask = dists / 2.0 <= (pair_r + thresh)
        active_pairs = np.argwhere(active_mask)
        # Ensure i < j
        active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
        
        for i, j in active_pairs:
            d = dists[i, j]
            if d > 1e-10:
                v = (points[i] - points[j]) / d
                grad[i] += v
                grad[j] -= v
    else:
        # Bottleneck is a wall
        active_indices = np.argwhere(wall_dists <= (wall_r + thresh)).flatten()
        for k in active_indices:
            x, y = points[k]
            dl = 0.5 + x
            dr = 0.5 - x
            db = 0.5 + y
            dt = 0.5 - y
            min_w = min(dl, dr, db, dt)
            
            g = np.zeros(2)
            if min_w == dl:
                g[0] = 1.0
            elif min_w == dr:
                g[0] = -1.0
            elif min_w == db:
                g[1] = 1.0
            elif min_w == dt:
                g[1] = -1.0
            grad[k] += g
    
    # Update
    points = points + alpha * grad
    # Project to square
    points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
    points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
    
    return points, r


def _generate_hex(s, N):
    """Generate a hexagonal grid and select N points."""
    pts = []
    # Generate in a box
    range_x = np.arange(-0.6, 0.6, s)
    range_y = np.arange(-0.6, 0.6, s * np.sqrt(3) / 2)
    for i, y in enumerate(range_y):
        offset = (s / 2) if i % 2 else 0
        for x in range_x:
            pts.append((x + offset, y))
    pts = np.array(pts)
    if len(pts) == 0:
        return np.random.uniform(-0.4, 0.4, (N, 2))
        
    # Filter to square
    mask = (np.abs(pts[:, 0]) <= 0.5) & (np.abs(pts[:, 1]) <= 0.5)
    pts = pts[mask]
    
    if len(pts) > N:
        idx = np.random.choice(len(pts), N, replace=False)
        pts = pts[idx]
    elif len(pts) < N:
        # Add random points
        while len(pts) < N:
            p = np.random.uniform(-0.45, 0.45, 2)
            pts = np.vstack([pts, p])
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = 0.0
    best_points = None
    
    # Initializations
    inits = []
    
    # Hex grids with different spacings
    for s in [0.07, 0.072, 0.075, 0.078]:
        pts = _generate_hex(s, N)
        pts += np.random.normal(0, 0.005, pts.shape)
        pts = np.clip(pts, -0.5, 0.5)
        inits.append(pts)
        
    # Square grid
    m = 15
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    sq_pts = grid[idx]
    sq_pts += np.random.normal(0, 0.005, sq_pts.shape)
    sq_pts = np.clip(sq_pts, -0.5, 0.5)
    inits.append(sq_pts)
    
    # Random
    rand_pts = np.random.uniform(-0.45, 0.45, (N, 2))
    inits.append(rand_pts)
    
    for init in inits:
        if time.time() > deadline:
            break
            
        current_points = init.copy()
        current_score = _score(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
            
        # Optimization Loop
        # Use a decaying step size or fixed? Fixed is simpler.
        # Let's try 2000 steps.
        for step in range(2000):
            if time.time() > deadline:
                break
                
            current_points, current_score = _step(current_points, alpha=0.01)
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                
        if time.time() > deadline:
            break
            
    if best_points is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
        
    return best_points
# EVOLVE-BLOCK-END
