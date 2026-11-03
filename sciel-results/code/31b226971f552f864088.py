"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(pts):
    """Compute the packing radius for a set of points in [-0.5, 0.5]^2."""
    # Wall constraint
    wall_r = 0.5 - np.max(np.abs(pts), axis=1)
    if wall_r.size == 0:
        return 0.0
    wall_min = float(np.min(wall_r))
    
    # Pairwise distances
    # Using broadcasting for O(N^2) but N=200 is fine
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = float(np.min(dists)) / 2.0
    
    return min(wall_min, min_pair)


def _hex_init(n):
    """Generate initial hexagonal packing positions."""
    # Hex grid: rows offset, spacing ~1 in x, ~sqrt(3)/2 in y
    # We want to fit as many as possible in a 1x1 square
    # Start with a rough hex grid and clip to square
    # Actually, for n=200, a hex grid with spacing s needs area ~ n * s^2 * 2/sqrt(3)
    # Area = 1, so s ~ sqrt(1 / (200 * 2/1.732)) ~ sqrt(1/230) ~ 0.065
    # Let's just generate a dense hex grid and pick the best n points? 
    # Simpler: generate hex grid points in [-0.5, 0.5]^2 and sort by distance to center, take first n?
    # Or just create a grid that roughly fits.
    
    # Let's create a hex grid covering the square.
    # Horizontal spacing: 1.0
    # Vertical spacing: sqrt(3)/2 * (horizontal spacing)
    # But we need to fit 200 points.
    # 200 points in a 1x1 square.
    # Hex grid density: 2/sqrt(3) per unit area of spacing s.
    # Number of points ~ (1/s)^2 * (2/sqrt(3))
    # 200 = (1/s)^2 * 1.155
    # (1/s)^2 = 173
    # 1/s = 13.1
    # s = 0.076
    
    s = 0.076
    points = []
    # y from -0.5 to 0.5
    # Row spacing dy = s * sqrt(3)/2
    dy = s * np.sqrt(3) / 2
    y = -0.5
    row = 0
    while y <= 0.5:
        x_offset = 0 if row % 2 == 0 else s / 2
        x = -0.5 + x_offset
        while x <= 0.5:
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append((x, y))
            x += s
        y += dy
        row += 1
    
    points = np.array(points)
    if len(points) < n:
        # If we didn't get enough, add random points
        extra = np.random.uniform(-0.5, 0.5, size=(n - len(points), 2))
        points = np.vstack([points, extra])
    else:
        # If we have too many, we can randomly select or sort by some criterion
        # For now, just take the first n? Or sort by distance to center to keep them centered?
        # Sorting by distance to center might create a compact cluster, which is good.
        d = np.sum(points ** 2, axis=1)
        idx = np.argsort(d)
        points = points[idx[:n]]
        
    return points


def _optimize(pts, time_limit=100.0):
    """Optimize the packing radius using gradient-free or simple gradient methods."""
    start_time = time.time()
    best_r = _compute_radius(pts)
    cur_r = best_r
    cur_pts = pts.copy()
    
    # Simple simulated annealing or hill climbing on radius
    # Since radius is a min function, it's non-smooth.
    # We can use a soft-min approximation for gradient, or just coordinate ascent.
    # Let's try coordinate ascent with random perturbations (simulated annealing style).
    
    T = 0.01  # Temperature
    T_min = 1e-6
    alpha = 0.999
    
    # We need to move points to increase the minimum radius.
    # This is equivalent to maximizing the minimum distance.
    # This is a max-min problem.
    
    # Let's use a simple iterative approach:
    # 1. Find the pair of points (or point-wall) that defines the current radius.
    # 2. Move the points in that pair apart.
    # 3. Check if any other constraint is violated (i.e., if another pair becomes closer).
    # This is essentially the "bottleneck elimination" strategy mentioned in the prompt.
    
    # To make it smoother, we can use a soft-min.
    # But let's stick to the discrete bottleneck approach for simplicity and robustness.
    
    # Pre-compute adjacency? No, just scan. N=200, N^2=40k, fast enough.
    
    while time.time() - start_time < time_limit:
        # Find the bottleneck
        diff = cur_pts[:, np.newaxis, :] - cur_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Find closest pair
        min_dist_idx = np.argmin(dists)
        i, j = np.unravel_index(min_dist_idx, dists.shape)
        min_dist = dists[i, j]
        
        # Check walls
        wall_dists = 0.5 - np.max(np.abs(cur_pts), axis=1)
        min_wall_idx = np.argmin(wall_dists)
        min_wall = wall_dists[min_wall_idx]
        
        # Determine which constraint is active
        # r_pair = min_dist / 2
        # r_wall = min_wall
        # If r_pair < r_wall, move i and j apart.
        # If r_wall < r_pair, move min_wall_idx away from wall.
        
        r_pair = min_dist / 2.0
        r_wall = min_wall
        
        if r_pair <= r_wall:
            # Move i and j apart
            d = cur_pts[i] - cur_pts[j]
            norm_d = np.linalg.norm(d)
            if norm_d > 1e-9:
                d = d / norm_d
            else:
                d = np.array([1.0, 0.0])
            
            # Step size
            # We want to increase min_dist.
            # Let's try a small step.
            step = 0.001
            # Move i in direction d, j in direction -d
            new_i = cur_pts[i] + step * d
            new_j = cur_pts[j] - step * d
            
            # Check if this is valid (inside square)
            if np.all(np.abs(new_i) < 0.5) and np.all(np.abs(new_j) < 0.5):
                cur_pts[i] = new_i
                cur_pts[j] = new_j
                new_r = _compute_radius(cur_pts)
                if new_r > cur_r:
                    cur_r = new_r
                    if cur_r > best_r:
                        best_r = cur_r
                        best_pts = cur_pts.copy()
                else:
                    # Revert if not improved
                    cur_pts[i] = new_i - step * d
                    cur_pts[j] = new_j + step * d
                    # Maybe try a smaller step or different direction?
                    # For now, just revert.
        else:
            # Move point min_wall_idx away from the wall
            p = cur_pts[min_wall_idx]
            # Direction away from wall
            # Wall is at x=0.5, x=-0.5, y=0.5, y=-0.5
            # Determine which wall is closest
            # Actually, wall_dists computed as 0.5 - max(abs(p))
            # So the closest wall is the one with the maximum abs coordinate.
            idx_max = np.argmax(np.abs(p))
            # If p[idx_max] > 0, move in -x direction (or -y)
            # If p[idx_max] < 0, move in +x direction (or +y)
            direction = -np.sign(p[idx_max])
            d = np.zeros(2)
            d[idx_max] = direction
            
            step = 0.001
            new_p = p + step * d
            if np.all(np.abs(new_p) < 0.5):
                cur_pts[min_wall_idx] = new_p
                new_r = _compute_radius(cur_pts)
                if new_r > cur_r:
                    cur_r = new_r
                    if cur_r > best_r:
                        best_r = cur_r
                        best_pts = cur_pts.copy()
                else:
                    cur_pts[min_wall_idx] = p
    
    return best_pts, best_r


def construct():
    # Start with hex init
    pts = _hex_init(N)
    # Optimize
    best_pts, best_r = _optimize(pts, time_limit=110.0)
    return best_pts
# EVOLVE-BLOCK-END
