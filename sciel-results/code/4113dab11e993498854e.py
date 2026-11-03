"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _score(positions):
    """Compute the radius for a given set of positions."""
    n = len(positions)
    if n == 0:
        return 0.0
    
    # Compute pairwise distances
    # For efficiency, we can use broadcasting but N=200 is small enough
    # d[i, j] = distance between i and j
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    min_dist = np.min(dists[mask])
    
    # Distance to boundary
    # positions are in [-0.5, 0.5]
    # min_k (0.5 - |pos[i, k]|)
    # This is equivalent to min(0.5 - x, 0.5 + x) = 0.5 - |x|
    # So for each point, dist_to_boundary = min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y)
    # Which is min(0.5 - |x|, 0.5 - |y|)
    abs_pos = np.abs(positions)
    dist_to_bound = np.min(0.5 - abs_pos, axis=1)
    min_bound = np.min(dist_to_bound)
    
    r = min(min_dist / 2.0, min_bound)
    return r


def _construct_grid(m=14):
    """Create a grid of m x m points, pick N best."""
    # We want to pack N=200 circles.
    # A 14x14 grid has 196 points.
    # A 15x15 grid has 225 points.
    # Let's try 14x14 plus some extra points?
    # Or just use a 15x15 grid and drop 25?
    # Or use a 14x14 grid and add 4 points?
    # Actually, 14x14 is 196. We need 200.
    # Let's use a 15x15 grid and drop the 25 worst points (e.g., center ones)?
    # Or just use a 15x15 grid and drop 25 random ones?
    # Better: use a 15x15 grid, but shift it?
    
    # Let's try a 14x14 grid first, it's a known good packing for 196.
    # For 200, we can take a 14x14 grid and add 4 points in the gaps?
    # Or take a 15x15 grid and drop 25 points.
    
    # Let's generate a 15x15 grid.
    ticks = (np.arange(15) + 0.5) / 15.0 - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    # We need to select 200 points from 225.
    # The points in the center are more "crowded" in terms of potential for improvement?
    # Actually, in a square grid, the radius is determined by the spacing (1/15) and boundary (1/30).
    # Spacing = 1/15 = 0.0666. Radius = 0.0333.
    # Boundary dist = 1/30 = 0.0333.
    # So r = 0.0333.
    
    # If we drop points, we can increase r.
    # Which points to drop?
    # Dropping corner points increases boundary distance for remaining points? No, corners are far from boundary.
    # Dropping center points increases min distance?
    
    # Let's just return the 14x14 grid + 4 extra points?
    # 14x14 grid:
    ticks14 = (np.arange(14) + 0.5) / 14.0 - 0.5
    grid14 = np.array([(x, y) for y in ticks14 for x in ticks14])
    # 196 points.
    # Add 4 points? Where?
    # Maybe in the middle of the 4 large squares formed by 14x14?
    # The gaps are at (x, y) where x, y are midpoints between grid points.
    # Midpoints: (i + 0.5)/14 - 0.5 for i in 0..13? No, between i and i+1.
    # x = (i + 0.5)/14 - 0.5 is the grid point.
    # Midpoint between i and i+1 is (i+1)/14 - 0.5.
    # Let's add points at (0, 0), (0.25, 0.25), etc?
    
    # Actually, let's just use the 15x15 grid and drop 25 points.
    # We can drop the 25 points closest to the center?
    # Or drop random points?
    # Let's drop the 25 points with the highest "potential" to be moved?
    # For now, let's just drop the 25 points closest to the origin.
    
    dists_to_origin = np.sum(grid**2, axis=1)
    indices_to_drop = np.argsort(dists_to_origin)[:25]
    mask = np.ones(len(grid), dtype=bool)
    mask[indices_to_drop] = False
    return grid[mask]


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    # Initial guess: 15x15 grid with 25 center points dropped
    x0 = _construct_grid(15)
    
    # Shuffle x0 slightly to break symmetry?
    # x0 = x0 + 0.001 * np.random.randn(*x0.shape)
    
    best_x = x0
    best_r = _score(best_x)
    
    # Objective function for scipy: minimize -r
    def objective(x):
        x = x.reshape(-1, 2)
        # Check validity?
        # If any point is outside, return a large penalty
        if np.any(np.abs(x) > 0.5):
            return 1e6
        r = _score(x)
        return -r
    
    # Gradient?
    # We can use numerical gradient, but it's slow.
    # Let's try L-BFGS-B with numerical gradient.
    
    x_flat = x0.flatten()
    
    # We will run a few restarts with different initializations
    # 1. Grid 15x15 minus 25
    # 2. Grid 14x14 plus 4 (maybe random)
    # 3. Random
    
    results = []
    
    # Run 1: Grid 15x15 minus 25
    try:
        res = minimize(objective, x_flat, method='L-BFGS-B', options={'maxiter': 500, 'gtol': 1e-5})
        if res.success or res.fun < -best_r:
            r = -res.fun
            if r > best_r:
                best_r = r
                best_x = res.x.reshape(-1, 2)
        results.append((best_r, best_x))
    except Exception as e:
        pass
    
    if time.time() > deadline:
        return best_x
    
    # Run 2: Grid 14x14 plus 4 random points
    ticks14 = (np.arange(14) + 0.5) / 14.0 - 0.5
    grid14 = np.array([(x, y) for y in ticks14 for x in ticks14])
    # Add 4 random points
    extra = 0.05 * np.random.randn(4, 2)
    x0_2 = np.vstack([grid14, extra])
    # Shuffle
    np.random.shuffle(x0_2)
    x_flat_2 = x0_2.flatten()
    
    try:
        res = minimize(objective, x_flat_2, method='L-BFGS-B', options={'maxiter': 500, 'gtol': 1e-5})
        if res.success or res.fun < -best_r:
            r = -res.fun
            if r > best_r:
                best_r = r
                best_x = res.x.reshape(-1, 2)
    except Exception as e:
        pass
    
    if time.time() > deadline:
        return best_x
    
    # Run 3: Random
    x0_3 = 0.4 * np.random.randn(N, 2)
    # Clip to square
    x0_3 = np.clip(x0_3, -0.45, 0.45)
    x_flat_3 = x0_3.flatten()
    
    try:
        res = minimize(objective, x_flat_3, method='L-BFGS-B', options={'maxiter': 500, 'gtol': 1e-5})
        if res.success or res.fun < -best_r:
            r = -res.fun
            if r > best_r:
                best_r = r
                best_x = res.x.reshape(-1, 2)
    except Exception as e:
        pass
    
    if time.time() > deadline:
        return best_x
    
    # Run 4: Hexagonal-ish grid
    # Generate a hexagonal packing pattern
    # Spacing s
    # Rows of points
    # Let's try to generate a hex grid and pick N points
    # s = 0.07
    # This is complex, let's skip for now and use more random restarts
    
    # Run 5: Another random
    x0_5 = 0.4 * np.random.randn(N, 2)
    x0_5 = np.clip(x0_5, -0.45, 0.45)
    x_flat_5 = x0_5.flatten()
    
    try:
        res = minimize(objective, x_flat_5, method='L-BFGS-B', options={'maxiter': 500, 'gtol': 1e-5})
        if res.success or res.fun < -best_r:
            r = -res.fun
            if r > best_r:
                best_r = r
                best_x = res.x.reshape(-1, 2)
    except Exception as e:
        pass
    
    # Final check
    # Ensure all points are inside
    best_x = np.clip(best_x, -0.5, 0.5)
    
    return best_x
# EVOLVE-BLOCK-END
