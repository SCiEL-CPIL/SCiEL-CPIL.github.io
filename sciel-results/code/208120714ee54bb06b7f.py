"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

START_TIME = time.time()
TIME_LIMIT = 115.0  # Leave 5s margin for the 120s limit


def _compute_radius(c):
    """Compute the maximum radius for given centers."""
    # Distance to boundaries
    boundary_dist = 0.5 - np.abs(c).max(axis=1)
    # Pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = d.min()
    return min(boundary_dist.min(), min_pair_dist / 2.0)


def _initial_hex_grid(n, offset_x=0.0, offset_y=0.0, scale=1.0):
    """
    Generate a hexagonal grid of points in [-0.5, 0.5]^2.
    We generate a slightly larger grid and then select the best n points
    that are closest to the ideal hex positions, or just clip.
    To ensure we get exactly n points, we generate a grid with density
    slightly higher than needed and then prune the ones furthest from center
    or just take the first n.
    """
    points = []
    
    # Hexagonal packing parameters
    # Let spacing be 's'. Vertical distance is s * sqrt(3)/2.
    # Area per point ~ s^2 * sqrt(3)/2.
    # n points in area 1 -> s ~ sqrt(2 / (n * sqrt(3)))
    s = np.sqrt(2.0 / (n * np.sqrt(3.0))) * scale
    
    # We need to cover [-0.5, 0.5] + margin
    # Number of rows
    row_height = s * np.sqrt(3.0) / 2.0
    num_rows = int(np.ceil(1.0 / row_height)) + 4
    num_cols = int(np.ceil(1.0 / s)) + 4
    
    start_row = -num_rows // 2
    start_col = -num_cols // 2
    
    for i in range(start_row, start_row + num_rows):
        y = i * row_height + offset_y
        if y < -0.5 - s or y > 0.5 + s:
            continue
        x_offset = (s / 2.0) if i % 2 != 0 else 0.0
        for j in range(start_col, start_col + num_cols):
            x = j * s + x_offset + offset_x
            if x < -0.5 - s or x > 0.5 + s:
                continue
            # Clip to square
            cx = np.clip(x, -0.5, 0.5)
            cy = np.clip(y, -0.5, 0.5)
            points.append([cx, cy])
            
    points = np.array(points)
    
    # We have more than n points usually. We need to select n.
    # A good heuristic is to pick the ones that maximize the minimum distance.
    # But that's expensive. Instead, we can just take a random subset or 
    # the ones closest to the ideal lattice positions.
    # Since we generated a grid, the "ideal" positions are the grid points themselves.
    # If we have too many, we can remove the ones furthest from the center 
    # to keep the packing dense in the middle? No, we want uniform.
    # Actually, the simplest robust way for a fixed N is to adjust 's' 
    # such that the number of points is close to N, then randomly sample 
    # or use a greedy selection.
    
    # Let's try to adjust scale to get close to N points
    # Current count:
    if len(points) == 0:
        # Fallback to square grid if grid generation failed
        return _initial_square_grid(n)
    
    # If we have significantly more, we can subsample.
    # If we have significantly less, we can add random points.
    
    # A better strategy: 
    # 1. Generate a dense grid.
    # 2. Use a greedy algorithm to select N points that maximize min distance.
    # This is O(N * M) where M is grid size. M is ~ N. So O(N^2).
    # For N=200, this is 40k ops, very fast.
    
    if len(points) > n:
        # Greedy selection
        selected = [points[0]]
        min_dists = np.ones(len(points)) * np.inf
        min_dists[0] = 0
        
        # We want to pick points that are far from already selected ones.
        # This is the standard "farthest point sampling".
        for _ in range(n - 1):
            # Find point with max min_dist to selected set
            # This is O(M) per step if we update min_dists incrementally, 
            # but cdist is faster.
            # Let's just use cdist for simplicity, it's fast for M~500.
            d = cdist(points, np.array(selected))
            min_d = d.min(axis=1)
            idx = np.argmax(min_d)
            selected.append(points[idx])
            
        points = np.array(selected)
    elif len(points) < n:
        # Pad with random points
        while len(points) < n:
            points = np.vstack([points, np.random.uniform(-0.45, 0.45, size=(1, 2))])
            
    return points[:n]


def _initial_square_grid(n):
    """Generate a square grid of n points in [-0.5, 0.5]^2."""
    m = int(np.ceil(np.sqrt(n)))
    # Adjust to get closer to n
    while m * m < n:
        m += 1
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]


def _polish(c, time_budget=30.0):
    """
    Refine positions using a steepest ascent method on the minimum radius.
    This acts as a local search that pushes apart the closest pairs.
    """
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    start = time.time()
    
    step_size = 0.005
    no_improve_count = 0
    
    while time.time() - start < time_budget:
        # Identify constraints: 
        # 1. Pairs with distance < 2*best_r + epsilon
        # 2. Points with boundary distance < best_r + epsilon
        
        d = cdist(best_c, best_c)
        np.fill_diagonal(d, np.inf)
        
        # Find pairs that are "active" (limiting the radius)
        # We want to increase the min distance.
        # Gradient of min distance is sum of gradients of the smallest distances.
        # For a pair (i,j) with dist d_ij, the gradient wrt c_i is (c_i - c_j)/d_ij
        # wrt c_j is (c_j - c_i)/d_ij
        
        # Boundary: if x_i > 0.5 - r, grad is -1 in x.
        
        # We will compute a "force" on each point.
        forces = np.zeros_like(best_c)
        
        # 1. Pairwise forces
        # We only consider pairs that are close to the minimum
        threshold = best_r * 2.0 * 1.05 # 5% tolerance
        close_pairs = np.argwhere(d < threshold)
        
        # To avoid O(N^2) in python loop, we use vectorized updates if possible,
        # but N=200 is small enough for a loop if optimized, or we can just 
        # iterate over the close pairs.
        
        if len(close_pairs) > 0:
            for i, j in close_pairs:
                if i == j: continue
                diff = best_c[i] - best_c[j]
                dist = d[i, j]
                if dist < 1e-6:
                    # Perturb randomly if coincident
                    diff = np.random.normal(0, 0.001, size=2)
                    dist = np.linalg.norm(diff)
                    
                # Force to push apart
                # We want to increase distance, so i moves away from j
                force = diff / dist
                forces[i] += force
                forces[j] -= force
        
        # 2. Boundary forces
        # If a point is close to the boundary, push it inwards
        # Boundary distance is 0.5 - |c|
        # If 0.5 - c_x < best_r, push in -sign(c_x) direction
        # If 0.5 + c_x < best_r, push in +sign(c_x) direction
        
        # x-axis
        dist_to_right = 0.5 - best_c[:, 0]
        dist_to_left = 0.5 + best_c[:, 0]
        
        # Points too close to right boundary
        mask_right = dist_to_right < best_r * 1.05
        if np.any(mask_right):
            forces[mask_right, 0] -= 1.0 # Push left
            
        mask_left = dist_to_left < best_r * 1.05
        if np.any(mask_left):
            forces[mask_left, 0] += 1.0 # Push right
            
        # y-axis
        dist_to_top = 0.5 - best_c[:, 1]
        dist_to_bottom = 0.5 + best_c[:, 1]
        
        mask_top = dist_to_top < best_r * 1.05
        if np.any(mask_top):
            forces[mask_top, 1] -= 1.0 # Push down
            
        mask_bottom = dist_to_bottom < best_r * 1.05
        if np.any(mask_bottom):
            forces[mask_bottom, 1] += 1.0 # Push up
            
        # Normalize forces to have unit magnitude? 
        # Or just use a fixed step size.
        # Let's normalize the total force vector per point to avoid huge jumps.
        norms = np.linalg.norm(forces, axis=1)
        norms[norms == 0] = 1.0
        forces = forces / norms[:, np.newaxis]
        
        # Apply move
        new_c = best_c + step_size * forces
        new_c = np.clip(new_c, -0.5 + 1e-10, 0.5 - 1e-10)
        
        new_r = _compute_radius(new_c)
        
        if new_r > best_r:
            best_r = new_r
            best_c = new_c
            no_improve_count = 0
            # Increase step size slightly if improving
            step_size *= 1.05
        else:
            no_improve_count += 1
            step_size *= 0.8
            if step_size < 1e-5:
                # Try a random kick to escape local minimum
                if np.random.rand() < 0.1:
                    kick_idx = np.random.choice(N, 5, replace=False)
                    best_c[kick_idx] += np.random.normal(0, 0.01, size=(5, 2))
                    best_c = np.clip(best_c, -0.5 + 1e-10, 0.5 - 1e-10)
                    best_r = _compute_radius(best_c)
                    step_size = 0.005
                else:
                    break
            if no_improve_count > 50:
                break
                
    return best_c, best_r


def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing minimum radius."""
    global START_TIME, TIME_LIMIT
    
    candidates = []
    
    # 1. Square grid
    c_square = _initial_square_grid(N)
    candidates.append(c_square)
    
    # 2. Hex grids with different offsets
    np.random.seed(42)
    for ox in [0.0, 0.01, -0.01]:
        for oy in [0.0, 0.01, -0.01]:
            c_hex = _initial_hex_grid(N, offset_x=ox, offset_y=oy)
            candidates.append(c_hex)
            
    # 3. Slightly perturbed hex
    c_hex_base = _initial_hex_grid(N)
    c_perturbed = c_hex_base + np.random.normal(0, 0.005, size=(N, 2))
    c_perturbed = np.clip(c_perturbed, -0.49, 0.49)
    candidates.append(c_perturbed)
    
    # Evaluate initial candidates
    best_c = None
    best_r = 0
    for c in candidates:
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c
    
    # Refine the best candidate
    elapsed = time.time() - START_TIME
    time_remaining = TIME_LIMIT - elapsed - 10  # 10s safety margin
    
    if time_remaining > 10:
        c_refined, r_refined = _polish(best_c, time_budget=time_remaining)
        if r_refined > best_r:
            best_r = r_refined
            best_c = c_refined
            
    # Ensure valid output
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Final check
    if not np.all(np.isfinite(best_c)):
        return _initial_square_grid(N)
    
    return best_c
# EVOLVE-BLOCK-END
