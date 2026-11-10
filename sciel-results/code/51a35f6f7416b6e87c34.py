"""
Circle packing optimizer using local search and heuristic improvements.
Replaces the physics-based force simulation with a direct constraint-resolution
local search algorithm, which is more effective for discrete packing constraints.
"""
N = 200
import time
import numpy as np
from scipy.spatial import cKDTree

# EVOLVE-BLOCK-START

def _radius_of_placement(centers):
    """
    Compute the largest common radius for a given set of centers in [-0.5, 0.5]^2.
    Returns (radius, min_pair_dist, min_wall_dist, bottleneck_info).
    bottleneck_info: tuple describing the current constraint.
    """
    # Distance from each center to the nearest wall
    # x in [-0.5, 0.5], y in [-0.5, 0.5]
    dist_x = np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5)
    dist_y = np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    dist_to_wall = np.minimum(dist_x, dist_y)
    min_wall = np.min(dist_to_wall)
    
    # Find index of point closest to wall
    idx_wall = np.argmin(dist_to_wall)
    
    # Minimum pairwise distance / 2 using KDTree
    if len(centers) <= 1:
        min_pair = np.inf
        pair = (-1, -1)
    else:
        tree = cKDTree(centers)
        dists, idxs = tree.query(centers, k=2)
        min_pair_dist = np.min(dists[:, 1])
        min_pair = min_pair_dist / 2.0
        
        # Find the pair
        idx_min = np.argmin(dists[:, 1])
        i = idxs[idx_min, 0]
        j = idxs[idx_min, 1]
        pair = (i, j)
    
    r = min(min_wall, min_pair)
    
    # Determine bottleneck type for optimization
    # 0: Wall, 1: Pair
    if min_wall < min_pair:
        bottleneck = (0, idx_wall, -1)
    else:
        bottleneck = (1, pair[0], pair[1])
        
    return r, bottleneck

def _hex_lattice_init(jitter=0.0):
    """
    Generate a hexagonal lattice initialization.
    """
    # Approximate optimal arrangement for 200 circles
    # Try to find a grid size that fits ~200
    # Hex lattice density is pi / (2 * 3^0.5 * r^2) ~ 0.9069
    # Area = 1, N=200 -> r ~ 0.0366
    # Spacing dx = 2*r, dy = r * sqrt(3)
    
    # We don't know r yet, so we estimate.
    # Let's try a grid of rows and cols.
    best_centers = None
    best_r = 0
    
    # Try different aspect ratios for the hex grid
    for rows in range(12, 22):
        # cols approx N/rows
        cols = int(np.ceil(N / rows))
        
        # Hex spacing
        # If we fit 'cols' circles in width 1, spacing dx = 1/cols
        # But hex is staggered.
        # Let's just generate a dense hex pattern and clip.
        
        # Standard hex lattice generation
        # dx = 1.0 / (cols - 0.5)  # roughly
        # dy = dx * sqrt(3) / 2
        
        # Let's use a simpler approach: generate on a grid, then offset odd rows.
        # This is effectively a distorted hex lattice if we adjust spacing.
        
        # Estimate radius:
        # Area per circle ~ 1/200 = 0.005
        # Hex cell area = r^2 * sqrt(3)
        # r^2 = 0.005 / 1.732 = 0.00288 -> r = 0.053 (Too high, boundary effects)
        # Known best r ~ 0.0366
        
        # Let's just place them on a grid with spacing based on expected r
        # If r=0.0366, dx = 0.0732. 1/0.0732 = 13.6 cols.
        
        centers = []
        # Use spacing that fits the number of points
        # Let's try to fit exactly N points in a hex-like pattern
        # Rows: rows, Cols: cols
        # If row is even, x = (col + 0.5) * dx
        # If row is odd, x = (col + 1.0) * dx  (shifted by dx/2? No, hex shift is dx/2)
        
        # Let's calculate dx, dy to fill [-0.5, 0.5]
        # Width 1, Height 1.
        # cols * dx <= 1 => dx = 1/cols
        # rows * dy <= 1 => dy = 1/rows
        # Hex constraint: dy = dx * sqrt(3) / 2
        
        # We can't satisfy both exactly for arbitrary rows/cols.
        # We will generate a hex lattice with a specific spacing and crop.
        
        # Let's assume a base spacing s.
        # Number of points ~ (1/s)^2 * (2/sqrt(3))
        # 200 = (1/s)^2 * 1.1547
        # s^2 = 1.1547 / 200 = 0.00577
        # s = 0.076
        
        s = 0.075
        dx = s
        dy = s * np.sqrt(3) / 2
        
        x_min = -0.5
        y_min = -0.5
        
        # Generate points
        y = y_min
        row = 0
        while y <= 0.5:
            x = x_min
            col = 0
            if row % 2 == 1:
                x += dx / 2.0
            while x <= 0.5:
                centers.append([x, y])
                x += dx
                col += 1
            y += dy
            row += 1
        
        centers = np.array(centers)
        
        # If we have too many, take first N? No, random selection or center crop?
        # Better: Scale the lattice to fit N points better?
        # For now, if len > N, we take a random subset or the first N (biased).
        # Let's try to generate exactly N by adjusting s.
        
        # This is getting complicated. Let's use a simpler "Grid with Jitter" and "Hex Grid" 
        # and let the optimizer do the work.
        
        if len(centers) >= N:
            # If we have more, subsample or just take first N? 
            # Taking first N is biased to top-left. 
            # Let's shuffle and take N.
            idx = np.random.choice(len(centers), N, replace=False)
            centers = centers[idx]
            centers = np.clip(centers, -0.5, 0.5)
            
            if jitter > 0:
                centers += np.random.randn(N, 2) * jitter
                centers = np.clip(centers, -0.5, 0.5)
                
            r, _ = _radius_of_placement(centers)
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
        else:
            # If too few, pad with random? Or just use what we have? 
            # We need exactly N.
            # Add random points in empty space?
            # For simplicity, if we don't have enough, this init is bad.
            pass
            
    if best_centers is None:
        # Fallback: Square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers

def _random_init():
    return np.random.uniform(-0.45, 0.45, (N, 2))

def _local_optimize(centers, time_limit=10.0, step_size=0.01, max_no_improve=1000):
    """
    Local search optimization.
    At each step, identify the bottleneck (min wall or min pair).
    Attempt to move the involved points to increase the minimum radius.
    """
    current = centers.copy()
    current_r, bottleneck = _radius_of_placement(current)
    best = current.copy()
    best_r = current_r
    
    start_time = time.time()
    no_improve = 0
    
    # Adaptive step size
    step = step_size
    
    while time.time() - start_time < time_limit:
        if no_improve > max_no_improve:
            break
            
        r, bottleneck = _radius_of_placement(current)
        b_type, idx1, idx2 = bottleneck
        
        improved = False
        
        if b_type == 0:
            # Bottleneck is wall: idx1 is the point closest to wall
            p = current[idx1]
            # Determine which wall
            dist_x = np.minimum(0.5 - p[0], p[0] + 0.5)
            dist_y = np.minimum(0.5 - p[1], p[1] + 0.5)
            
            if dist_x < dist_y:
                # X wall
                if p[0] > 0:
                    # Right wall, move left
                    move = np.array([-step, 0])
                else:
                    # Left wall, move right
                    move = np.array([step, 0])
            else:
                # Y wall
                if p[1] > 0:
                    # Top wall, move down
                    move = np.array([0, -step])
                else:
                    # Bottom wall, move up
                    move = np.array([0, step])
            
            # Try moving this point
            candidate = current.copy()
            candidate[idx1] += move
            candidate[idx1] = np.clip(candidate[idx1], -0.5, 0.5)
            
            new_r, _ = _radius_of_placement(candidate)
            if new_r > current_r:
                current = candidate
                current_r = new_r
                improved = True
            else:
                # Try random perturbation if direct move failed
                # This helps escape local minima where moving directly towards center
                # reduces pair distance
                for _ in range(5):
                    rand_move = np.random.randn(2) * step
                    candidate = current.copy()
                    candidate[idx1] += rand_move
                    candidate[idx1] = np.clip(candidate[idx1], -0.5, 0.5)
                    new_r, _ = _radius_of_placement(candidate)
                    if new_r > current_r:
                        current = candidate
                        current_r = new_r
                        improved = True
                        break
                        
        elif b_type == 1:
            # Bottleneck is pair: idx1, idx2
            p1 = current[idx1]
            p2 = current[idx2]
            diff = p1 - p2
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                dir1 = np.array([1, 0])
                dir2 = np.array([-1, 0])
            else:
                dir1 = diff / dist
                dir2 = -dir1
            
            # Try moving them apart
            candidate = current.copy()
            candidate[idx1] += dir1 * step
            candidate[idx2] += dir2 * step
            candidate[idx1] = np.clip(candidate[idx1], -0.5, 0.5)
            candidate[idx2] = np.clip(candidate[idx2], -0.5, 0.5)
            
            new_r, _ = _radius_of_placement(candidate)
            if new_r > current_r:
                current = candidate
                current_r = new_r
                improved = True
            else:
                # Try random perturbations of both
                for _ in range(5):
                    rand_move1 = np.random.randn(2) * step
                    rand_move2 = np.random.randn(2) * step
                    candidate = current.copy()
                    candidate[idx1] += rand_move1
                    candidate[idx2] += rand_move2
                    candidate[idx1] = np.clip(candidate[idx1], -0.5, 0.5)
                    candidate[idx2] = np.clip(candidate[idx2], -0.5, 0.5)
                    new_r, _ = _radius_of_placement(candidate)
                    if new_r > current_r:
                        current = candidate
                        current_r = new_r
                        improved = True
                        break
            
        if improved:
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
            no_improve = 0
            # Slightly increase step if we've been improving a lot? 
            # Or keep constant. Let's keep constant for stability.
        else:
            no_improve += 1
            # Decrease step size to allow finer adjustments
            step *= 0.95
            if step < 1e-6:
                break
                
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_centers = None
    best_r = 0
    
    # 1. Hexagonal Lattice Init
    hex_centers = _hex_lattice_init(jitter=0.005)
    hex_r, _ = _radius_of_placement(hex_centers)
    if hex_r > best_r:
        best_r = hex_r
        best_centers = hex_centers.copy()
    
    # 2. Optimize Hex Init
    if time.time() < deadline - 30:
        opt_hex = _local_optimize(hex_centers, time_limit=15.0)
        r, _ = _radius_of_placement(opt_hex)
        if r > best_r:
            best_r = r
            best_centers = opt_hex.copy()
    
    # 3. Random Init + Optimize (Multiple restarts)
    # Use remaining time for random restarts
    remaining = deadline - time.time()
    # Estimate time per restart: ~5-10s
    # Let's do as many as fit in remaining time, leaving 5s buffer
    
    # We can also try a "Grid" init
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N]
    grid_r, _ = _radius_of_placement(grid)
    if grid_r > best_r:
        best_r = grid_r
        best_centers = grid.copy()
    
    if time.time() < deadline - 10:
        opt_grid = _local_optimize(grid, time_limit=10.0)
        r, _ = _radius_of_placement(opt_grid)
        if r > best_r:
            best_r = r
            best_centers = opt_grid.copy()
    
    # Random Restarts
    while time.time() < deadline - 5:
        rand_centers = _random_init()
        # Quick pre-optimization to get out of obvious overlaps
        # Or just run local optimize directly
        opt_rand = _local_optimize(rand_centers, time_limit=5.0)
        r, _ = _radius_of_placement(opt_rand)
        if r > best_r:
            best_r = r
            best_centers = opt_rand.copy()
    
    # Final refinement on the best found
    if best_centers is not None and time.time() < deadline - 2:
        final_opt = _local_optimize(best_centers, time_limit=3.0)
        r, _ = _radius_of_placement(final_opt)
        if r > best_r:
            best_r = r
            best_centers = final_opt.copy()
            
    if best_centers is None:
        # Absolute fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers

# EVOLVE-BLOCK-END
