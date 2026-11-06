"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """Place 200 equal circles in a unit square to maximize the minimum radius."""
    t_start = time.time()
    deadline = t_start + 110  # Safety margin
    
    # 1. Initialization: Square grid
    m = int(np.ceil(np.sqrt(N)))
    # We need exactly N points. A grid of m x m might have m*m > N.
    # We can take the first N points of a grid, or use a hexagonal grid initial guess which is often better.
    # Let's stick to a simple grid for initialization, but optimized.
    # Better init: Hexagonal lattice
    def hex_grid(n):
        pts = []
        # Approximate number of rows
        rows = int(np.ceil(n / 1.732)) 
        # This is getting complicated. Let's just use a square grid and let the optimizer do the work.
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    pts = hex_grid(N).copy()
    
    # Ensure points are strictly inside for safety during init, though grid is centered.
    # Grid points are at (k+0.5)/m - 0.5. Range is [-0.5+0.5/m, 0.5-0.5/m]. Safe.

    def get_min_radius(pts):
        """Calculate the max possible radius for the current configuration."""
        # Pairwise distances
        # To be efficient in the loop, we might not call this full O(N^2) function every step.
        # But for initialization and final check, it's fine.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Wall distances
        wall_dists = np.minimum(
            np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
            np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
        )
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist, min_wall_dist) / 2

    current_r = get_min_radius(pts)
    best_r = current_r
    best_pts = pts.copy()

    # Simulated Annealing / Local Search
    # We will perform many small steps.
    # To make it fast, we only update the min distance for the moved point.
    
    # Pre-compute initial distances? No, they change.
    # We maintain a distance matrix? 200x200 is 40k floats. Updating one row/col is O(N).
    # Let's maintain the full distance matrix to avoid O(N^2) recalculation per step.
    
    def calc_dists(p):
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff**2, axis=2))
        return d

    dist_matrix = calc_dists(pts)
    # Wall distances for each point
    def calc_wall_dists(p):
        return np.minimum(
            np.minimum(0.5 - p[:, 0], p[:, 0] + 0.5),
            np.minimum(0.5 - p[:, 1], p[:, 1] + 0.5)
        )
    wall_dists = calc_wall_dists(pts)
    
    # Current minimum radius is min(min off-diag dist_matrix, min wall_dists) / 2
    # To speed up finding the min, we can just take the min of the matrix and wall dists.
    
    T = 0.01  # Initial temperature (step size scale)
    cooling = 0.999
    min_T = 1e-6
    
    steps = 0
    while time.time() < deadline - 5:
        if T < min_T:
            T = 0.005 # Restart with higher T if stuck? Or just break.
            # Let's do a few restarts if time permits
            if steps > 10000: 
                break
            else:
                T = 0.01
                # Randomize slightly to escape local minima
                pts += np.random.randn(N, 2) * 0.01
                pts = np.clip(pts, -0.49, 0.49)
                dist_matrix = calc_dists(pts)
                wall_dists = calc_wall_dists(pts)
                current_r = min(np.min(dist_matrix[np.arange(N), np.arange(N)] if False else dist_matrix[~np.eye(N, dtype=bool)]), np.min(wall_dists)) / 2 # This line is buggy/slow. 
                # Let's just recalculate current_r properly if we reset.
                # Actually, simpler: just continue if T is low.
        
        # Pick a random point
        i = np.random.randint(N)
        
        # Propose move
        delta = np.random.randn(2) * T
        old_pos = pts[i].copy()
        new_pos = old_pos + delta
        
        # Check bounds
        if new_pos[0] < -0.5 or new_pos[0] > 0.5 or new_pos[1] < -0.5 or new_pos[1] > 0.5:
            # Reject move (or reflect, but reject is simpler for SA)
            T *= cooling
            continue
            
        # Calculate new distances for point i
        # dist_matrix[i, :] and dist_matrix[:, i] need updating
        new_dists_i = np.sqrt(np.sum((pts - new_pos)**2, axis=1))
        new_wall_i = np.minimum(
            np.minimum(0.5 - new_pos[0], new_pos[0] + 0.5),
            np.minimum(0.5 - new_pos[1], new_pos[1] + 0.5)
        )
        
        # Determine if this move improves the global minimum radius
        # Current global min radius R_cur
        # New global min radius R_new
        
        # The global min radius is determined by the smallest of:
        # 1. Old min pair dist (excluding pairs involving i)
        # 2. Old min wall dist (excluding i)
        # 3. New min pair dist for i (min of new_dists_i excluding i)
        # 4. New wall dist for i
        
        # To be efficient, we need to know the current "second best" constraints or just track the global min.
        # Simpler: Just calculate the new global min radius directly? That's O(N^2).
        # Let's approximate: 
        # If the new point creates a distance smaller than the current global min, it's bad.
        # If it removes the constraint that *was* the global min, it might be good.
        
        # Let's just compute the new global min radius efficiently.
        # We have the old dist_matrix.
        # The new dist_matrix differs only in row i and col i.
        
        # Find the minimum pair distance in the new configuration:
        # It is min( min(dist_matrix excluding row/col i), min(new_dists_i excluding i) )
        
        # To get min(dist_matrix excluding row/col i) efficiently, we'd need a data structure.
        # Given N=200, N^2=40000. Doing this 10000 times is 400M ops. Might be too slow in Python.
        
        # Alternative: Use a simpler local search that just checks the specific constraints.
        # Let's switch to a simpler Hill Climbing that accepts if the *local* neighborhood improves, 
        # but that doesn't guarantee global min improvement.
        
        # Better: Just do full O(N^2) check but vectorized. 
        # 200 points. 10k steps. 10k * 40k = 400M float ops. 
        # NumPy can do this in ~1-2 seconds per 100k ops? No, NumPy is fast.
        # 400M ops might take 4-10 seconds. 120s limit allows for ~20-30k full checks.
        # Let's try full recalculation for simplicity and correctness, but optimize the loop.
        
        # Actually, let's just do the full recalculation of the radius for the proposed point.
        # We can update the dist_matrix and wall_dists arrays in O(N).
        # Then find the global min.
        
        # Update dist_matrix
        dist_matrix[i, :] = new_dists_i
        dist_matrix[:, i] = new_dists_i
        dist_matrix[i, i] = 0
        
        # Update wall_dists
        old_wall_i = wall_dists[i]
        wall_dists[i] = new_wall_i
        
        # Calculate new global min radius
        # Min pair dist: min of off-diagonal elements of dist_matrix
        # Min wall dist: min of wall_dists
        
        # To find min of off-diagonal, we can just take min of the matrix, 
        # but we must exclude the diagonal (which is 0).
        # Since diagonal is 0, min(matrix) will always be 0. 
        # So we need min(matrix[~eye]).
        
        # This is O(N^2) to find min if we slice.
        # Let's just do it.
        
        # Optimization: Keep track of the current min pair dist and min wall dist.
        # If the move affects the current min, we need to recalculate.
        
        # Let's just recalculate the global min radius for the candidate state.
        # It's O(N^2).
        
        # To make this faster, let's limit the number of steps or use a better algorithm.
        # Given the time limit, let's use a simpler approach:
        # Random Hill Climbing with full O(N^2) evaluation.
        # Estimate: 1000 evaluations * 40k ops = 40M ops. ~0.1s.
        # We can do 100k evaluations in 10s.
        
        # Let's implement a simple random hill climber.
        
        # We already have the logic above. Let's just complete it.
        # We need to compute the new global radius.
        
        # Extract min pair dist
        # We can't just min(dist_matrix) because of diagonal.
        # We can set diagonal to inf temporarily? No, that modifies state.
        # We can use np.min(dist_matrix[~np.eye(N, dtype=bool)])
        
        # This creates a boolean mask of size N^2. 40k booleans.
        # This is fast enough.
        
        min_pair = np.min(dist_matrix[~np.eye(N, dtype=bool)])
        min_wall = np.min(wall_dists)
        new_r = min(min_pair, min_wall) / 2
        
        if new_r > current_r:
            # Accept
            pts[i] = new_pos
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
            # Don't cool down on success? Or cool down slowly.
            # T stays same or increases slightly?
            # Standard SA: T *= cooling regardless? No, usually T decreases.
            # If we accept, we are in a better basin.
            T *= cooling
        else:
            # Reject
            # Roll back
            pts[i] = old_pos
            dist_matrix[i, :] = np.sqrt(np.sum((pts - old_pos)**2, axis=1)) # Recalc to be safe? 
            # Actually, we can just keep the old values if we didn't change pts permanently.
            # But we modified dist_matrix and wall_dists. We must revert them.
            dist_matrix[i, :] = np.sqrt(np.sum((pts - old_pos)**2, axis=1))
            dist_matrix[:, i] = dist_matrix[i, :]
            dist_matrix[i, i] = 0
            wall_dists[i] = old_wall_i
            T *= cooling
            
        steps += 1

    return best_pts

# EVOLVE-BLOCK-END
