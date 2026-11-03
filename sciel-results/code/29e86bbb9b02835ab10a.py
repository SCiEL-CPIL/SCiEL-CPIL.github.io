import numpy as np
import time
import sys

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin: 150s limit, 5s buffer
    
    # 1. Initialization
    # Use a random initialization with a fixed seed for reproducibility, 
    # but ensure points are reasonably spread out.
    rng = np.random.RandomState(42)
    
    # A good initial guess: place points on a grid and add small random jitter
    # to break symmetry.
    # For N=50, a 4x4x4 grid (64 points) is too dense if we take all, 
    # but we can take a subset or use a 3x4x5 grid (60 points) and take 50.
    # Let's try a 4x4x4 grid, take first 50.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Shuffle to ensure we don't just take a corner
    idx = rng.permutation(len(grid))
    x0 = grid[idx[:N]].copy()
    
    # Add small random jitter to avoid exact symmetries
    x0 += rng.randn(N, 3) * 0.01
    
    # Clip to stay inside
    x0 = np.clip(x0, -0.45, 0.45)
    
    best_x = x0.copy()
    best_r = 0.0
    
    def compute_radius(x):
        """
        Compute the maximum radius for given positions.
        Returns (radius, min_dist_pair, min_dist_wall)
        """
        positions = x
        # Minimum distance to boundary
        # Distance from center to wall is 0.5 - |coord|
        boundary_dists = 0.5 - np.abs(positions)
        min_boundary = np.min(boundary_dists)
        
        # Minimum pairwise distance / 2
        # Using broadcasting for clarity, efficient for N=50
        # dist[i, j] = |x_i - x_j|
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        r = min(min_boundary, min_pair)
        return r, min_pair, min_boundary

    def repulsion_step(x, step_size):
        """
        Perform one iteration of repulsion-based local search.
        For each point, move it away from its closest neighbor/wall.
        """
        new_x = x.copy()
        
        for i in range(N):
            # Calculate distances from i to all others
            diff = x - x[i]
            dists = np.sqrt(np.sum(diff**2, axis=1))
            dists[i] = np.inf # Ignore self
            
            # Distances to walls
            # Wall distances: 
            # x: 0.5 - x[i,0], x[i,0] + 0.5
            # y: 0.5 - x[i,1], x[i,1] + 0.5
            # z: 0.5 - x[i,2], x[i,2] + 0.5
            wall_dists = [
                0.5 - x[i, 0],  # -x wall
                0.5 + x[i, 0],  # +x wall
                0.5 - x[i, 1],  # -y wall
                0.5 + x[i, 1],  # +y wall
                0.5 - x[i, 2],  # -z wall
                0.5 + x[i, 2]   # +z wall
            ]
            
            # Find the closest constraint
            min_dist = np.min(dists)
            closest_idx = np.argmin(dists)
            
            # Check if a wall is closer
            min_wall_dist = np.min(wall_dists)
            closest_wall_idx = np.argmin(wall_dists)
            
            move_dir = np.zeros(3)
            
            if min_dist < min_wall_dist:
                # Move away from sphere closest_idx
                # Direction is x[i] - x[closest_idx]
                move_dir = x[i] - x[closest_idx]
                norm = np.linalg.norm(move_dir)
                if norm > 1e-9:
                    move_dir /= norm
            else:
                # Move away from wall
                # Determine which wall
                if closest_wall_idx == 0: move_dir = [-1, 0, 0]
                elif closest_wall_idx == 1: move_dir = [1, 0, 0]
                elif closest_wall_idx == 2: move_dir = [0, -1, 0]
                elif closest_wall_idx == 3: move_dir = [0, 1, 0]
                elif closest_wall_idx == 4: move_dir = [0, 0, -1]
                else: move_dir = [0, 0, 1]
                
            # Apply move
            # Adaptive step: smaller steps as we get closer? 
            # For now, fixed step size passed in.
            new_x[i] += move_dir * step_size
            
            # Ensure we don't go out of bounds
            new_x[i] = np.clip(new_x[i], -0.5, 0.5)
            
        return new_x

    # Main optimization loop
    # We use a schedule of decreasing step sizes
    # Start with larger steps to escape bad local minima, end with small for precision
    
    # Initial radius check
    r, _, _ = compute_radius(best_x)
    best_r = r
    
    # Phase 1: Coarse search
    # Use a few random restarts with different initializations if time permits
    # But first, refine the current one.
    
    steps = 500
    for i in range(steps):
        if time.time() > deadline - 20.0:
            break
            
        # Step size schedule: linear decay from 0.05 to 0.001
        progress = i / steps
        step_size = 0.05 * (1 - progress) + 0.001
        
        new_x = repulsion_step(best_x, step_size)
        
        # Check if improvement
        r_new, _, _ = compute_radius(new_x)
        if r_new > best_r:
            best_r = r_new
            best_x = new_x
        else:
            # If no improvement, we might be stuck. 
            # For now, we just keep the old position. 
            # A more advanced version would do a line search.
            pass

    # Phase 2: Fine polishing with Nelder-Mead or similar
    # We can use scipy.optimize.minimize with Nelder-Mead on the negative radius
    # but the function is non-smooth. 
    # Instead, let's do a few more iterations with very small steps.
    
    if time.time() < deadline - 5.0:
        steps_fine = 200
        for i in range(steps_fine):
            if time.time() > deadline - 2.0:
                break
            step_size = 0.001 * (1 - i/steps_fine) + 0.0001
            new_x = repulsion_step(best_x, step_size)
            r_new, _, _ = compute_radius(new_x)
            if r_new > best_r:
                best_r = r_new
                best_x = new_x

    # Final check and return
    # Ensure finite values and correct shape
    if not np.all(np.isfinite(best_x)):
        return x0.reshape(-1, 3)
        
    return best_x.reshape(-1, 3)

# Run once at import to verify it doesn't crash immediately? 
# No, the evaluator calls construct().

if __name__ == "__main__":
    # Quick test
    import time
    start = time.time()
    result = construct()
    end = time.time()
    print(f"Time: {end-start}s")
    print(f"Shape: {result.shape}")
    # Calculate radius
    positions = result
    boundary_dists = 0.5 - np.abs(positions)
    min_boundary = np.min(boundary_dists)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    r = min(min_boundary, min_pair)
    print(f"Radius: {r}")
    print(f"Target: 0.1359545126417")
