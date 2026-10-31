# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    n = 21
    # Higher resolution grid for better precision
    # 100x100 is a good balance between resolution and search space size
    grid_res = 100 
    rng = random.Random(42)
    
    # Precompute grid coordinates for speed
    # x_coords[i] = x-coordinate of column i
    # y_coords[j] = y-coordinate of row j
    # We will represent positions as (gx, gy) indices into these arrays
    
    def get_coords_from_indices(gx, gy, w, h):
        cell_w = w / grid_res
        cell_h = h / grid_res
        x = (gx + 0.5) * cell_w
        y = (gy + 0.5) * cell_h
        return x, y

    def compute_score_fast(positions_idx, w, h):
        """
        Vectorized score computation.
        positions_idx: array of shape (n, 2) with (gx, gy)
        Returns: sum of radii, array of radii
        """
        cell_w = w / grid_res
        cell_h = h / grid_res
        
        # Get actual coordinates
        x = (positions_idx[:, 0] + 0.5) * cell_w
        y = (positions_idx[:, 1] + 0.5) * cell_h
        
        # Calculate pairwise distances
        # dist[i, j] = distance between circle i and circle j
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate boundary limits for each circle
        # min(x, w-x, y, h-y)
        boundary_limits = np.minimum.reduce([
            x, 
            w - x, 
            y, 
            h - y
        ])
        
        # Calculate minimum distance to any neighbor for each circle
        # We need min(dist[i, j]) for j != i
        # dists[i, i] is 0, so we set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        min_neighbor_dists = np.min(dists, axis=1)
        
        # Radius is limited by both boundary and neighbors
        # r_i = min(boundary_limit_i, min_neighbor_dist_i / 2)
        radii = np.minimum(boundary_limits, min_neighbor_dists * 0.5)
        
        # Ensure radii are non-negative
        radii = np.maximum(radii, 0.0)
        
        return np.sum(radii), radii

    def run_sa(w, h, max_iters=20000, t_start=1.0, t_end=0.01):
        """
        Run Simulated Annealing for a specific aspect ratio (w, h).
        """
        # Initialize with a random valid configuration
        # To avoid overlaps in the initial state, we can just pick random cells
        # and then let SA fix it, but starting with a decent pack is better.
        # Let's try a simple greedy initialization: sort cells by distance to center, pick closest non-conflicting?
        # Actually, random initialization + SA is robust if SA is strong enough.
        
        # Create a list of all possible grid indices
        all_indices = np.array([(i, j) for i in range(grid_res) for j in range(grid_res)])
        
        # Random initialization
        init_indices = rng.sample(range(len(all_indices)), n)
        positions_idx = all_indices[init_indices]
        
        current_score, _ = compute_score_fast(positions_idx, w, h)
        best_score = current_score
        best_positions = positions_idx.copy()
        
        T = t_start
        cooling_rate = (t_end / t_start) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            # Generate a move
            move_type = rng.random()
            new_positions = positions_idx.copy()
            
            if move_type < 0.5:
                # Move one circle to a nearby cell
                i = rng.randint(0, n-1)
                gx, gy = new_positions[i]
                dx = rng.randint(-3, 3)
                dy = rng.randint(-3, 3)
                new_gx = gx + dx
                new_gy = gy + dy
                if 0 <= new_gx < grid_res and 0 <= new_gy < grid_res:
                    # Check if the target cell is occupied by another circle
                    # We need to ensure no two circles are in the same cell
                    # This is a hard constraint in the grid model
                    conflict = False
                    for k in range(n):
                        if k != i and new_positions[k][0] == new_gx and new_positions[k][1] == new_gy:
                            conflict = True
                            break
                    if not conflict:
                        new_positions[i] = [new_gx, new_gy]
                    else:
                        continue
            elif move_type < 0.8:
                # Swap two circles
                i, j = rng.sample(range(n), 2)
                new_positions[i], new_positions[j] = new_positions[j], new_positions[i]
            else:
                # Teleport one circle to a random cell
                i = rng.randint(0, n-1)
                new_gx, new_gy = rng.randint(0, grid_res-1), rng.randint(0, grid_res-1)
                conflict = False
                for k in range(n):
                    if k != i and new_positions[k][0] == new_gx and new_positions[k][1] == new_gy:
                        conflict = True
                        break
                if not conflict:
                    new_positions[i] = [new_gx, new_gy]
                else:
                    continue
            
            # Ensure no duplicates (should be handled by conflict check, but double check)
            if len(set(map(tuple, new_positions))) < n:
                continue

            new_score, _ = compute_score_fast(new_positions, w, h)
            delta = new_score - current_score
            
            if delta > 0 or rng.random() < np.exp(delta / T):
                positions_idx = new_positions
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_positions = positions_idx.copy()
            
            T *= cooling_rate
            if T < t_end:
                T = t_end
                
        return best_score, best_positions

    best_score_global = -float('inf')
    best_solution_global = None
    
    # Try a range of aspect ratios
    # Perimeter 4 => 2(w+h) = 4 => w+h = 2
    # t is deviation from square. w = 1+t, h = 1-t.
    # t ranges from -0.4 to 0.4 to allow elongated rectangles
    aspect_ratios = np.linspace(-0.4, 0.4, 17)
    
    start_time = time.time()
    time_limit = 100.0 # Leave 20s buffer for final processing
    
    for t in aspect_ratios:
        if time.time() - start_time > time_limit:
            break
            
        w = 1.0 + t
        h = 1.0 - t
        if w <= 0.2 or h <= 0.2: continue
        
        # Run SA for this aspect ratio
        # Reduce max_iters if time is tight
        remaining_time = time_limit - (time.time() - start_time)
        max_iters = int(remaining_time * 500) # Heuristic: ~500 iters/sec
        if max_iters < 1000:
            max_iters = 1000
            
        score, positions = run_sa(w, h, max_iters=max_iters)
        
        if score > best_score_global:
            best_score_global = score
            best_solution_global = (positions, w, h)
            
    if best_solution_global is None:
        # Fallback
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= n: break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.005
                cells.append([x, y, r])
        return np.array(cells[:n])
    
    best_positions, w, h = best_solution_global
    
    # Convert grid indices to coordinates and calculate final radii
    _, radii = compute_score_fast(best_positions, w, h)
    
    result = []
    for i in range(n):
        gx, gy = best_positions[i]
        x, y = get_coords_from_indices(gx, gy, w, h)
        r = radii[i]
        # Clamp to ensure validity
        r = max(0.001, min(r, min(w, h) / 2 - 0.001))
        x = max(r, min(x, w - r))
        y = max(r, min(y, h - r))
        result.append([x, y, r])
    
    result = np.array(result)
    
    # Final validation and repair
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((result[i,0]-result[j,0])**2 + (result[i,1]-result[j,1])**2)
            if dist < result[i,2] + result[j,2] - 1e-7:
                valid = False
        if result[i,0] < result[i,2] or result[i,0] > w - result[i,2]:
            valid = False
        if result[i,1] < result[i,2] or result[i,1] > h - result[i,2]:
            valid = False
            
    if not valid:
        # If invalid, scale down radii slightly to fix
        scale = 0.9
        for i in range(n):
            result[i,2] *= scale
            result[i,0] = max(result[i,2], min(result[i,0], w - result[i,2]))
            result[i,1] = max(result[i,2], min(result[i,1], h - result[i,2]))
            
    return result
# EVOLVE-BLOCK-END
