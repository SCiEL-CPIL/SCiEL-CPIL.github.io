import numpy as np

N = 26

def max_radius(x, y, positions, radii, i):
    """
    Calculate the maximum radius for circle i at (x, y) 
    given other circles at positions with radii.
    """
    # Distance to walls
    r_wall = min(x, 1.0 - x, y, 1.0 - y)
    
    # Distance to other circles
    min_dist = np.inf
    for j in range(N):
        if i == j:
            continue
        dx = x - positions[j, 0]
        dy = y - positions[j, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        # The new circle of radius r at (x,y) must satisfy dist >= r + radii[j]
        # So r <= dist - radii[j]
        available = dist - radii[j]
        if available < min_dist:
            min_dist = available
            
    return max(0.0, min(r_wall, min_dist))

def construct():
    # 1. Greedy Initialization
    positions = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Place first circle in center
    positions[0] = [0.5, 0.5]
    radii[0] = 0.5
    
    for i in range(1, N):
        best_r = -1.0
        best_pos = [0.5, 0.5]
        
        # Sample random positions to find a good spot
        # For greedy, we can just pick the point that maximizes min distance to existing
        # To be efficient, we can sample a grid or random points
        # N is small, let's do a fine grid search for the best position
        # Or just random sampling with many iterations
        rng = np.random.default_rng(42 + i)
        for _ in range(2000):
            px = rng.uniform(0, 1)
            py = rng.uniform(0, 1)
            r = max_radius(px, py, positions, radii, i)
            if r > best_r:
                best_r = r
                best_pos = [px, py]
                
        positions[i] = best_pos
        radii[i] = best_r
        
    # Initial Score
    current_score = np.sum(radii)
    best_score = current_score
    best_positions = positions.copy()
    best_radii = radii.copy()
    
    # 2. Simulated Annealing
    T_start = 0.1
    T_end = 1e-6
    iterations = 50000
    rng = np.random.default_rng(123)
    
    for it in range(iterations):
        # Cool down
        T = T_start * (T_end / T_start) ** (it / iterations)
        
        # Pick a random circle to move
        i = rng.integers(N)
        
        # Save old state
        old_x, old_y = positions[i]
        old_r = radii[i]
        old_score = current_score
        
        # Perturb position
        # Step size decreases with temperature? Or fixed?
        # Let's use a step size proportional to current radius or a fixed small amount
        step = 0.05 * (T / T_start + 0.1) # Larger steps at high T
        new_x = old_x + rng.normal(0, step)
        new_y = old_y + rng.normal(0, step)
        
        # Clip to bounds
        new_x = np.clip(new_x, 0, 1)
        new_y = np.clip(new_y, 0, 1)
        
        # Update position
        positions[i] = [new_x, new_y]
        
        # Recalculate max radius for this circle
        new_r = max_radius(new_x, new_y, positions, radii, i)
        radii[i] = new_r
        
        # Calculate new score
        # Only radius i changed
        delta = new_r - old_r
        new_score = current_score + delta
        
        # Acceptance criterion
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_positions = positions.copy()
                best_radii = radii.copy()
        else:
            # Worse move
            prob = np.exp(-delta / T) if T > 0 else 0
            if rng.random() < prob:
                current_score = new_score
            else:
                # Reject: restore old state
                positions[i] = [old_x, old_y]
                radii[i] = old_r
    
    # Final check and cleanup
    # Ensure no overlaps due to floating point errors in the final best solution
    # We can do a final "tightening" pass
    for i in range(N):
        # Re-calculate max radius for best solution
        r = max_radius(best_positions[i, 0], best_positions[i, 1], best_positions, best_radii, i)
        # If the calculated max radius is smaller than current, shrink it to avoid overlap
        if r < best_radii[i]:
            best_radii[i] = r
        else:
            # If it can be bigger, we might have missed an opportunity, but we trust the SA
            # Actually, let's just use the calculated max to be safe against FP errors
            best_radii[i] = r
            
    # Verify validity strictly
    # Check walls
    for i in range(N):
        x, y = best_positions[i]
        r = best_radii[i]
        if r > min(x, 1-x, y, 1-y) + 1e-9:
            best_radii[i] = min(x, 1-x, y, 1-y)
            
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            dx = best_positions[i, 0] - best_positions[j, 0]
            dy = best_positions[i, 1] - best_positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            sum_r = best_radii[i] + best_radii[j]
            if dist < sum_r + 1e-9:
                # Scale down both or one? 
                # Simplest: reduce the larger one or both proportionally
                # To be safe, reduce both slightly
                factor = (dist - 1e-9) / sum_r
                best_radii[i] *= factor
                best_radii[j] *= factor
                # This might break wall constraints, so re-check walls
                for k in [i, j]:
                    x, y = best_positions[k]
                    r = best_radii[k]
                    max_r_wall = min(x, 1-x, y, 1-y)
                    if r > max_r_wall:
                        best_radii[k] = max_r_wall

    result = np.hstack([best_positions, best_radii.reshape(-1, 1)])
    return result
