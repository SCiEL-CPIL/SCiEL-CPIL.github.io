"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 21

def _max_possible_radius(x, y, w, h, positions, radii, idx):
    """Calculate the maximum radius for a circle at (x,y) given other circles."""
    min_dist = min(x, w - x, y, h - y)
    for j in range(N):
        if j == idx:
            continue
        dx = x - positions[j, 0]
        dy = y - positions[j, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        available = dist - radii[j]
        if available < min_dist:
            min_dist = available
    return max(0.0, min_dist)

def _is_valid(w, h, positions, radii, tol=1e-8):
    """Check if the configuration is valid."""
    # Boundary check
    for i in range(N):
        x, y = positions[i]
        r = radii[i]
        if x < -tol or x > w + tol or y < -tol or y > h + tol:
            return False
        if r > x + tol or r > w - x + tol or r > y + tol or r > h - y + tol:
            return False
    
    # Overlap check
    for i in range(N):
        for j in range(i+1, N):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

def _greedy_pack(w):
    """
    Place N circles greedily in a rectangle of width w and height 2-w.
    Returns positions and radii.
    """
    h = 2.0 - w
    positions = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # We will place circles one by one.
    # For each new circle, we search for the position that maximizes the minimum distance
    # to existing circles and boundaries.
    
    # To speed up search, we can use a grid or random sampling + local optimization.
    # Given N=21, a simple random sampling with local refinement for each insertion is feasible.
    
    for i in range(N):
        best_x, best_y, best_r = 0, 0, 0
        # Use multiple random starts for finding the best spot for circle i
        # As i increases, the space gets tighter, so we need more precision.
        num_samples = 500 + i * 100
        
        for _ in range(num_samples):
            x = np.random.uniform(0, w)
            y = np.random.uniform(0, h)
            
            # Calculate max radius at this spot
            r = _max_possible_radius(x, y, w, h, positions, radii, i)
            
            if r > best_r:
                best_r = r
                best_x = x
                best_y = y
        
        # Local refinement of the best spot found
        # Use a small grid around (best_x, best_y) or jitter
        current_x, current_y = best_x, best_y
        current_r = best_r
        step = 0.05
        for _ in range(50):
            improved = False
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    nx = current_x + dx
                    ny = current_y + dy
                    if 0 <= nx <= w and 0 <= ny <= h:
                        nr = _max_possible_radius(nx, ny, w, h, positions, radii, i)
                        if nr > current_r + 1e-6:
                            current_r = nr
                            current_x = nx
                            current_y = ny
                            improved = True
            if not improved:
                step *= 0.7
                if step < 1e-4:
                    break
        
        positions[i, 0] = current_x
        positions[i, 1] = current_y
        radii[i] = current_r
        
    return positions, radii

def _refine(w, positions, radii, max_iters=2000):
    """
    Local refinement to increase sum of radii.
    Strategy: Iterate through circles, try to expand radius. If blocked, jitter position.
    """
    h = 2.0 - w
    best_score = np.sum(radii)
    best_positions = positions.copy()
    best_radii = radii.copy()
    
    T = 0.05 # Initial jitter temperature
    alpha = 0.999
    
    for it in range(max_iters):
        if T < 1e-5:
            break
            
        # Pick a random circle to expand
        i = np.random.randint(N)
        
        # Try to expand radius i
        # Calculate max possible radius for i given current positions of others
        max_r = _max_possible_radius(positions[i, 0], positions[i, 1], w, h, positions, radii, i)
        
        if max_r > radii[i] + 1e-6:
            # We can expand
            # Expand by a fraction of the gap to be safe
            new_r = radii[i] + (max_r - radii[i]) * 0.5
            radii[i] = new_r
        else:
            # Cannot expand without moving. Try to move i to a better spot?
            # Or just accept the current state and move on.
            # A better strategy: Jitter i to find a spot with larger clearance.
            new_x = positions[i, 0] + np.random.normal(0, T)
            new_y = positions[i, 1] + np.random.normal(0, T)
            
            # Keep within bounds
            new_x = np.clip(new_x, 0, w)
            new_y = np.clip(new_y, 0, h)
            
            # Check if this move allows for a larger radius or at least maintains validity
            # Actually, moving might reduce max_r if we move into a crowded area.
            # Let's evaluate the "potential" of the new spot.
            # Potential = max_radius_at_new_spot
            new_max_r = _max_possible_radius(new_x, new_y, w, h, positions, radii, i)
            
            # Accept move if it increases the potential radius significantly
            # or if it's a random walk to explore.
            # Since we want to maximize sum of radii, moving to a spot with higher max_r is good.
            if new_max_r > radii[i] + 1e-5:
                 positions[i, 0] = new_x
                 positions[i, 1] = new_y
                 # Don't increase radius immediately, let the next iteration handle it or do it now?
                 # Let's do it now to take advantage
                 radii[i] = new_max_r
            
        # Update best
        current_score = np.sum(radii)
        if current_score > best_score:
            best_score = current_score
            best_positions = positions.copy()
            best_radii = radii.copy()
            
        T *= alpha
        
    return best_positions, best_radii, best_score

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_overall_score = -np.inf
    best_overall_result = None
    
    # Try different aspect ratios
    # Perimeter 4 -> w + h = 2.
    # Square is w=1, h=1.
    # We'll test w in [0.7, 1.3]
    
    np.random.seed(42) # For reproducibility in development, but random is fine for submission
    
    num_restarts = 10
    w_candidates = np.linspace(0.7, 1.3, num_restarts)
    
    for w in w_candidates:
        if time.time() - start_time > time_limit:
            break
            
        # Greedy Pack
        positions, radii = _greedy_pack(w)
        
        # Refine
        positions, radii, score = _refine(w, positions, radii, max_iters=1000)
        
        # Final validity check and cleanup
        h = 2.0 - w
        
        # Shrink radii slightly to ensure strict validity due to floating point
        valid = True
        for _ in range(5):
            for i in range(N):
                # Boundary
                max_r_bound = min(positions[i, 0], w - positions[i, 0], positions[i, 1], h - positions[i, 1])
                if radii[i] > max_r_bound:
                    radii[i] = max(0, max_r_bound - 1e-7)
                
                # Overlap
                for j in range(i+1, N):
                    dist = np.sqrt(np.sum((positions[i] - positions[j])**2))
                    required = radii[i] + radii[j]
                    if dist < required:
                        overlap = required - dist
                        shrink_i = overlap / 2.0 + 1e-7
                        shrink_j = overlap / 2.0 + 1e-7
                        radii[i] = max(0, radii[i] - shrink_i)
                        radii[j] = max(0, radii[j] - shrink_j)
        
        # Check if strictly valid
        if _is_valid(w, h, positions, radii, tol=1e-6):
            final_score = np.sum(radii)
            if final_score > best_overall_score:
                best_overall_score = final_score
                result = np.column_stack([positions, radii])
                best_overall_result = result
                
    if best_overall_result is None:
        # Fallback: Grid of small circles
        # This should rarely happen
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        positions = []
        radii = []
        for i in range(N):
            row = i // side
            col = i % side
            positions.append([(col + 0.5) / side, (row + 0.5) / side])
            radii.append(0.5 / side - 1e-3)
        return np.array([pos + [r] for pos, r in zip(positions, radii)])
    
    return best_overall_result

# EVOLVE-BLOCK-END
