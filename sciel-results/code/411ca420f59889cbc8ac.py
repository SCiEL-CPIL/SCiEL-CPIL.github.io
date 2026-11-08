"""
Circle packing in perimeter-4 rectangle.
Combines grid seeding with vectorized Simulated Annealing for robust optimization.
"""
import numpy as np
import time

N = 21
PERIMETER = 4.0
TIME_LIMIT = 118.0  # Safety margin before 120s limit

def _is_valid(circles, tol=1e-7):
    """Check if the configuration is valid."""
    if len(circles) == 0:
        return False
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    if np.any(r <= 0):
        return False
    
    x_min = np.min(x - r)
    x_max = np.max(x + r)
    y_min = np.min(y - r)
    y_max = np.max(y + r)
    
    W = x_max - x_min
    H = y_max - y_min
    
    if 2 * (W + H) > PERIMETER + tol:
        return False
    
    n = len(circles)
    # Vectorized distance check
    # Create NxN distance matrices
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    sum_r = r[:, np.newaxis] + r[np.newaxis, :]
    
    # We need dist[i,j] >= r[i] + r[j] for i != j
    # dist[i,i] is 0, sum_r[i,i] is 2r[i], so this will fail, so we mask diagonal
    mask = ~np.eye(n, dtype=bool)
    violations = (dist < (sum_r - tol)) & mask
    
    if np.any(violations):
        return False
    return True

def _compute_value(circles):
    """Compute the sum of radii for a valid configuration, else 0."""
    if _is_valid(circles):
        return np.sum(circles[:, 2])
    return 0.0

def _seed_grid():
    """Create a simple grid-based seed configuration."""
    # Try a few grid configurations to find a valid starting point
    # 5x5 grid is 25 spots, we take 21.
    # Perimeter 4 -> W+H=2.
    # If W=1, H=1, area=1.
    # 5x5 grid in 1x1 square: cell size 0.2. Radius 0.1.
    # Sum radii = 21 * 0.1 = 2.1.
    # Can we do better?
    # If we use a 3x7 grid? W=1, H=1. Cell size 1/3 x 1/7? No, aspect ratio matters.
    # Let's try to fit 21 circles in 1x1 square.
    # sqrt(21) ~ 4.58. So 5x5 grid is reasonable.
    
    best_circles = None
    best_val = -1
    
    # Try 5x5 grid
    cols, rows = 5, 5
    W, H = 1.0, 1.0
    s = min(W / cols, H / rows)
    # Center the grid
    x_start = (W - cols * s) / 2
    y_start = (H - rows * s) / 2
    
    circles = []
    count = 0
    for i in range(cols):
        for j in range(rows):
            if count >= N: break
            cx = x_start + (i + 0.5) * s
            cy = y_start + (j + 0.5) * s
            r = 0.5 * s
            circles.append([cx, cy, r])
            count += 1
    circles = np.array(circles)
    
    if _is_valid(circles):
        val = _compute_value(circles)
        if val > best_val:
            best_val = val
            best_circles = circles.copy()
            
    # Try 4x6 grid (24 spots)
    cols, rows = 4, 6
    # Optimize aspect ratio for 4x6?
    # If we keep W+H=2, maybe W=1.2, H=0.8?
    # Let's just try a square 1x1 for simplicity first, SA will adjust.
    
    if best_circles is None:
        # Fallback: very small circles in a line
        circles = np.array([[i * 0.05, 0.5, 0.01] for i in range(N)])
        best_circles = circles
        
    return best_circles

def _optimize(circles, time_budget):
    """Optimize circle positions and radii using Simulated Annealing."""
    t_start = time.time()
    best_circles = circles.copy()
    best_val = _compute_value(best_circles)
    
    # SA Parameters
    T = 0.1  # Initial temperature
    T_min = 1e-5
    alpha = 0.999  # Cooling rate
    max_iters = 100000
    
    current_circles = circles.copy()
    current_val = best_val
    current_penalty = 0.0
    
    # We need a fitness function that penalizes invalid states to guide the SA
    # Fitness = -Sum(r) + Penalty(Overlap) + Penalty(Perimeter)
    
    def get_fitness(c):
        """Returns (value, penalty). Value is sum of radii, penalty is constraint violation."""
        x = c[:, 0]
        y = c[:, 1]
        r = c[:, 2]
        
        val = np.sum(r)
        
        # Perimeter Penalty
        x_min = np.min(x - r)
        x_max = np.max(x + r)
        y_min = np.min(y - r)
        y_max = np.max(y - r)
        W = x_max - x_min
        H = y_max - y_min
        perim_violation = max(0, 2 * (W + H) - PERIMETER)
        
        # Overlap Penalty
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        sum_r = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Mask diagonal
        mask = ~np.eye(len(c), dtype=bool)
        overlaps = np.maximum(0, (sum_r - dist) * mask)
        overlap_penalty = np.sum(overlaps**2)
        
        # Radius positivity penalty
        neg_r_penalty = np.sum(np.maximum(0, -r)**2)
        
        total_penalty = 100 * perim_violation**2 + 10 * overlap_penalty + 100 * neg_r_penalty
        
        return val, total_penalty

    current_val, current_penalty = get_fitness(current_circles)
    
    for i in range(max_iters):
        if time.time() - t_start > time_budget:
            break
        
        # Generate neighbor
        new_circles = current_circles.copy()
        
        # Randomly perturb 1-3 circles
        num_perturb = np.random.randint(1, 4)
        indices = np.random.choice(N, num_perturb, replace=False)
        
        # Perturb positions
        step = T * 0.1
        new_circles[indices, 0] += np.random.normal(0, step, num_perturb)
        new_circles[indices, 1] += np.random.normal(0, step, num_perturb)
        
        # Perturb radii
        step_r = T * 0.05
        new_circles[indices, 2] += np.random.normal(0, step_r, num_perturb)
        
        # Ensure radii are positive
        new_circles[:, 2] = np.maximum(new_circles[:, 2], 1e-6)
        
        new_val, new_penalty = get_fitness(new_circles)
        
        # Acceptance probability
        # We want to maximize val and minimize penalty
        # Objective to maximize: val - penalty
        old_obj = current_val - current_penalty
        new_obj = new_val - new_penalty
        
        delta_obj = new_obj - old_obj
        
        if delta_obj > 0:
            # Improvement, always accept
            current_circles = new_circles
            current_val = new_val
            current_penalty = new_penalty
            
            if new_obj > (best_val - best_penalty if False else best_val): # We track best valid val
                 # Check if valid to update best
                 if _is_valid(new_circles):
                     best_val = new_val
                     best_circles = new_circles.copy()
        else:
            # Deterioration, accept with probability exp(delta_obj / T)
            # delta_obj is negative
            if T > T_min:
                prob = np.exp(delta_obj / T)
                if np.random.random() < prob:
                    current_circles = new_circles
                    current_val = new_val
                    current_penalty = new_penalty
        
        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min
            # Restart with higher temp if stuck?
            # Simple restart:
            if (i // 1000) % 2 == 1 and i > 1000:
                 T = 0.05
        
        # Periodically check best valid
        if i % 100 == 0 and _is_valid(current_circles):
             v = _compute_value(current_circles)
             if v > best_val:
                 best_val = v
                 best_circles = current_circles.copy()
                 
    return best_circles

def construct():
    """Construct 21 disjoint circles in a perimeter-4 rectangle maximizing sum of radii."""
    t_start = time.time()
    
    # Start with a good seed
    circles = _seed_grid()
    
    if circles is None or len(circles) < N:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        circles = np.array(cells[:N])
        if not _is_valid(circles):
             circles[:, 2] *= 0.5
             if not _is_valid(circles):
                 circles[:, 2] *= 0.5
    
    # Optimize
    time_remaining = TIME_LIMIT - (time.time() - t_start) - 2.0
    if time_remaining > 1.0:
        circles = _optimize(circles, time_remaining)
    
    # Final validation
    if not _is_valid(circles):
        # If optimization failed to produce a valid result, fallback to seed
        seed_circles = _seed_grid()
        if seed_circles is not None and _is_valid(seed_circles):
            circles = seed_circles
        else:
            # Construct a trivially valid small configuration
            # 21 circles in a line
            # Perimeter 4 -> Width can be up to 2 if Height is 0 (degenerate)
            # Let's use a 1x1 square, place 21 small circles
            # 5x5 grid with small radii
            s = 0.15
            circles = []
            for i in range(5):
                for j in range(5):
                    if len(circles) >= N: break
                    circles.append([(i+0.5)*s, (j+0.5)*s, 0.4*s])
            circles = np.array(circles[:N])
    
    return circles
