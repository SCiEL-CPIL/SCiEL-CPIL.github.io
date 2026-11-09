"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _is_valid(circles, eps=1e-9):
    """Check if circles are valid: inside unit square and non-overlapping."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check bounds
    if np.any(x < -eps) or np.any(x > 1 + eps) or np.any(y < -eps) or np.any(y > 1 + eps):
        return False
    if np.any(r < -eps):
        return False
        
    # Check non-overlapping
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - eps:
                return False
    return True

def _sum_radii(circles):
    return np.sum(circles[:, 2])

def _init_grid(n):
    """
    Initialize circles in a grid pattern.
    This provides a good, valid starting point that is not too dense.
    """
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                # Start with small radii to ensure validity
                r = 0.5 / side * 0.5 
                cells.append([x, y, r])
    return np.array(cells[:n])

def _init_random_valid(n, seed=None):
    """
    Initialize with random non-overlapping small circles.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    circles = []
    attempts = 0
    while len(circles) < n and attempts < 2000:
        attempts += 1
        # Sample position
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        
        # Determine max possible radius at this location given existing circles
        r_max = min(x, 1-x, y, 1-y)
        for cx, cy, cr in circles:
            dist = np.sqrt((x-cx)**2 + (y-cy)**2)
            r_limit = dist - cr - 0.01 # small buffer
            if r_limit < r_max:
                r_max = r_limit
        
        if r_max > 0.01:
            # Pick a radius slightly less than max to ensure safety
            r = random.uniform(0.02, max(0.03, r_max * 0.7))
            circles.append([x, y, r])
            
    # If we didn't place all, pad with very small circles at random valid spots
    while len(circles) < n:
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        circles.append([x, y, 0.01])
        
    return np.array(circles)

def _max_radius_at(circles, idx):
    """
    Calculate the maximum possible radius for circle 'idx' given its position
    and the positions/radii of all other circles and the walls.
    """
    x, y = circles[idx, 0], circles[idx, 1]
    # Distance to walls
    r_max = min(x, 1 - x, y, 1 - y)
    
    n = len(circles)
    for i in range(n):
        if i == idx:
            continue
        cx, cy, cr = circles[i, 0], circles[i, 1], circles[i, 2]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        # The new radius cannot exceed the distance to the other circle's edge
        r_limit = dist - cr
        if r_limit < r_max:
            r_max = r_limit
            
    return max(0.0, r_max)

def _ls_expand(circles, time_limit=None, start_time=None):
    """
    Lubachevsky-Steampole style iterative expansion.
    Repeatedly pick the circle with the smallest "clearance" (max possible radius increase)
    and expand it to that limit.
    This drives the system towards a "jammed" state where no single circle can grow.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 10 # Default 10 seconds per LS run
        
    n = len(circles)
    circles = circles.copy()
    
    # We iterate until no circle can grow anymore (within tolerance) or time runs out
    # To avoid getting stuck in a specific local minimum, we can restart if stuck
    # But standard LS just expands the most constrained one.
    
    # Optimization: Keep track of max radii? No, just recompute.
    # For N=26, recomputing is cheap.
    
    prev_sum = -1
    stall_counter = 0
    
    while True:
        if time.time() - start_time > time_limit:
            break
            
        # Find the circle with the smallest "slack" or just iterate through?
        # Standard LS: Find circle i that maximizes (r_max_i - r_i)? 
        # Or simply: Find circle i where r_max_i > r_i. 
        # A common heuristic is to pick the one with the smallest r_max_i - r_i 
        # to resolve conflicts early, or the one with the largest gain.
        # Let's try: Pick the circle with the smallest clearance (r_max - r).
        # If all clearances are ~0, we are jammed.
        
        min_slack = float('inf')
        idx_to_expand = -1
        
        for i in range(n):
            r_max = _max_radius_at(circles, i)
            slack = r_max - circles[i, 2]
            if slack < min_slack:
                min_slack = slack
                idx_to_expand = i
                
        if min_slack < 1e-7:
            # Jammed
            break
            
        # Expand idx_to_expand to its max radius
        r_max = _max_radius_at(circles, idx_to_expand)
        circles[idx_to_expand, 2] = r_max
        
        current_sum = _sum_radii(circles)
        if abs(current_sum - prev_sum) < 1e-9:
            stall_counter += 1
            if stall_counter > 50:
                break
        else:
            stall_counter = 0
            prev_sum = current_sum

    return circles

def _local_search(circles, time_limit, start_time):
    """
    High-frequency local search to escape local minima.
    Uses random jiggling of positions and radii.
    """
    n = len(circles)
    best_circles = circles.copy()
    best_score = _sum_radii(best_circles)
    
    # We will do many small steps
    # Step size decays
    step = 0.02
    temp = 0.1
    
    while time.time() - start_time < time_limit:
        # Pick a random circle
        idx = random.randint(0, n - 1)
        
        new_circles = circles.copy()
        
        # Perturb position
        dx = random.uniform(-step, step)
        dy = random.uniform(-step, step)
        
        new_x = new_circles[idx, 0] + dx
        new_y = new_circles[idx, 1] + dy
        
        # Check bounds roughly before full check
        r = new_circles[idx, 2]
        if new_x < 0 or new_x > 1 or new_y < 0 or new_y > 1:
            continue
            
        new_circles[idx, 0] = new_x
        new_circles[idx, 1] = new_y
        
        # Try to expand this circle to its max radius after moving
        # This is a "greedy" move: move, then pack tight.
        r_max = _max_radius_at(new_circles, idx)
        new_circles[idx, 2] = r_max
        
        # Also, maybe try expanding a neighbor? No, keep it simple.
        
        # Check validity
        if _is_valid(new_circles):
            new_score = _sum_radii(new_circles)
            delta = new_score - _sum_radii(circles)
            
            # Accept if better or with some probability (Metropolis)
            if delta > 0 or random.random() < np.exp(delta / temp):
                circles = new_circles
                if new_score > best_score:
                    best_score = new_score
                    best_circles = circles.copy()
                    
        # Decay step size and temperature
        step *= 0.999
        temp *= 0.999
        
        if step < 1e-5:
            break

    return best_circles

def construct():
    deadline = time.time() + 145 # Safety margin
    
    best_circles = None
    best_score = -1
    
    # Strategy:
    # 1. Generate multiple diverse initial configurations.
    # 2. For each, run Lubachevsky-Steampole (LS) expansion to reach a jammed state.
    # 3. Run Local Search (LS) to escape local minima and improve packing.
    # 4. Keep the best result.
    
    starts = []
    
    # Grid start
    starts.append(_init_grid(N))
    
    # Random starts
    for seed in [42, 123, 999, 2024, 777]:
        starts.append(_init_random_valid(N, seed))
    
    # Corner-biased start (often good for packing)
    def _init_corner_biased(n, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        circles = []
        attempts = 0
        while len(circles) < n and attempts < 2000:
            attempts += 1
            # Bias towards corners and edges
            if random.random() < 0.3:
                x = random.choice([random.uniform(0.05, 0.2), random.uniform(0.8, 0.95)])
                y = random.choice([random.uniform(0.05, 0.2), random.uniform(0.8, 0.95)])
            else:
                x = random.uniform(0.1, 0.9)
                y = random.uniform(0.1, 0.9)
                
            r_max = min(x, 1-x, y, 1-y)
            for cx, cy, cr in circles:
                dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                r_limit = dist - cr - 0.01
                if r_limit < r_max:
                    r_max = r_limit
                    
            if r_max > 0.01:
                r = random.uniform(0.02, max(0.03, r_max * 0.7))
                circles.append([x, y, r])
                
        while len(circles) < n:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            circles.append([x, y, 0.01])
        return np.array(circles)
        
    starts.append(_init_corner_biased(N, 555))
    starts.append(_init_corner_biased(N, 666))
    
    for start_circles in starts:
        if time.time() > deadline - 10:
            break
            
        if not _is_valid(start_circles):
            # Fallback: shrink if invalid
            start_circles[:, 2] *= 0.5
            if not _is_valid(start_circles):
                continue

        # Phase 1: Lubachevsky-Steampole Expansion
        # Allocate some time to LS expansion
        ls_time = 2.0 # 2 seconds for LS
        ls_start = time.time()
        packed_circles = _ls_expand(start_circles, time_limit=ls_time, start_time=ls_start)
        
        if not _is_valid(packed_circles):
            # LS expansion should maintain validity by construction (expanding to max allowed),
            # but numerical errors might creep in. If invalid, shrink slightly.
            packed_circles[:, 2] *= 0.99
            if not _is_valid(packed_circles):
                continue
                
        current_score = _sum_radii(packed_circles)
        if current_score > best_score:
            best_score = current_score
            best_circles = packed_circles.copy()
            
        # Phase 2: Local Search
        # Use remaining time for this start
        ls_start = time.time()
        remaining_time = deadline - 15 - ls_start
        if remaining_time > 1:
            ls_result = _local_search(packed_circles, remaining_time, ls_start)
            
            if _is_valid(ls_result):
                ls_score = _sum_radii(ls_result)
                if ls_score > best_score:
                    best_score = ls_score
                    best_circles = ls_result.copy()

    if best_circles is None:
        best_circles = _init_grid(N)
        
    return best_circles
# EVOLVE-BLOCK-END
