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
    # Adjust side to fit n circles better if possible, but simple grid is robust
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                # Start with small radii to ensure validity
                r = 0.5 / side * 0.8 
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
    while len(circles) < n and attempts < 1000:
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
            # Pick a radius slightly less than max to ensure safety, or random small
            r = random.uniform(0.02, max(0.03, r_max * 0.8))
            circles.append([x, y, r])
            
    # If we didn't place all, pad with very small circles at random valid spots
    while len(circles) < n:
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        circles.append([x, y, 0.01])
        
    return np.array(circles)

def _generate_move(circles, temperature):
    """
    Generate a new candidate state by perturbing the current state.
    Strategy:
    1. Pick a random circle.
    2. Either move it slightly, or change its radius slightly.
    3. Ensure the new state is valid. If not, return None.
    """
    n = len(circles)
    new_circles = circles.copy()
    
    idx = random.randint(0, n - 1)
    
    # Type of move: 0 = move position, 1 = change radius
    move_type = random.choice([0, 0, 1, 1]) # Bias towards position moves
    
    if move_type == 0:
        # Jiggle position
        # Step size decreases with temperature
        step = 0.01 * (1.0 + temperature)
        dx = random.uniform(-step, step)
        dy = random.uniform(-step, step)
        
        new_x = new_circles[idx, 0] + dx
        new_y = new_circles[idx, 1] + dy
        
        # Check bounds
        r = new_circles[idx, 2]
        if new_x < r or new_x > 1 - r or new_y < r or new_y > 1 - r:
            return None
            
        new_circles[idx, 0] = new_x
        new_circles[idx, 1] = new_y
        
    else:
        # Change radius
        # Try to expand or shrink
        # Expansion is more likely early on, or if we have space
        current_r = new_circles[idx, 2]
        
        # Calculate max possible radius at this position
        x, y = new_circles[idx, 0], new_circles[idx, 1]
        r_max = min(x, 1-x, y, 1-y)
        for i in range(n):
            if i == idx:
                continue
            cx, cy, cr = new_circles[i, 0], new_circles[i, 1], new_circles[i, 2]
            dist = np.sqrt((x-cx)**2 + (y-cy)**2)
            r_limit = dist - cr
            if r_limit < r_max:
                r_max = r_limit
        
        # Propose new radius
        # If we want to expand, pick a random value between current and max
        # If we want to shrink, pick a random value between 0.1*current and current
        if random.random() < 0.7: # Bias towards expansion
            if r_max > current_r + 1e-6:
                # Expand by a random amount up to max, or a small step
                # Let's try to jump to a random fraction of the gap
                gap = r_max - current_r
                new_r = current_r + random.uniform(0.1, 1.0) * gap
            else:
                return None # Can't expand
        else:
            # Shrink
            new_r = current_r * random.uniform(0.8, 0.95)
            if new_r < 0.01:
                new_r = 0.01
        
        new_circles[idx, 2] = new_r
        
        # Final validity check for radius change (bounds)
        if new_circles[idx, 2] > min(new_circles[idx, 0], 1-new_circles[idx, 0], new_circles[idx, 1], 1-new_circles[idx, 1]):
            return None

    # Full validity check
    if not _is_valid(new_circles):
        return None
        
    return new_circles

def construct():
    deadline = time.time() + 145 # Safety margin
    
    best_circles = None
    best_score = -1
    
    # Run Simulated Annealing from multiple starts
    starts = [
        _init_grid(N),
        _init_random_valid(N, 42),
        _init_random_valid(N, 123),
        _init_random_valid(N, 999)
    ]
    
    for start_circles in starts:
        if time.time() > deadline - 5:
            break
            
        if not _is_valid(start_circles):
            # Fallback: shrink if invalid
            start_circles[:, 2] *= 0.5
            if not _is_valid(start_circles):
                continue

        current_circles = start_circles.copy()
        current_score = _sum_radii(current_circles)
        
        if current_score > best_score:
            best_score = current_score
            best_circles = current_circles.copy()
            
        # SA Parameters
        T = 1.0 # Initial temperature
        T_min = 1e-6
        cooling_rate = 0.995 # Slow cooling to explore more
        step_size = 0.05 # Base step size for moves
        
        while T > T_min:
            if time.time() > deadline - 5:
                break
                
            # Generate a move
            # We might need to try multiple moves if the first is invalid
            new_circles = None
            for _ in range(5): # Try up to 5 times to find a valid move
                new_circles = _generate_move(current_circles, T)
                if new_circles is not None:
                    break
            
            if new_circles is None:
                # No valid move found, cool down faster
                T *= 0.9
                continue
                
            new_score = _sum_radii(new_circles)
            delta = new_score - current_score
            
            # Acceptance criterion
            if delta > 0:
                # Always accept improvement
                current_circles = new_circles
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_circles = current_circles.copy()
            else:
                # Accept worse move with probability exp(delta / T)
                # Since delta is negative, this is < 1
                # If T is high, we are more likely to accept
                prob = np.exp(delta / T)
                if random.random() < prob:
                    current_circles = new_circles
                    current_score = new_score
                    
            # Cool down
            T *= cooling_rate
            
            # If we are stuck (no improvement for a long time), we can restart T slightly
            # But simple cooling is usually fine for this size.
            
    if best_circles is None:
        best_circles = _init_grid(N)
        
    return best_circles
# EVOLVE-BLOCK-END
