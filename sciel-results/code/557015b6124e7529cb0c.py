import numpy as np
import time
import random

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # State:
    # t: aspect ratio parameter, w = 2*t, h = 2*(1-t)
    # circles: array of shape (N, 3) [x, y, r]
    
    def get_dims(t):
        # Clip t to avoid extreme aspect ratios
        t = np.clip(t, 0.15, 0.85)
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        return w, h

    def evaluate(t, circles):
        """
        Returns (score, penalty, is_valid)
        score = sum(r)
        penalty = sum of squared overlaps + sum of squared boundary violations
        """
        w, h = get_dims(t)
        
        # Boundary violations
        # x < r, x > w-r, y < r, y > h-r
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        
        # Vectorized boundary check
        # left
        viol_left = np.maximum(0, r - x)
        # right
        viol_right = np.maximum(0, x - (w - r))
        # bottom
        viol_bottom = np.maximum(0, r - y)
        # top
        viol_top = np.maximum(0, y - (h - r))
        
        boundary_penalty = np.sum(viol_left**2) + np.sum(viol_right**2) + np.sum(viol_bottom**2) + np.sum(viol_top**2)
        
        # Overlaps
        # For each pair (i, j) with i < j
        overlap_penalty = 0.0
        is_valid = True
        
        # Efficient loop for small N
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                sum_r = r[i] + r[j]
                if dist_sq < sum_r**2:
                    # Overlap
                    dist = np.sqrt(dist_sq)
                    overlap = sum_r - dist
                    overlap_penalty += overlap**2
                    is_valid = False
        if boundary_penalty > 1e-9:
            is_valid = False

        score = np.sum(r)
        penalty = boundary_penalty + overlap_penalty
        return score, penalty, is_valid

    def make_valid(t, circles):
        """
        Attempts to fix a slightly invalid configuration by shrinking circles or moving them.
        Returns (t, circles, is_valid)
        """
        w, h = get_dims(t)
        # Simple projection: shrink radii if they stick out
        # This is a heuristic to salvage a state
        for i in range(N):
            x, y, r = circles[i]
            # Max possible radius at (x,y)
            max_r = min(x, w-x, y, h-y)
            if r > max_r:
                circles[i, 2] = max_r * 0.99
        
        # Check overlaps and shrink/move if necessary
        # This is expensive, so we only do it if we need a valid result at the end
        # For now, just return as is, the main loop tracks best valid.
        return t, circles, False

    # Initialize
    # Start with a square
    t = 0.5
    w, h = get_dims(t)
    
    # Grid initialization
    side = int(np.ceil(np.sqrt(N)))
    # We have 21 circles. 5x5 grid has 25 spots. Use first 21.
    # Or 4x6 = 24.
    # Let's use a 5x5 grid but only fill 21.
    # Actually, let's just place them in a grid that fits.
    # For 21, a 5x5 grid is good.
    # Cell size: w/5, h/5
    circles = np.zeros((N, 3))
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            cx = (i + 0.5) * (w / side)
            cy = (j + 0.5) * (h / side)
            # Radius slightly less than half cell size
            r = 0.45 * min(w, h) / side
            circles[count, 0] = cx
            circles[count, 1] = cy
            circles[count, 2] = r
            count += 1
            
    # Ensure initial validity
    # The grid initialization might have overlaps if aspect ratio is extreme, 
    # but with t=0.5, w=h=1, cell=0.2, r=0.09. Distance=0.2 > 0.18. Valid.
    
    score, penalty, is_valid = evaluate(t, circles)
    best_t = t
    best_circles = circles.copy()
    best_score = score if is_valid else -np.inf
    current_score = score
    current_penalty = penalty
    
    # SA Parameters
    T = 0.1  # Initial temperature
    T_min = 1e-6
    cooling = 0.9995
    
    random.seed(42)
    np.random.seed(42)
    
    while time.time() < deadline:
        # Generate move
        move_type = random.random()
        
        new_t = t
        new_circles = circles.copy()
        
        if move_type < 0.1:
            # Change aspect ratio
            new_t = t + random.gauss(0, 0.05)
            new_t = np.clip(new_t, 0.15, 0.85)
        elif move_type < 0.6:
            # Move a circle
            i = random.randint(0, N - 1)
            new_circles[i, 0] += random.gauss(0, 0.05)
            new_circles[i, 1] += random.gauss(0, 0.05)
        else:
            # Change radius of a circle
            i = random.randint(0, N - 1)
            new_circles[i, 2] *= (1 + random.gauss(0, 0.1))
            new_circles[i, 2] = max(0.01, new_circles[i, 2])
            
        new_score, new_penalty, new_is_valid = evaluate(new_t, new_circles)
        
        # Objective: Maximize score, minimize penalty
        # Energy = -score + penalty
        E_old = -current_score + current_penalty
        E_new = -new_score + new_penalty
        dE = E_new - E_old
        
        if dE < 0 or random.random() < np.exp(-dE / T):
            t = new_t
            circles = new_circles
            current_score = new_score
            current_penalty = new_penalty
            
            if new_is_valid:
                if new_score > best_score:
                    best_score = new_score
                    best_t = t
                    best_circles = circles.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Optional: Reheat if stuck? 
        # For simplicity, keep cooling.
        
        # Check time frequently
        if time.time() > deadline - 5:
            # Last 5 seconds: try to polish the best valid solution?
            # Or just break to ensure we return in time.
            pass

    # Return best valid solution
    # Ensure the returned solution is strictly valid
    # If best_circles is not valid (shouldn't happen if we only update best when valid), 
    # we might need to fix it. But we only update best_score if new_is_valid is True.
    # So best_circles should be valid.
    
    # Final check and projection just in case
    # If for some reason best_score is -inf, return the initial grid
    if best_score == -np.inf:
        return circles # Fallback, might be invalid but better than crashing? 
                       # Actually, the initial grid is valid.
                       # Let's re-evaluate initial to be sure.
        s, p, v = evaluate(t, circles)
        if v:
             return circles
        else:
             # Force valid by shrinking
             for i in range(N):
                 circles[i, 2] *= 0.5
             return circles

    return best_circles
