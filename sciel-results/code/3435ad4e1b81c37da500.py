"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # State: 
    # w, h derived from aspect ratio ar (w/h), w+h=2
    # circles: array of [x, y, r]
    
    def get_wh(ar):
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        return w, h

    def calculate_energy(circles, w, h):
        """
        Returns (penalty, sum_radii).
        We want to minimize penalty and maximize sum_radii.
        Combined score for SA: sum_radii - penalty * penalty_weight
        """
        penalty = 0.0
        sum_r = 0.0
        
        # Boundary check
        for i in range(N):
            x, y, r = circles[i]
            if x < r:
                penalty += (r - x) ** 2
            elif x > w - r:
                penalty += (x - (w - r)) ** 2
            
            if y < r:
                penalty += (r - y) ** 2
            elif y > h - r:
                penalty += (y - (h - r)) ** 2
            
            sum_r += r
        
        # Overlap check
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist_sq = dx * dx + dy * dy
                min_dist = circles[i, 2] + circles[j, 2]
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    overlap = min_dist - dist
                    penalty += overlap ** 2
        
        return penalty, sum_r

    def is_valid(circles, w, h):
        # Strict validity check for final result
        for i in range(N):
            x, y, r = circles[i]
            if x < r - 1e-9 or x > w - r + 1e-9 or y < r - 1e-9 or y > h - r + 1e-9:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist_sq = dx * dx + dy * dy
                min_dist = circles[i, 2] + circles[j, 2]
                if dist_sq < min_dist * min_dist - 1e-9:
                    return False
        return True

    # Initialize with a good heuristic: Long rectangle, staggered grid
    # 21 circles. 3 rows of 7 is a 3x7 grid.
    # Or 1 row of 21? No, too thin.
    # Let's start with ar = 2.33 (w=1.33, h=0.67) approx 2:1
    # Actually, let's just start with a random valid-ish state or a grid.
    
    best_circles = None
    best_ar = 1.0
    best_score = -np.inf
    
    # Initial configuration: 3x7 grid in a 1.4x0.6 box
    init_ar = 1.4 / 0.6
    w0, h0 = get_wh(init_ar)
    rows, cols = 3, 7
    init_circles = np.zeros((N, 3))
    idx = 0
    for i in range(rows):
        for j in range(cols):
            cx = (j + 0.5) * (w0 / cols)
            if i % 2 == 1:
                cx += (w0 / cols) / 2.0
            cx = np.clip(cx, 0.05, w0 - 0.05)
            cy = (i + 0.5) * (h0 / rows)
            r = min(w0 / cols, h0 / rows) * 0.48
            init_circles[idx] = [cx, cy, r]
            idx += 1
            
    current_ar = init_ar
    current_circles = init_circles.copy()
    w, h = get_wh(current_ar)
    curr_pen, curr_sum = calculate_energy(current_circles, w, h)
    curr_score = curr_sum - 100.0 * curr_pen
    
    if is_valid(current_circles, w, h):
        best_circles = current_circles.copy()
        best_ar = current_ar
        best_score = curr_sum

    T = 0.05 # Initial temperature
    T_min = 0.0001
    cooling_rate = 0.99995
    
    step_size_pos = 0.05
    step_size_r = 0.01
    step_size_ar = 0.05
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Choose a move
        move_type = random.random()
        new_circles = current_circles.copy()
        new_ar = current_ar
        
        if move_type < 0.4:
            # Move a circle
            i = random.randint(0, N - 1)
            new_circles[i, 0] += random.gauss(0, step_size_pos)
            new_circles[i, 1] += random.gauss(0, step_size_pos)
        elif move_type < 0.8:
            # Resize a circle
            i = random.randint(0, N - 1)
            new_circles[i, 2] += random.gauss(0, step_size_r)
            # Keep radius positive
            if new_circles[i, 2] < 0.01:
                new_circles[i, 2] = 0.01
        else:
            # Change aspect ratio
            new_ar += random.gauss(0, step_size_ar)
            # Clamp aspect ratio to reasonable bounds [0.5, 3.0]
            new_ar = np.clip(new_ar, 0.5, 3.0)
            
        new_w, new_h = get_wh(new_ar)
        
        # Clip positions to new bounds to avoid huge penalties from being far out
        # But don't change radius yet, just clip x,y to [r, w-r] roughly
        # Actually, better to just calculate energy with penalties.
        
        new_pen, new_sum = calculate_energy(new_circles, new_w, new_h)
        new_score = new_sum - 100.0 * new_pen
        
        delta = new_score - curr_score
        
        # Acceptance probability
        if delta > 0:
            accept = True
        else:
            accept = random.random() < np.exp(delta / T)
            
        if accept:
            current_circles = new_circles
            current_ar = new_ar
            curr_pen = new_pen
            curr_sum = new_sum
            curr_score = new_score
            
            # Update best if valid
            if curr_pen < 1e-6:
                w_curr, h_curr = get_wh(current_ar)
                if is_valid(current_circles, w_curr, h_curr):
                    if curr_sum > best_score:
                        best_score = curr_sum
                        best_circles = current_circles.copy()
                        best_ar = current_ar

            
        T *= cooling_rate
        
        # Periodic restart if stuck? 
        # If we haven't improved best in a long time, maybe jitter?
        # For now, simple cooling.
        
    # Final check
    if best_circles is not None:
        w_final, h_final = get_wh(best_ar)
        if is_valid(best_circles, w_final, h_final):
            return best_circles
    else:
        # Fallback: return the initial grid if it was valid
        if is_valid(init_circles, w0, h0):
            return init_circles
        else:
            # Emergency fallback: tiny circles in a grid
            fallback = np.zeros((N, 3))
            for i in range(N):
                fallback[i] = [0.1 + (i%7)*0.1, 0.1 + (i//7)*0.1, 0.04]
            return fallback

# EVOLVE-BLOCK-END
