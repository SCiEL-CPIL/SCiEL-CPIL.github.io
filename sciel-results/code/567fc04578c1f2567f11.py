import numpy as np
import random

N = 26

def construct():
    # Simulated Annealing Parameters
    T = 1.0  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.999
    max_iter = 50000
    
    # Initialize positions and radii
    # Start with a grid-like structure to avoid initial heavy overlaps
    # 26 circles in a 5x5 grid (25) + 1, or 6x5 grid.
    # Let's use a hexagonal packing initialization for better start.
    
    def init_hexagonal():
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Estimate max radius for 26 circles in unit square
        # Area approx: 26 * pi * r^2 <= 1 => r <= sqrt(1/(26*pi)) ~ 0.11
        # But packing efficiency is lower, so r ~ 0.08-0.10
        
        # Create a grid
        cols = 5
        rows = 6
        # We need 26, so 5x6 = 30 slots, we take 26.
        # Or 5 cols, 5 rows = 25, plus 1.
        
        # Let's just scatter them in a grid pattern with some randomness
        # to break symmetry.
        
        # 5x5 grid covers 25 circles.
        # Spacing: 1/5 = 0.2. Radius max 0.1.
        
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx >= N: break
                x[idx] = (i + 0.5) / 5.0 + np.random.uniform(-0.02, 0.02)
                y[idx] = (j + 0.5) / 5.0 + np.random.uniform(-0.02, 0.02)
                r[idx] = 0.08 + np.random.uniform(0, 0.02)
                idx += 1
        
        # Add the 26th circle in the middle or a gap
        # Actually, 5x5 is 25. The 26th can be placed centrally.
        # Or we can use a 6x5 grid and remove 4.
        
        # Let's use a more flexible init: random uniform but with min distance check?
        # No, simple random is fine if SA is strong.
        # But a structured init helps.
        
        # Let's try 5x5 grid + 1 center.
        # The center (0.5, 0.5) is occupied by (2,2) in 0-indexed 5x5.
        # So we shift the grid slightly or just use random for the last one.
        
        # Alternative: 6x5 grid.
        # x: 0.1, 0.3, 0.5, 0.7, 0.9 (5 cols)
        # y: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 (9 rows? No)
        
        # Let's stick to the 5x5 grid for 25, and place the 26th at (0.5, 0.5) 
        # and push the central one out? 
        # Simpler: Just use random initialization but with a better range.
        
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.05, 0.10, N)
        
        return x, y, r

    x, y, r = init_hexagonal()
    
    # Vectorized fitness calculation
    def get_score(x, y, r):
        # Boundary penalty
        # x < r or x > 1-r
        # dist to boundary: min(x, 1-x) for x, min(y, 1-y) for y
        # If dist < r, penalty is (r - dist)^2
        
        x_min = np.minimum(x, 1.0 - x)
        y_min = np.minimum(y, 1.0 - y)
        
        # Boundary violation
        b_viol_x = np.maximum(0.0, r - x_min)
        b_viol_y = np.maximum(0.0, r - y_min)
        boundary_penalty = np.sum(b_viol_x**2 + b_viol_y**2)
        
        # Overlap penalty
        # Compute all pairwise distances
        # x: (N,), y: (N,)
        # dx: (N, N), dy: (N, N)
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        
        # min_dist: r_i + r_j
        min_dist = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Only consider i < j to avoid double counting and self-interaction
        # Mask for upper triangle
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        # Overlap amount: max(0, min_dist - dist)
        # dist = sqrt(dist_sq)
        # We only care where min_dist > dist
        # To be safe and vectorized:
        # overlap = max(0, min_dist - sqrt(dist_sq))
        # penalty = sum(overlap^2)
        
        # Optimization: Only compute sqrt where necessary? 
        # For N=26, N^2=676, sqrt is fast.
        dist = np.sqrt(dist_sq)
        overlap = np.maximum(0.0, min_dist - dist)
        
        # Zero out lower triangle and diagonal
        overlap[~mask] = 0.0
        
        overlap_penalty = np.sum(overlap**2)
        
        total_penalty = boundary_penalty + overlap_penalty
        score = np.sum(r) - 100.0 * total_penalty
        return score

    current_score = get_score(x, y, r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_score = current_score
    
    # Simulated Annealing Loop
    # Step size: start large, decay
    step_size = 0.1
    
    for i in range(max_iter):
        if T < T_min:
            break
            
        # Perturb
        # Move a random subset of circles
        # Or move all with small steps?
        # Moving all with small steps is smoother.
        
        # Random direction
        dx_perturb = np.random.normal(0, step_size, N)
        dy_perturb = np.random.normal(0, step_size, N)
        dr_perturb = np.random.normal(0, step_size * 0.5, N)
        
        new_x = x + dx_perturb
        new_y = y + dy_perturb
        new_r = r + dr_perturb
        
        # Clamp radii
        new_r = np.clip(new_r, 0.01, 0.5)
        
        # Calculate new score
        new_score = get_score(new_x, new_y, new_r)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / T):
            x, y, r = new_x, new_y, new_r
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
                
        # Cooling
        T *= cooling_rate
        step_size *= 0.9999
        
    # Final check and return
    # Ensure validity
    # If the best solution found is invalid (penalty > 0), we might need to repair it.
    # However, SA usually finds valid solutions if penalty weight is high.
    
    # Let's verify validity of best solution
    # Re-calculate penalty for best
    # If penalty is not 0, we try a local repair: shrink radii until valid?
    # Or just return the best one, hoping the penalty is negligible.
    # The task says "disjoint interiors". 
    # If there is overlap, it's invalid.
    
    # Let's do a final strict check
    def is_valid(x, y, r):
        for i in range(N):
            if x[i] < r[i] or x[i] > 1 - r[i] or y[i] < r[i] or y[i] > 1 - r[i]:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                min_dist = r[i] + r[j]
                if dist_sq < min_dist * min_dist - 1e-9:
                    return False
        return True

    if is_valid(best_x, best_y, best_r):
        return np.column_stack((best_x, best_y, best_r))
    else:
        # Fallback: Shrink radii to make valid?
        # Or use a simpler grid solution which is always valid.
        # Grid solution:
        # 5x5 grid.
        # x = (i+0.5)/5, y = (j+0.5)/5
        # r = 0.5/5 - epsilon = 0.1 - 0.01 = 0.09
        # Sum = 25 * 0.09 = 2.25.
        # Current PSO score 0.89 * seed? 
        # Seed score is not given, but 0.8907 is the score.
        # If seed is ~2.5, then 0.89*2.5 = 2.22.
        # So a simple grid might be competitive.
        
        # Let's try to repair the SA solution by shrinking radii.
        # This might lose value but ensures validity.
        # Or just return the grid.
        
        # Let's try a slightly better grid: Hexagonal?
        # No, simple grid is safe.
        
        # But wait, the SA *should* find a valid solution. 
        # If it doesn't, the penalty was not high enough or steps were too big.
        # Let's increase penalty weight in the final check? 
        # No, we already ran.
        
        # Let's just return the grid fallback to be safe, but try to scale it up.
        # Max radius for 5x5 grid: 0.1.
        # Can we do 6x5?
        # 6 cols, 5 rows = 30 circles.
        # Spacing 1/6 = 0.166. Radius 0.083.
        # Sum = 30 * 0.083 = 2.49.
        # But we only have 26.
        # So we can use a 6x5 grid and remove 4 circles, or just place 26 in a 5x6 grid.
        
        # 5 cols, 6 rows = 30.
        # x spacing 0.2, y spacing 0.166.
        # Min spacing 0.166. Radius 0.083.
        # Sum = 26 * 0.083 = 2.158.
        
        # 5x5 grid radius 0.1. Sum 2.5.
        # 5x5 is better.
        
        # Let's return the 5x5 grid.
        x_grid = np.zeros(N)
        y_grid = np.zeros(N)
        r_grid = np.zeros(N)
        for i in range(5):
            for j in range(5):
                if i*5 + j < N:
                    x_grid[i*5+j] = (i + 0.5) / 5.0
                    y_grid[i*5+j] = (j + 0.5) / 5.0
                    r_grid[i*5+j] = 0.1 - 1e-6
        # The 26th circle (index 25)
        # Place it in the center? (0.5, 0.5) is occupied by (2,2).
        # We can shift the whole grid?
        # Or just put the 26th circle in the middle and shrink all?
        
        # Actually, the SA solution is likely valid. The fallback is just a safety net.
        # Let's return the SA solution but with a slight shrink if invalid.
        
        # Check if best is valid again.
        # If not, shrink radii by 10% and check again.
        
        r_check = best_r.copy()
        valid = is_valid(best_x, best_y, r_check)
        
        if not valid:
             # Shrink
             r_check = r_check * 0.9
             valid = is_valid(best_x, best_y, r_check)
             if not valid:
                 r_check = r_check * 0.9
                 valid = is_valid(best_x, best_y, r_check)
        
        if valid:
             return np.column_stack((best_x, best_y, r_check))
        
        # Final fallback: Grid
        return np.column_stack((x_grid, y_grid, r_grid))
