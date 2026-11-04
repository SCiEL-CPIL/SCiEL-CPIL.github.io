import numpy as np
import random

N = 26

def construct():
    # 1. Initialization: Hexagonal grid
    # We want to pack ~26 circles.
    # A hexagonal grid with spacing d has density ~ 0.9069 * (pi/4) for circles? No, area fraction is pi/sqrt(12) ~ 0.9069.
    # Let's just generate a grid and pick 26 points.
    
    # Generate a hexagonal lattice of points
    points = []
    # Try different spacings to ensure we have at least 26 points in [0,1]^2
    # Spacing s. Number of points approx 1/s^2 * (2/sqrt(3)) 
    # For N=26, s ~ sqrt(2/26) ~ 0.27?
    
    # Let's create a dense hex grid and then select the best 26 or just use a specific pattern.
    # Actually, simpler: Start with a grid of 5x5 = 25 points and add one? No, 5x5 is 25.
    # Let's use a 5x5 grid shifted slightly or a hex pattern.
    
    # Let's generate a hex pattern
    rows = 5
    cols = 6
    spacing = 0.18 # Estimate
    
    # Better initialization:
    # Use a simple grid of 5x5 = 25, and maybe split one or use 5x6=30 and remove 4?
    # Let's just use a random initialization or a hex grid.
    
    # Hex grid generation
    hex_points = []
    s = 0.22 # Spacing
    h = s * np.sqrt(3) / 2
    for i in range(8):
        for j in range(8):
            x = j * s + (i % 2) * s / 2
            y = i * h
            if 0 <= x <= 1 and 0 <= y <= 1:
                hex_points.append((x, y))
                
    # If we have more than N, we need to select N.
    # If we have less, we need to adjust spacing.
    # Let's just generate a list of candidate positions and pick the first N or random N.
    # To ensure they are well distributed, we can just take the first N from a sorted list or random sample.
    
    # Let's try to fill the square more uniformly.
    # Simple approach: Random positions, then relax.
    
    np.random.seed(42)
    xs = np.random.rand(N)
    ys = np.random.rand(N)
    rs = np.full(N, 0.05) # Small initial radius
    
    # Optimization Loop
    # We want to maximize sum(rs)
    # Constraints:
    # 1. Inside square: r <= x, r <= 1-x, r <= y, r <= 1-y
    # 2. No overlap: dist(i,j) >= ri + rj
    
    # We will use a simple simulated annealing on positions and radii.
    # However, changing radii directly is tricky.
    # Alternative: Fix positions, maximize radii? No, positions matter.
    
    # Let's use a force-directed approach.
    # Each circle wants to grow.
    # If it overlaps, it gets pushed.
    
    # Simpler: Hill Climbing on positions only, with radii set to max possible given neighbors?
    # That's hard.
    
    # Let's stick to SA on (x, y, r).
    # State: (x, y, r) for all N circles.
    # Objective: sum(r)
    # Validity:
    #   - All inside square
    #   - No overlaps
    
    # We will define a function to check validity and calculate score.
    
    def get_score(xs, ys, rs):
        # Check boundaries
        for i in range(N):
            if rs[i] > xs[i] + 1e-9 or rs[i] > 1 - xs[i] + 1e-9 or rs[i] > ys[i] + 1e-9 or rs[i] > 1 - ys[i] + 1e-9:
                return -np.inf
            if rs[i] < 0:
                return -np.inf
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - 1e-9:
                    return -np.inf
        
        return np.sum(rs)

    # Initial state
    # Let's try to initialize with a hex grid to be closer to optimal.
    # Generate hex grid
    hex_xs = []
    hex_ys = []
    s = 0.25
    h = s * np.sqrt(3) / 2
    for i in range(6):
        for j in range(6):
            x = j * s + (i % 2) * s / 2
            y = i * h
            if 0.1 <= x <= 0.9 and 0.1 <= y <= 0.9: # Keep away from edges for initial radius
                hex_xs.append(x)
                hex_ys.append(y)
    
    # If we have enough points, use them. If not, randomize.
    # We need exactly N=26 points.
    if len(hex_xs) >= N:
        # Shuffle and take first N
        indices = np.random.permutation(len(hex_xs))[:N]
        xs = np.array(hex_xs)[indices]
        ys = np.array(hex_ys)[indices]
    else:
        # Fallback to random
        xs = np.random.rand(N)
        ys = np.random.rand(N)
        
    # Initial radii: small
    rs = np.full(N, 0.05)
    
    # Simulated Annealing
    T = 0.05
    T_min = 0.0001
    alpha = 0.995
    steps = 5000
    
    current_xs, current_ys, current_rs = xs, ys, rs
    current_score = get_score(current_xs, current_ys, current_rs)
    
    if current_score == -np.inf:
        # If initial is invalid, shrink radii
        for i in range(N):
            current_rs[i] *= 0.5
        current_score = get_score(current_xs, current_ys, current_rs)
        if current_score == -np.inf:
             # Reset to very small
             current_rs = np.full(N, 0.01)
             current_score = get_score(current_xs, current_ys, current_rs)
             
    best_xs, best_ys, best_rs = current_xs.copy(), current_ys.copy(), current_rs.copy()
    best_score = current_score
    
    for step in range(steps):
        # Perturb
        new_xs = current_xs.copy()
        new_ys = current_ys.copy()
        new_rs = current_rs.copy()
        
        # Pick a random circle to move/grow
        i = np.random.randint(N)
        
        # Move
        dx = np.random.normal(0, T * 0.1)
        dy = np.random.normal(0, T * 0.1)
        new_xs[i] += dx
        new_ys[i] += dy
        
        # Try to grow radius
        # Calculate max possible radius for circle i given current positions of others and walls
        # This is a good heuristic: set r_i to max possible, then check if it overlaps with others?
        # No, just perturb radius too.
        dr = np.random.normal(0, T * 0.05)
        new_rs[i] += dr
        
        # Clamp radius
        new_rs[i] = max(0.001, min(new_rs[i], 0.5))
        
        # Clamp position to be inside square (radius must fit)
        # Actually, if position is outside, score is -inf.
        
        new_score = get_score(new_xs, new_ys, new_rs)
        
        # Acceptance
        if new_score > current_score:
            current_xs, current_ys, current_rs = new_xs, new_ys, new_rs
            current_score = new_score
            if current_score > best_score:
                best_xs, best_ys, best_rs = new_xs.copy(), new_ys.copy(), new_rs.copy()
                best_score = current_score
        else:
            # Accept with probability exp((new - old) / T)
            # Since score can be -inf, handle carefully
            if new_score == -np.inf:
                delta = -100
            else:
                delta = new_score - current_score
            prob = np.exp(delta / T)
            if np.random.rand() < prob:
                current_xs, current_ys, current_rs = new_xs, new_ys, new_rs
                current_score = new_score
                if current_score > best_score:
                    best_xs, best_ys, best_rs = new_xs.copy(), new_ys.copy(), new_rs.copy()
                    best_score = current_score
                    
        T *= alpha
        if T < T_min:
            T = T_min
            
    # Final cleanup: Ensure validity
    # Sometimes SA might leave it slightly invalid due to float errors
    # We can try to shrink radii slightly if invalid
    final_score = get_score(best_xs, best_ys, best_rs)
    if final_score == -np.inf:
        # Fallback to a valid configuration
        # Shrink all radii until valid
        for i in range(N):
            best_rs[i] *= 0.9
        # Re-check
        while get_score(best_xs, best_ys, best_rs) == -np.inf:
             for i in range(N):
                 best_rs[i] *= 0.9
             if np.all(best_rs < 1e-5):
                 break

    result = np.zeros((N, 3))
    result[:, 0] = best_xs
    result[:, 1] = best_ys
    result[:, 2] = best_rs
    
    return result
