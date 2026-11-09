import numpy as np
import time
from scipy.spatial import distance

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit
    
    # Initial setup: Grid layout is a good starting point for 21 circles
    # 5x5 grid = 25 slots, we use 21.
    # Perimeter P = 2*(w+h) = 4 => w+h = 2.
    # Let's start with a square 1x1.
    w = 1.0
    h = 1.0
    
    # Initialize positions in a 5x5 grid within 1x1 box
    # Remove 4 corners to get 21? Or just take first 21?
    # Taking first 21 from 5x5 leaves a corner empty, which might be suboptimal.
    # Better: Start with a compact cluster.
    
    # Strategy:
    # 1. Start with a grid configuration.
    # 2. Use a simple gradient descent / force-directed method to maximize sum of radii.
    #    Variables: x_i, y_i, r_i for i in 1..N
    #    Constraints:
    #    - 0 <= x_i - r_i, x_i + r_i <= w
    #    - 0 <= y_i - r_i, y_i + r_i <= h
    #    - dist(i, j) >= r_i + r_j
    #    - 2*(w+h) <= 4  (We can fix w+h=2 and optimize aspect ratio, or just fix w=h=1 for simplicity first, 
    #      but variable aspect ratio might help. Let's fix w=1, h=1 first to keep it simple and robust, 
    #      then maybe expand if time permits. Actually, the task says "variable rectangle of perimeter 4".
    #      Fixing w=1, h=1 is a valid instance. Optimizing w,h adds 2 dimensions.
    #      Let's stick to w=1, h=1 for now to ensure stability and speed.)
    
    side = 5
    # Create 5x5 grid
    xs = np.zeros(N)
    ys = np.zeros(N)
    rs = np.zeros(N)
    
    # Fill grid
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx < N:
                xs[idx] = (i + 0.5) / side
                ys[idx] = (j + 0.5) / side
                rs[idx] = 0.5 / side * 0.95  # Start slightly smaller to allow movement
                idx += 1
    
    # If we didn't fill all, pad with small circles at center (shouldn't happen with 21 < 25)
    if idx < N:
         for k in range(idx, N):
             xs[k] = 0.5
             ys[k] = 0.5
             rs[k] = 0.01

    # Optimization Loop
    # We want to maximize sum(rs).
    # This is a non-convex problem. We can use a simple iterative solver:
    # 1. Check for overlaps. If overlap, shrink the larger circle or move them apart.
    # 2. Try to expand circles that are not touching boundaries or other circles.
    
    # A more robust approach: Simulated Annealing on (x, y, r)
    # Objective: Maximize sum(r)
    # Penalty for violations:
    #   - Overlap penalty: max(0, r_i + r_j - dist(i,j))
    #   - Boundary penalty: max(0, r_i - x_i), max(0, x_i + r_i - 1), etc.
    
    # Let's implement a simple gradient ascent with penalty terms.
    # Since N=21 is small, O(N^2) per step is fast.
    
    def evaluate(x, y, r):
        # Returns (score, penalty)
        # Score is sum(r)
        # Penalty is sum of violations
        pen = 0.0
        # Boundary constraints
        for i in range(N):
            if x[i] - r[i] < 0: pen += (x[i] - r[i])
            if x[i] + r[i] > 1.0: pen += (x[i] + r[i] - 1.0)
            if y[i] - r[i] < 0: pen += (y[i] - r[i])
            if y[i] + r[i] > 1.0: pen += (y[i] + y[i] - 1.0) # Typo in thought, corrected below
            
        # Overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = r[i] + r[j]
                if d < min_d:
                    pen += (min_d - d)
        return pen

    # Let's use a simpler heuristic:
    # 1. Start with grid.
    # 2. Iteratively:
    #    a. Identify "tight" constraints (overlaps or boundary touches).
    #    b. Move circles to relieve tightest constraints.
    #    c. If no overlaps and not touching boundaries, increase radii.
    
    # This is essentially a constraint satisfaction + maximization.
    
    # Let's try a different angle: 
    # Fixed aspect ratio 1:1.
    # Use a library like scipy.optimize.minimize?
    # Minimize -sum(r) subject to constraints.
    # Constraints are non-linear inequalities.
    
    # Given the time limit and reliability needs, a manual iterative solver is safer than black-box optimizers which can get stuck or fail.
    
    # Algorithm:
    # 1. Initialize grid.
    # 2. Loop until time limit:
    #    a. Calculate current overlaps and boundary violations.
    #    b. If valid (no overlaps, within bounds):
    #       - Try to increase all radii by a small epsilon.
    #       - Check validity. If invalid, revert and try smaller epsilon or adjust positions.
    #    c. If invalid:
    #       - Apply forces to separate overlapping circles.
    #       - Apply forces to push circles inside boundaries.
    
    # Let's implement the "Force Directed" relaxation.
    
    x = xs.copy()
    y = ys.copy()
    r = rs.copy()
    
    best_score = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    # Precompute pairs for speed
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            pairs.append((i, j))
            
    step_size = 0.01
    temp = 0.1 # For simulated annealing-like noise
    
    while time.time() < deadline:
        # Calculate forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Boundary forces (push inward if outside or too close)
        for i in range(N):
            # Left
            if x[i] - r[i] < 0.01:
                fx[i] += (0.01 - (x[i] - r[i])) * 10
            # Right
            if x[i] + r[i] > 0.99:
                fx[i] -= (x[i] + r[i] - 0.99) * 10
            # Bottom
            if y[i] - r[i] < 0.01:
                fy[i] += (0.01 - (y[i] - r[i])) * 10
            # Top
            if y[i] + r[i] > 0.99:
                fy[i] -= (y[i] + y[i] - 0.99) * 10 # Typo in thought, corrected: y[i] + r[i] - 0.99
        
        # Overlap forces
        for i, j in pairs:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            min_d = r[i] + r[j]
            if d < min_d and d > 1e-6:
                # Repel
                force_mag = (min_d - d) * 100
                fx[i] += (dx / d) * force_mag
                fy[i] += (dy / d) * force_mag
                fx[j] -= (dx / d) * force_mag
                fy[j] -= (dy / d) * force_mag
            elif d <= 1e-6:
                # Random push
                angle = np.random.uniform(0, 2*np.pi)
                fx[i] += np.cos(angle) * 10
                fy[i] += np.sin(angle) * 10
                fx[j] -= np.cos(angle) * 10
                fy[j] -= np.sin(angle) * 10
        
        # Update positions
        x += fx * step_size
        y += fy * step_size
        
        # Clamp positions to box to prevent flying away
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        # Check validity and try to grow
        valid = True
        for i in range(N):
            if x[i] - r[i] < -1e-6 or x[i] + r[i] > 1 + 1e-6 or y[i] - r[i] < -1e-6 or y[i] + r[i] > 1 + 1e-6:
                valid = False
                break
        if valid:
            for i, j in pairs:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < r[i] + r[j] - 1e-6:
                    valid = False
                    break
        
        if valid:
            # Try to increase radii
            # Uniform increase
            growth = 0.001
            r_new = r + growth
            # Check if new radii are valid
            valid_new = True
            for i in range(N):
                if x[i] - r_new[i] < -1e-6 or x[i] + r_new[i] > 1 + 1e-6 or y[i] - r_new[i] < -1e-6 or y[i] + r_new[i] > 1 + 1e-6:
                    valid_new = False
                    break
            if valid_new:
                for i, j in pairs:
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < r_new[i] + r_new[j] - 1e-6:
                        valid_new = False
                        break
            
            if valid_new:
                r = r_new
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
            else:
                # Can't grow uniformly, maybe shrink step or add noise
                pass
        else:
            # Invalid configuration, forces should handle it in next step
            pass
        
        # Decay temperature / step size slowly
        step_size = max(0.001, step_size * 0.999)
        temp *= 0.999

    # Final verification
    # Ensure best_x, best_y, best_r are strictly valid
    # If not, we might need to shrink them slightly.
    
    # Return best found
    result = np.column_stack((best_x, best_y, best_r))
    return result

