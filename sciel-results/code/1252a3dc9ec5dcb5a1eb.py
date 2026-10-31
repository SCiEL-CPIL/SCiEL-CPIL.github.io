"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Pack 21 circles in a rectangle of perimeter 4, maximizing sum of radii.
    
    Strategy:
    - Start with a grid-based initial placement.
    - Use gradient-based optimization to push circles apart while keeping them inside.
    - Use simulated annealing with random perturbations to escape local minima.
    - The rectangle dimensions are fixed aspect ratio but we can optimize the shape too.
    
    Actually, let's think more carefully. Perimeter = 2*(w+h) = 4, so w+h = 2.
    We want to maximize sum of radii subject to:
    1. All circles inside rectangle [0,w] x [0,h]
    2. No two circles overlap: dist(i,j) >= r_i + r_j
    3. w + h = 2
    
    Let's parameterize: w = 1 + t, h = 1 - t for t in [-0.5, 0.5] (to keep positive)
    Actually w > 0 and h > 0, so t in (-1, 1). Let's use w = 1 + t, h = 1 - t.
    
    For each circle i: center (x_i, y_i), radius r_i.
    Constraints:
    - x_i >= r_i, x_i <= w - r_i
    - y_i >= r_i, y_i <= h - r_i
    - dist(i,j) >= r_i + r_j
    
    We want to maximize sum(r_i).
    
    Approach: Use scipy.optimize or a custom annealing.
    
    Let me try a different approach: fix the rectangle aspect ratio to be close to 1x1 (square)
    or optimize it. For 21 circles, a hexagonal packing in a square might be good.
    
    Let me use a direct optimization approach with scipy.
    """
    from scipy.optimize import minimize
    
    # Initial guess: grid arrangement
    # For 21 circles, use a 5x5 grid minus 4 corners, or similar
    # Let's start with a hexagonal-like pattern
    
    # Parameter vector: [x_1, y_1, r_1, x_2, y_2, r_2, ..., x_21, y_21, r_21, t]
    # where t controls aspect ratio: w = 1+t, h = 1-t
    
    n = N
    # Initial positions in a square of side 1 (w=1, h=1, t=0)
    # Use a grid
    cols = 5
    rows = 5
    # Place 21 circles in a 5x5 grid, skipping 4 positions
    positions = []
    # Skip corners and some others to get 21
    # 5x5 = 25, skip 4
    skip = [(0,0), (0,4), (4,0), (4,4)]
    for i in range(cols):
        for j in range(rows):
            if (i, j) not in skip:
                x = (i + 0.5) / cols
                y = (j + 0.5) / rows
                r = 0.5 / cols - 0.01
                positions.append((x, y, r))
    
    x0 = []
    for (x, y, r) in positions:
        x0.extend([x, y, r])
    x0.append(0.0)  # t parameter
    
    def objective(params):
        """Negative sum of radii (we minimize negative sum)"""
        radii_sum = 0.0
        penalty = 0.0
        w = 1 + params[-1]
        h = 1 - params[-1]
        
        if w <= 0.1 or h <= 0.1:
            return 1e10
        
        for i in range(n):
            xi = params[3*i]
            yi = params[3*i+1]
            ri = params[3*i+2]
            radii_sum += ri
            
            # Boundary constraints
            if xi < ri:
                penalty += (ri - xi) ** 2 * 100
            if xi > w - ri:
                penalty += (xi - (w - ri)) ** 2 * 100
            if yi < ri:
                penalty += (ri - yi) ** 2 * 100
            if yi > h - ri:
                penalty += (yi - (h - ri)) ** 2 * 100
        
        # Overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                xi = params[3*i]
                yi = params[3*i+1]
                ri = params[3*i+2]
                xj = params[3*j]
                yj = params[3*j+1]
                rj = params[3*j+2]
                dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                min_dist = ri + rj
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 100
        
        return -radii_sum + penalty
    
    # Use Nelder-Mead or Powell
    result = minimize(objective, x0, method='Powell', 
                      options={'maxiter': 500, 'xtol': 1e-6, 'ftol': 1e-6})
    
    best_params = result.x
    w = 1 + best_params[-1]
    h = 1 - best_params[-1]
    
    circles = []
    for i in range(n):
        x = best_params[3*i]
        y = best_params[3*i+1]
        r = best_params[3*i+2]
        # Clamp to valid range
        r = max(0.01, min(r, min(w, h)/2 - 0.001))
        x = max(r, min(x, w - r))
        y = max(r, min(y, h - r))
        circles.append([x, y, r])
    
    # Verify and adjust
    circles = np.array(circles)
    
    # Final check: ensure no overlaps and within bounds
    # If any issues, fall back to grid
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((circles[i,0]-circles[j,0])**2 + (circles[i,1]-circles[j,1])**2)
            if dist < circles[i,2] + circles[j,2] - 1e-6:
                valid = False
        if circles[i,0] < circles[i,2] or circles[i,0] > w - circles[i,2]:
            valid = False
        if circles[i,1] < circles[i,2] or circles[i,1] > h - circles[i,2]:
            valid = False
    
    if not valid:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return circles
# EVOLVE-BLOCK-END
