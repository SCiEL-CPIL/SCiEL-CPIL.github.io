# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Place 10 squares in the unit square to maximize the sum of their side lengths.
    Uses local optimization to refine a starting configuration.
    """
    n = 10
    
    # Initial configuration: 3x3 grid of small squares + 1 large square
    # This is the current baseline, slightly perturbed to allow optimization to move.
    s_init = 0.3333333
    L_init = 1.0 - 3.0 * s_init
    
    centers_init = np.zeros((n, 2), dtype=np.float64)
    sides_init = np.zeros(n, dtype=np.float64)
    
    for i in range(9):
        row = i // 3
        col = i % 3
        cx = (col + 0.5) * s_init
        cy = (row + 0.5) * s_init
        centers_init[i] = [cx, cy]
        sides_init[i] = s_init
        
    cx = 1.0 - L_init / 2.0
    cy = 1.0 - L_init / 2.0
    centers_init[9] = [cx, cy]
    sides_init[9] = L_init

    # Flatten for optimization
    x0 = np.concatenate([centers_init.flatten(), sides_init])
    
    def objective(x):
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        
        # Penalty for negative sides
        pen = 0
        if np.any(sides <= 0):
            pen += 100 * np.sum(np.maximum(0, -sides))
            
        # Penalty for boundary violations
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            # x-boundary
            if cx - s/2 < 0:
                pen += 10 * (0 - (cx - s/2))
            if cx + s/2 > 1:
                pen += 10 * (cx + s/2 - 1)
            # y-boundary
            if cy - s/2 < 0:
                pen += 10 * (0 - (cy - s/2))
            if cy + s/2 > 1:
                pen += 10 * (cy + s/2 - 1)
                
        # Penalty for overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i,0] - centers[j,0])
                dy = abs(centers[i,1] - centers[j,1])
                req_x = (sides[i] + sides[j]) / 2
                req_y = (sides[i] + sides[j]) / 2
                
                # If they overlap in both x and y, penalize the penetration
                if dx < req_x and dy < req_y:
                    pen_x = req_x - dx
                    pen_y = req_y - dy
                    pen += 10 * (pen_x + pen_y)
                    
        # We want to maximize sum(sides), so minimize -sum(sides) + penalty
        return -np.sum(sides) + pen

    # Run optimization
    # Use Nelder-Mead as it is robust for small problems with penalties
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    x_opt = res.x
    centers = x_opt[:2*n].reshape(n, 2)
    sides = x_opt[2*n:]
    
    # Ensure sides are positive and finite
    sides = np.maximum(sides, 1e-9)
    
    # Check if the result is valid. If not, fall back to the initial valid configuration
    def is_valid(c, s):
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(s)):
            return False
        if np.any(s <= 0):
            return False
        tol = 1e-7
        for i in range(n):
            cx, cy = c[i]
            si = s[i]
            if cx - si/2 < -tol or cx + si/2 > 1 + tol:
                return False
            if cy - si/2 < -tol or cy + si/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(c[i,0] - c[j,0])
                dy = abs(c[i,1] - c[j,1])
                t = (s[i] + s[j]) / 2 - tol
                if dx < t and dy < t:
                    return False
        return True

    if is_valid(centers, sides):
        return centers, sides
    else:
        # Fallback to initial valid configuration
        return centers_init, sides_init

# EVOLVE-BLOCK-END
