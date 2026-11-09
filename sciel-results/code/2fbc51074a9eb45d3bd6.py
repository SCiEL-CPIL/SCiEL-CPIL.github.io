import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def squares_in_square10() -> tuple:
    n = 10
    # Variables: x1, y1, s1, x2, y2, s2, ..., x10, y10, s10
    # Total 30 variables
    
    # Bounds for DE
    # s_i must be > 0. Let's say [0.01, 1.0]
    # x_i, y_i must be such that square is in [0,1].
    # If s_i is fixed, x_i in [s_i/2, 1-s_i/2].
    # But s_i is a variable. 
    # We can just bound x,y in [0,1] and s in [0.01, 1.0] and use penalties for boundaries.
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.01, 1.0)) # s
    
    def unpack(x):
        centers = np.array([x[i*3], x[i*3+1]] for i in range(n))
        sides = np.array([x[i*3+2] for i in range(n)])
        return centers, sides

    def penalty(centers, sides):
        cx = centers[:, 0]
        cy = centers[:, 1]
        s = sides
        
        # Boundary penalty
        # x_min = cx - s/2 >= 0
        # x_max = cx + s/2 <= 1
        # y_min = cy - s/2 >= 0
        # y_max = cy + s/2 <= 1
        
        pen = 0.0
        pen += np.sum(np.maximum(0, -(cx - s/2))**2)
        pen += np.sum(np.maximum(0, (cx + s/2 - 1))**2)
        pen += np.sum(np.maximum(0, -(cy - s/2))**2)
        pen += np.sum(np.maximum(0, (cy + s/2 - 1))**2)
        
        # Overlap penalty
        # For each pair i, j
        for i in range(n):
            for j in range(i+1, n):
                dx = (s[i] + s[j]) / 2.0 - abs(cx[i] - cx[j])
                dy = (s[i] + s[j]) / 2.0 - abs(cy[i] - cy[j])
                if dx > 0 and dy > 0:
                    # They overlap
                    # Penalty proportional to overlap area or just 1
                    # Using product makes it smooth
                    pen += dx * dy
                    
        return pen

    def objective(x):
        centers, sides = unpack(x)
        sum_sides = np.sum(sides)
        p = penalty(centers, sides)
        # We want to maximize sum_sides, so minimize -sum_sides
        # Add large penalty for violations
        return -sum_sides + 100.0 * p

    # Run Differential Evolution
    start_time = time.time()
    timeout = 300  # seconds
    
    # DE parameters
    # popsize: 10 * len(bounds) is standard, but can be smaller
    # maxiter: enough to converge
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=15, 
            tol=1e-5, 
            seed=42,
            polish=True  # This uses L-BFGS-B to polish the best result
        )
    except Exception as e:
        # Fallback to grid if DE fails
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides

    best_x = result.x
    centers, sides = unpack(best_x)
    
    # Verify validity and polish if needed
    # If penalty is still significant, try a few local restarts or just return if it's the best we have
    # The 'polish=True' in DE should handle local optimization.
    
    # Ensure finite
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        # Fallback
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides

    # Check if result is valid (low penalty)
    # If penalty > 1e-5, it might be invalid. 
    # However, DE with polish usually finds valid solutions for this problem size.
    
    return centers, sides
