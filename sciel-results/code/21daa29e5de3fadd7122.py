import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Maximizes the total side length of 10 non-overlapping axis-aligned squares
    inside the unit square [0, 1]^2 using Multi-start Local Optimization (BFGS).
    """
    n = 10
    start_time = time.time()
    # Safety margin: stop 10 seconds before the 600s hard kill
    hard_deadline = start_time + 600.0 - 10.0
    
    # Objective function for scipy: minimize -sum(sides) + penalty
    # x vector: [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ..., cx_9, cy_9, s_9]
    
    def objective(x):
        # Time check inside objective to prevent infinite loops if scipy hangs
        if time.time() > hard_deadline:
            return 1e10
            
        cx = x[0::3]
        cy = x[1::3]
        s = x[2::3]
        
        # Penalty for boundaries
        # We want s_i > 0 and inside [0,1]
        # Boundary violation: max(0, s/2 - cx) + max(0, cx - (1 - s/2)) + same for y
        # Using smooth approximations or just raw penalty. 
        # For BFGS, raw max is non-smooth but often works. Let's use a soft penalty or just standard.
        # Standard: 
        xmin = cx - s/2
        xmax = cx + s/2
        ymin = cy - s/2
        ymax = cy + s/2
        
        bound_pen = np.sum(np.maximum(0, -xmin)) + np.sum(np.maximum(0, xmax - 1))
        bound_pen += np.sum(np.maximum(0, -ymin)) + np.sum(np.maximum(0, ymax - 1))
        
        # Penalty for s <= 0
        s_pen = np.sum(np.maximum(0, -s))
        
        # Penalty for overlaps
        # Overlap if dx < (si+sj)/2 and dy < (si+sj)/2
        # Penalty = sum over pairs of max(0, (si+sj)/2 - dx) * max(0, (si+sj)/2 - dy)
        # This is a convex penalty for each pair (product of linear terms? No, product of max(0, linear) is non-convex but continuous).
        # Actually, to make it smoother for BFGS, we can use a squared penalty or just rely on the structure.
        # Let's use the standard overlap area penalty which is 0 when separated.
        
        # Vectorized overlap calculation
        # i < j
        i_idx, j_idx = np.triu_indices(n, k=1)
        
        dx = np.abs(cx[i_idx] - cx[j_idx])
        dy = np.abs(cy[i_idx] - cy[j_idx])
        
        # Required separation
        sep = (s[i_idx] + s[j_idx]) / 2.0
        
        # Overlap dimensions
        ox = np.maximum(0, sep - dx)
        oy = np.maximum(0, sep - dy)
        
        # Overlap area
        overlap_pen = np.sum(ox * oy)
        
        total_penalty = 100.0 * (bound_pen + s_pen + overlap_pen)
        
        # We want to maximize sum(s), so minimize -sum(s)
        return -np.sum(s) + total_penalty

    def is_valid(x):
        cx = x[0::3]
        cy = x[1::3]
        s = x[2::3]
        
        # Check s > 0
        if np.any(s <= 0):
            return False
            
        # Check boundaries with tol
        tol = 1e-7
        if np.any(cx - s/2 < -tol) or np.any(cx + s/2 > 1 + tol):
            return False
        if np.any(cy - s/2 < -tol) or np.any(cy + s/2 > 1 + tol):
            return False
            
        # Check overlaps
        i_idx, j_idx = np.triu_indices(n, k=1)
        dx = np.abs(cx[i_idx] - cx[j_idx])
        dy = np.abs(cy[i_idx] - cy[j_idx])
        sep = (s[i_idx] + s[j_idx]) / 2.0 - tol
        
        # Violation if dx < sep AND dy < sep
        if np.any((dx < sep) & (dy < sep)):
            return False
            
        return True

    # Initializations
    # We will generate several diverse starting points.
    # 1. Grid-based perturbations
    # 2. Random feasible-ish points
    
    best_x = None
    best_val = -np.inf
    
    # Define a few structured initializers
    def init_grid(perturb=0.01):
        # 3x3 grid + 1 center? Or 4x4 grid?
        # 10 squares. 
        # Let's try a 3x3 grid where one cell is split or shifted.
        # Or just random positions constrained to be non-overlapping initially?
        # Simple random is hard to make feasible.
        # Let's use a loose grid: 4x4 grid, pick 10 cells.
        # Side length 0.25.
        # This is a valid feasible start.
        # We can then try to expand them.
        
        # Generate 4x4 grid indices
        # Pick 10 random distinct cells from 16
        cells = np.random.choice(16, 10, replace=False)
        cols = cells % 4
        rows = cells // 4
        
        cx = (cols + 0.5) * 0.25 + np.random.uniform(-0.01, 0.01, 10)
        cy = (rows + 0.5) * 0.25 + np.random.uniform(-0.01, 0.01, 10)
        s = np.full(10, 0.25)
        
        x = np.zeros(30)
        x[0::3] = cx
        x[1::3] = cy
        x[2::3] = s
        return x

    def init_random():
        # Random positions, small sizes to ensure feasibility, then grow
        cx = np.random.uniform(0.1, 0.9, 10)
        cy = np.random.uniform(0.1, 0.9, 10)
        s = np.full(10, 0.1)
        x = np.zeros(30)
        x[0::3] = cx
        x[1::3] = cy
        x[2::3] = s
        return x

    # Run multiple optimizations
    n_starts = 50 # We have time for many
    
    for i in range(n_starts):
        if time.time() > hard_deadline:
            break
            
        if i % 2 == 0:
            x0 = init_grid()
        else:
            x0 = init_random()
            
        # Bounds for BFGS
        # cx, cy in [0, 1]
        # s in [0.01, 1.0] (max side 1.0 is theoretical max, but practically smaller)
        bounds = []
        for _ in range(n):
            bounds.append((0.0, 1.0)) # cx
            bounds.append((0.0, 1.0)) # cy
            bounds.append((0.01, 1.0)) # s
            
        try:
            # Use BFGS as it handles bounds well and is robust
            # Method 'L-BFGS-B' is also good for bounds
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 2000, 'maxfun': 10000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < 1e5: # If it converged or didn't explode
                if is_valid(res.x):
                    val = -res.fun # Since objective is -sum(s) + pen, and pen=0 for valid, val=sum(s)
                    if val > best_val:
                        best_val = val
                        best_x = res.x.copy()
        except Exception:
            pass
            
    # If no valid solution found from optimization (unlikely), fallback to a known valid one
    if best_x is None:
        # 4x4 grid, 10 squares
        cells = np.arange(10)
        cols = cells % 4
        rows = cells // 4
        cx = (cols + 0.5) * 0.25
        cy = (rows + 0.5) * 0.25
        s = np.full(10, 0.25)
        best_x = np.zeros(30)
        best_x[0::3] = cx
        best_x[1::3] = cy
        best_x[2::3] = s
        best_val = np.sum(s)

    # Extract final result
    cx = best_x[0::3]
    cy = best_x[1::3]
    s = best_x[2::3]
    
    centers = np.column_stack((cx, cy))
    sides = s
    
    return centers, sides

# EVOLVE-BLOCK-END
