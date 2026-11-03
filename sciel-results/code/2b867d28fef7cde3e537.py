"""Rewritten program for circle packing in unit square; stronger algorithm."""
import numpy as np
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N circles in the unit square maximizing sum of radii.
    Uses SciPy's L-BFGS-B optimizer with a penalty-based objective function.
    """
    
    def objective(vars):
        # vars is a flat array of [x0, y0, r0, x1, y1, r1, ...]
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        # Objective: Minimize -sum(r) + penalties
        score = -np.sum(r)
        
        # Boundary penalties
        # We want r <= x <= 1-r and r <= y <= 1-r
        # Violation 1: x < r  => x - r < 0. Penalty: (max(0, r-x))^2
        # Violation 2: x > 1-r => x + r > 1. Penalty: (max(0, x+r-1))^2
        # Same for y
        
        # Vectorized boundary penalties
        # lower_x = r - x. If > 0, penalty.
        viol_lower_x = np.maximum(0, r - x)
        viol_upper_x = np.maximum(0, x + r - 1)
        viol_lower_y = np.maximum(0, r - y)
        viol_upper_y = np.maximum(0, y + r - 1)
        
        bound_pen = np.sum(viol_lower_x**2) + np.sum(viol_upper_x**2) + \
                    np.sum(viol_lower_y**2) + np.sum(viol_upper_y**2)
        
        # Overlap penalties
        # For each pair i < j, check if dist < ri + rj
        # dist_sq = (xi-xj)^2 + (yi-yj)^2
        # viol = max(0, (ri+rj) - sqrt(dist_sq))
        # Penalty = sum(viol^2)
        
        # To avoid sqrt in gradient (though L-BFGS-B approximates it), 
        # we can work with squared distances but the penalty is on distance.
        # Let's compute pairwise distances.
        n = len(r)
        overlap_pen = 0.0
        # For 26 circles, 26*25/2 = 325 pairs. This is fast enough.
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    viol = min_dist - dist
                    overlap_pen += viol * viol
        
        # We need to balance the terms. 
        # Radii are around 0.1. Sum is around 2.6.
        # Penalties should be significant when violated.
        # Let's use a weight for penalties.
        weight = 10.0
        return score + weight * (bound_pen + overlap_pen)

    def construct_single(start_x, start_y, start_r):
        n = len(start_r)
        x0 = np.zeros(3 * n)
        x0[0::3] = start_x
        x0[1::3] = start_y
        x0[2::3] = start_r
        
        # Bounds for L-BFGS-B
        # x, y must be in [0, 1]
        # r must be > 0 (let's say [0.01, 0.5])
        # Note: L-BFGS-B bounds are per variable.
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.01, 0.5)) # r
            
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        
        # Extract results
        x_opt = res.x[0::3]
        y_opt = res.x[1::3]
        r_opt = res.x[2::3]
        
        # Check validity
        valid = True
        if np.any(x_opt < r_opt - 1e-9) or np.any(x_opt > 1 - r_opt + 1e-9):
            valid = False
        if np.any(y_opt < r_opt - 1e-9) or np.any(y_opt > 1 - r_opt + 1e-9):
            valid = False
        if np.any(r_opt < 1e-9):
            valid = False
            
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = (x_opt[i] - x_opt[j])**2 + (y_opt[i] - y_opt[j])**2
                min_dist = r_opt[i] + r_opt[j]
                if dist_sq < min_dist**2 - 1e-9:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            return np.sum(r_opt), x_opt, y_opt, r_opt
        else:
            # If invalid, try to repair or return a low score
            # For now, return 0 to ignore this restart
            return -np.inf, x_opt, y_opt, r_opt

    # Initial configurations
    best_score = -np.inf
    best_x, best_y, best_r = None, None, None
    
    np.random.seed(42)
    
    # Restart 1: Grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.4 / side])
            
    x0 = np.array([c[0] for c in cells])
    y0 = np.array([c[1] for c in cells])
    r0 = np.array([c[2] for c in cells])
    
    score, x_opt, y_opt, r_opt = construct_single(x0, y0, r0)
    if score > best_score:
        best_score = score
        best_x, best_y, best_r = x_opt, y_opt, r_opt
        
    # Restart 2-10: Random Jittered Grids
    for restart in range(10):
        x_jit = x0 + np.random.uniform(-0.02, 0.02, N)
        y_jit = y0 + np.random.uniform(-0.02, 0.02, N)
        r_jit = r0 * np.random.uniform(0.8, 1.2, N)
        
        # Ensure initial validity roughly
        # Clip to bounds
        x_jit = np.clip(x_jit, r_jit, 1 - r_jit)
        y_jit = np.clip(y_jit, r_jit, 1 - r_jit)
        
        score, x_opt, y_opt, r_opt = construct_single(x_jit, y_jit, r_jit)
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x_opt, y_opt, r_opt
            
    # If all failed (unlikely), return a safe default
    if best_x is None:
         # Fallback to simple grid with small radii
         x_fallback = np.array([(i % side + 0.5) / side for i in range(N)])
         y_fallback = np.array([(i // side + 0.5) / side for i in range(N)])
         r_fallback = np.full(N, 0.1)
         return np.column_stack([x_fallback, y_fallback, r_fallback])
         
    return np.column_stack([best_x, best_y, best_r])
# EVOLVE-BLOCK-END
