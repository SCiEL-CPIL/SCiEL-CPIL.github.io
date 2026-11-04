"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def _compute_value(x, W, H):
    """
    x: array of shape (3N,) = [x0, y0, r0, x1, y1, r1, ...]
    W, H: rectangle dimensions (width, height) with 2*(W+H) = 4 => W+H = 2
    Returns the sum of radii (the objective to maximize).
    Constraints are checked externally or via penalty.
    """
    n = N
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Objective: sum of radii
    return np.sum(rs)

def _constraints(x, W, H):
    """
    Return list of constraint functions for scipy.optimize.minimize (SLSQP).
    We maximize sum(rs), so we minimize -sum(rs).
    Constraints:
    1. Circles inside rectangle: 
       xi - ri >= 0, xi + ri <= W, yi - ri >= 0, yi + ri <= H
    2. Non-overlapping: 
       (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2
    """
    n = N
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    cons = []
    
    # Boundary constraints (inequality: g(x) >= 0)
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})  # xi - ri >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: W - x[3*i] - x[3*i+2]})  # W - xi - ri >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})  # yi - ri >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: H - x[3*i+1] - x[3*i+2]})  # H - yi - ri >= 0
    
    # Non-overlapping constraints
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2 - (x[3*i+2] + x[3*j+2])**2
            })
    
    return cons

def _objective_neg(x, W, H):
    """Minimize negative sum of radii."""
    rs = x[2::3]
    return -np.sum(rs)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Try several aspect ratios W/H with W + H = 2
    # W = width, H = height, 2*(W+H) = 4 => W + H = 2
    best_value = 0.0
    best_x = None
    best_W = None
    best_H = None
    
    # Aspect ratios to try
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.67, 1.3, 0.77, 1.1, 0.9, 1.4, 0.7, 1.6, 0.625]
    
    for ratio in aspect_ratios:
        if time.time() > deadline - 5.0:
            break
        
        # W + H = 2, W/H = ratio => W = 2*ratio/(1+ratio), H = 2/(1+ratio)
        W = 2.0 * ratio / (1.0 + ratio)
        H = 2.0 / (1.0 + ratio)
        
        # Initial guess: grid layout
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) * W / side
                cy = (j + 0.5) * H / side
                r = min(W / side, H / side) * 0.45
                cells.append([cx, cy, r])
        
        if len(cells) < N:
            # Pad with small circles
            while len(cells) < N:
                cells.append([W * 0.5, H * 0.5, 0.01])
        
        x0 = np.array(cells).flatten()
        
        # Bounds: 0 <= xi <= W, 0 <= yi <= H, 0 <= ri <= min(W,H)/2
        r_max = min(W, H) / 2.0
        bounds = []
        for i in range(N):
            bounds.append((0, W))
            bounds.append((0, H))
            bounds.append((0, r_max))
        
        # Try SLSQP optimization
        try:
            cons = _constraints(x0, W, H)
            result = minimize(
                _objective_neg, x0, method='SLSQP',
                bounds=bounds, constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if result.success or (hasattr(result, 'fun') and result.fun < 0):
                val = -result.fun
                if val > best_value:
                    best_value = val
                    best_x = result.x.copy()
                    best_W = W
                    best_H = H
                    
        except Exception:
            pass
    
    # If no good result, fall back to simple grid
    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        W = 1.0
        H = 1.0
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) * W / side
                cy = (j + 0.5) * H / side
                r = min(W / side, H / side) * 0.45
                cells.append([cx, cy, r])
        while len(cells) < N:
            cells.append([0.5, 0.5, 0.01])
        best_x = np.array(cells).flatten()
        best_W = W
        best_H = H
        best_value = np.sum(best_x[2::3])
    
    # Extract circles and format output
    xs = best_x[0::3]
    ys = best_x[1::3]
    rs = best_x[2::3]
    
    # Clip radii to be non-negative and small
    rs = np.maximum(rs, 0.0)
    
    # Ensure circles are within bounds
    for i in range(N):
        rs[i] = min(rs[i], xs[i], best_W - xs[i], ys[i], best_H - ys[i])
        xs[i] = np.clip(xs[i], rs[i], best_W - rs[i])
        ys[i] = np.clip(ys[i], rs[i], best_H - rs[i])
    
    result = np.column_stack([xs, ys, rs])
    return result
# EVOLVE-BLOCK-END
