"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _soft_min(vals, k=10.0):
    """Smooth approximation of min(vals) using log-sum-exp trick."""
    if len(vals) == 0:
        return 0.0
    m = np.min(vals)
    return m - np.log(np.sum(np.exp(-k * (vals - m)))) / k

def _objective(params, k=5.0):
    """
    Objective to minimize: penalty for overlaps and wall violations,
    plus negative sum of radii (so minimizing increases radii).
    
    params layout:
      x, y, r for each of N circles, then w, h (rectangle dimensions)
      Total: 3*N + 2
    """
    x = params[0:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    w = params[3*N]
    h = params[3*N+1]
    
    # Perimeter constraint: 2*(w+h) <= 4 => w+h <= 2
    # We'll enforce w+h = 2 for max area (since for fixed perimeter, square is optimal for packing? 
    # Actually for circle packing, shape matters. Let's allow variable w,h with w+h <= 2.
    # To keep it simple, let's parameterize with w and h such that w+h=2, and optimize aspect ratio.
    # Let w = 2*h/(1+h) ... no, let's just use w and h directly with penalty if w+h > 2.
    
    pen = 0.0
    
    # Wall constraints: circles must be inside [0,w] x [0,h]
    # x_i - r_i >= 0, x_i + r_i <= w, y_i - r_i >= 0, y_i + r_i <= h
    wall_pen = np.maximum(0, r - x) + np.maximum(0, x + r - w) + \
               np.maximum(0, r - y) + np.maximum(0, y + r - h)
    pen += np.sum(wall_pen) ** 2
    
    # Overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist:
                # Penalty for overlap
                overlap = min_dist - np.sqrt(dist_sq)
                pen += overlap * overlap
    
    # Perimeter penalty: if w+h > 2, penalize
    if w + h > 2.0:
        pen += (w + h - 2.0) ** 2
    
    # Also penalize if w or h is too small
    if w < 0.1:
        pen += (0.1 - w) ** 2
    if h < 0.1:
        pen += (0.1 - h) ** 2
    
    # We want to MAXIMIZE sum of radii, so minimize -sum(r)
    # But we also need to ensure feasibility. Use a barrier approach.
    # Let's use a combined objective: pen - lambda * sum(r)
    # But this is tricky. Instead, let's use a sequential approach:
    # First, find a feasible configuration with small radii, then increase radii.
    
    # For now, let's just return pen + 0.1 * (2.0 - sum(r)) to encourage larger radii
    # But this might not work well. Let's try a different approach.
    
    # Alternative: minimize pen, subject to sum(r) being maximized.
    # Since we can't do constrained optimization easily with this, let's use a penalty method
    # where we add a term that encourages larger radii.
    
    # Let's use: objective = pen + alpha * (target_r - mean_r)^2
    # where target_r is a target average radius.
    
    # Actually, let's just return pen + 0.01 * (1.0 - sum(r)/N) ** 2
    # This encourages sum(r) to be close to 1.0 (which is a reasonable target).
    
    return pen + 0.05 * (1.0 - np.sum(r) / N) ** 2

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    # Initial guess: grid layout
    side = int(np.ceil(np.sqrt(N)))
    w0 = 1.0
    h0 = 1.0
    x0 = np.array([(i + 0.5) / side for i in range(side) for _ in range(side)][:N])
    y0 = np.array([(j + 0.5) / side for j in range(side) for _ in range(side)][:N])
    r0 = np.full(N, 0.5 / side - 0.01)
    
    # Ensure initial points are within [0, w0] x [0, h0]
    x0 = np.clip(x0, 0.01, w0 - 0.01)
    y0 = np.clip(y0, 0.01, h0 - 0.01)
    
    params0 = np.concatenate([x0, y0, r0, [w0, h0]])
    
    best_params = params0.copy()
    best_val = _objective(best_params)
    
    # Try multiple restarts with different initial configurations
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            params_cur = params0.copy()
        else:
            # Perturb the initial guess
            np.random.seed(restart)
            x_cur = np.random.uniform(0.1, 0.9, N)
            y_cur = np.random.uniform(0.1, 0.9, N)
            r_cur = np.full(N, 0.1)
            w_cur = 1.0
            h_cur = 1.0
            params_cur = np.concatenate([x_cur, y_cur, r_cur, [w_cur, h_cur]])
        
        # Optimize
        try:
            res = minimize(_objective, params_cur, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    # Extract best solution
    x = best_params[0:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    w = best_params[3*N]
    h = best_params[3*N+1]
    
    # Enforce constraints: clip radii and positions
    # Ensure circles are inside the rectangle
    r = np.maximum(r, 0.01)
    
    # Scale to ensure perimeter <= 4
    if w + h > 2.0:
        scale = 2.0 / (w + h)
        x = x * scale
        y = y * scale
        r = r * scale
        w = w * scale
        h = h * scale
    
    # Final clipping to ensure validity
    r = np.clip(r, 0.01, min(w, h) / 2)
    x = np.clip(x, r, w - r)
    y = np.clip(y, r, h - r)
    
    # Check for overlaps and resolve by reducing radii if needed
    for it in range(100):
        max_overlap = 0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    max_overlap = max(max_overlap, overlap)
                    # Reduce radii slightly
                    r[i] = max(0.01, r[i] - overlap/2)
                    r[j] = max(0.01, r[j] - overlap/2)
        if max_overlap < 1e-6:
            break
    
    # Re-clip after radius adjustments
    x = np.clip(x, r, w - r)
    y = np.clip(y, r, h - r)
    
    # Final check: if any circle is invalid, use a fallback grid
    valid = True
    for i in range(N):
        if x[i] < r[i] - 1e-6 or x[i] > w - r[i] + 1e-6 or \
           y[i] < r[i] - 1e-6 or y[i] > h - r[i] + 1e-6:
            valid = False
            break
    
    if not valid:
        # Fallback: grid layout
        side = int(np.ceil(np.sqrt(N)))
        w_f = 1.0
        h_f = 1.0
        x_f = np.array([(i + 0.5) / side for i in range(side) for _ in range(side)][:N])
        y_f = np.array([(j + 0.5) / side for j in range(side) for _ in range(side)][:N])
        r_f = np.full(N, 0.5 / side - 0.05)
        return np.column_stack([x_f, y_f, r_f])
    
    return np.column_stack([x, y, r])
# EVOLVE-BLOCK-END
