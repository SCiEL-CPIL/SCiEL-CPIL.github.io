"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Initial grid-based packing
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.45 / side])
    
    x = np.array(cells, dtype=float)
    
    def objective(params):
        """Maximize sum of radii, subject to constraints."""
        # params: [x1, y1, r1, x2, y2, r2, ...]
        positions = params.reshape(N, 3)
        radii = positions[:, 2]
        return -np.sum(radii)
    
    def constraints(params):
        """All constraints as inequality constraints (>= 0)."""
        positions = params.reshape(N, 3)
        xs = positions[:, 0]
        ys = positions[:, 1]
        rs = positions[:, 2]
        
        cons = []
        
        # Bounding box: perimeter <= 4
        # width + height <= 2
        # x in [0, width], y in [0, height]
        # We'll handle this via variable transformation or as constraints
        
        # For now, use fixed rectangle and optimize radii
        # Rectangle: width W, height H, W + H <= 2
        # Let's use W = H = 1 for simplicity, or optimize W, H too
        
        # Circles inside rectangle [0, W] x [0, H]
        # x_i - r_i >= 0
        # x_i + r_i <= W
        # y_i - r_i >= 0
        # y_i + r_i <= H
        
        # Non-overlapping: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        # Soft version for optimization
        
        W = 1.0
        H = 1.0
        
        # Boundary constraints
        for i in range(N):
            cons.append(xs[i] - rs[i])
            cons.append(W - xs[i] - rs[i])
            cons.append(ys[i] - rs[i])
            cons.append(H - ys[i] - rs[i])
        
        # Non-overlap constraints (soft)
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2
                min_dist = rs[i] + rs[j]
                cons.append(dist_sq - min_dist**2)
        
        return cons
    
    # Use SLSQP optimization
    bounds = []
    for i in range(N):
        bounds.append((0.01, 1.0))  # x
        bounds.append((0.01, 1.0))  # y
        bounds.append((0.01, 0.5))  # r
    
    x0 = x.flatten()
    
    best_val = -np.sum(x0[:, 2])
    best_x = x0.copy()
    
    # Multiple restarts with different initial configurations
    for restart in range(5):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x_curr = x0.copy()
        else:
            # Perturb initial positions
            x_curr = x0.copy()
            x_curr[::3] += np.random.normal(0, 0.05, N)
            x_curr[1::3] += np.random.normal(0, 0.05, N)
            x_curr[2::3] += np.random.normal(0, 0.02, N)
            x_curr = np.clip(x_curr, 0.01, 0.99)
        
        try:
            res = minimize(objective, x_curr, method='SLSQP', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-8})
            
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Convert back to (x, y, r) format
    result = best_x.reshape(N, 3)
    
    # Ensure validity: clip radii and positions
    for i in range(N):
        result[i, 0] = np.clip(result[i, 0], result[i, 2], 1.0 - result[i, 2])
        result[i, 1] = np.clip(result[i, 1], result[i, 2], 1.0 - result[i, 2])
        result[i, 2] = max(0.001, min(result[i, 2], 0.5))
    
    return result
# EVOLVE-BLOCK-END
