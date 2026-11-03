import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    time_limit = 100.0  # Leave 20s margin for 120s limit
    
    best_score = -np.inf
    best_result = None

    # Search aspect ratios
    # Perimeter = 4 => 2(W+H) = 4 => W+H = 2
    # We search W in [0.6, 1.4]. Due to symmetry W and H, 
    # but packing might be asymmetric, so we check a range.
    aspect_ratios = np.arange(0.6, 1.41, 0.1)
    
    # Pre-define indices for pairs
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            pairs.append((i, j))
    pairs = np.array(pairs)
    i_idx, j_idx = pairs[:, 0], pairs[:, 1]

    def objective(v):
        # v: [x1, y1, r1, x2, y2, r2, ...]
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        # Sum of radii
        sum_r = np.sum(r)
        
        # Overlap penalty
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[i_idx] + r[j_idx]
        overlap = np.maximum(0, min_dist - dist)
        overlap_pen = np.sum(overlap**2)
        
        # Boundary penalty
        # Left/Right
        pen_x = np.sum(np.maximum(0, r - x)**2 + np.maximum(0, r - (W - x))**2)
        # Top/Bottom
        pen_y = np.sum(np.maximum(0, r - y)**2 + np.maximum(0, r - (H - y))**2)
        
        total_pen = overlap_pen + pen_x + pen_y
        
        # Objective: Maximize sum_r, minimize penalty
        # Using penalty method
        return -sum_r + 100.0 * total_pen

    def run_optimization(W):
        H = 2.0 - W
        
        # Initialize grid
        # 5x5 grid
        side = 5
        init_x = []
        init_y = []
        init_r = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                # Center of cell
                xi = (i + 0.5) * W / side
                yj = (j + 0.5) * H / side
                # Small initial radius
                ri = 0.05
                init_x.append(xi)
                init_y.append(yj)
                init_r.append(ri)
                count += 1
        
        v0 = np.array([val for pair in zip(init_x, init_y, init_r) for val in pair])
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0, W))
            bounds.append((0, H))
            bounds.append((0.01, 0.5))
        
        # Optimization
        res = minimize(objective, v0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8})
        
        # Extract result
        v_opt = res.x
        x = v_opt[0::3]
        y = v_opt[1::3]
        r = v_opt[2::3]
        
        # Check validity
        # 1. Bounds
        if np.any(r > x) or np.any(r > W - x) or np.any(r > y) or np.any(r > H - y) or np.any(r < 0):
            return None, 0
        
        # 2. Overlaps
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[i_idx] + r[j_idx]
        if np.any(dist < min_dist - 1e-9):
            return None, 0
            
        score = np.sum(r)
        result = np.column_stack((x, y, r))
        return result, score

    for W in aspect_ratios:
        if time.time() - start_time > time_limit:
            break
        
        result, score = run_optimization(W)
        
        if result is not None and score > best_score:
            best_score = score
            best_result = result
            
    if best_result is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
