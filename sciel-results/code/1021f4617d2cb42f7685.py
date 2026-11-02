import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    best_score = -np.inf
    best_result = None
    
    # Function to compute score (sum of radii)
    def get_score(res):
        return np.sum(res[:, 2])

    # Objective function: minimize negative sum of radii
    # Variables: x1, y1, r1, ..., xN, yN, rN, w (width)
    # Height h = 2 - w
    def objective(params):
        # Check time
        if time.time() > deadline:
            return 0  # Dummy return to stop optimization
        
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        w = params[3*N]
        h = 2.0 - w
        
        # If w or h is invalid, return large penalty
        if w <= 0 or h <= 0:
            return 1e6
        
        # Penalty for circles outside the box
        penalty = 0
        for i in range(N):
            if x[i] < r[i] or x[i] > w - r[i]:
                penalty += (min(x[i], w - x[i]) - r[i])**2
            if y[i] < r[i] or y[i] > h - r[i]:
                penalty += (min(y[i], h - y[i]) - r[i])**2
        
        # Penalty for overlapping circles
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                min_dist = r[i] + r[j]
                if dist_sq < min_dist**2:
                    penalty += (min_dist - np.sqrt(dist_sq))**2
        
        # Maximize sum of radii, so minimize negative sum
        # Add penalty to discourage violations
        return -np.sum(r) + penalty * 10.0

    # Constraints for SLSQP
    # 1. w must be in (0, 2)
    # 2. Bounds on x, y, r are handled by bounds parameter
    # 3. Non-overlap constraints are handled by penalty in objective for simplicity in SLSQP
    #    However, SLSQP is better with explicit constraints. Let's use explicit constraints.
    
    def constraints_func(params):
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        w = params[3*N]
        h = 2.0 - w
        
        cons = []
        
        # Non-overlap constraints: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
        # SLSQP requires g(x) >= 0
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda p, i=i, j=j: (p[i] - p[j+N])**2 + (p[i+N] - p[j+2*N])**2 - (p[i+2*N] + p[j+2*N])**2
                })
        
        # Bounds constraints are handled by 'bounds' argument
        return cons

    # Bounds for variables
    # x, y: (0, 2) is a safe upper bound since max dimension is 2
    # r: (0, 1) is a safe upper bound (max radius in 2x1 box is 0.5, but let's be loose)
    # w: (0.1, 1.9) to ensure h > 0.1
    
    bounds = []
    for i in range(N):
        bounds.append((1e-6, 2.0)) # x
        bounds.append((1e-6, 2.0)) # y
        bounds.append((1e-6, 1.0)) # r
    bounds.append((0.1, 1.9)) # w

    def init_params(w_init):
        h_init = 2.0 - w_init
        # Simple hexagonal or grid initialization
        # Let's try a grid of 5x5 (25 slots) and pick 21, or 3x7
        # For robustness, let's use a hexagonal pattern scaled to fit w, h
        
        # Approximate number of rows and cols for hex packing
        # Area of circle ~ pi*r^2. Total area ~ N*pi*r^2. Box area = w*h.
        # r ~ sqrt(w*h / (N * 1.1)) (1.1 is packing factor)
        r_est = np.sqrt((w_init * h_init) / (N * 1.15))
        
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.full(N, r_est)
        
        # Hexagonal layout
        rows = 0
        cols = 0
        # Estimate rows/cols
        # dx = 2*r, dy = sqrt(3)*r
        # Let's just place them in a grid for simplicity first, then let optimizer fix
        # Grid 5x5 is 25, we need 21.
        side = int(np.ceil(np.sqrt(N)))
        # side=5. 5x5=25.
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N: break
                # Center in cell
                x[idx] = (j + 0.5) * (w_init / side)
                y[idx] = (i + 0.5) * (h_init / side)
                r[idx] = 0.45 * min(w_init / side, h_init / side)
                idx += 1
        
        params = np.concatenate([x, y, r, [w_init]])
        return params

    # Try multiple initial aspect ratios
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8]
    
    for w_init in aspect_ratios:
        if time.time() > deadline - 5: break
        
        x0 = init_params(w_init)
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds,
                constraints=constraints_func,
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            
            if res.success or res.fun < 0: # res.fun < 0 means positive sum of radii (since we minimize -sum)
                # Extract result
                x = res.x[:N]
                y = res.x[N:2*N]
                r = res.x[2*N:3*N]
                w = res.x[3*N]
                h = 2.0 - w
                
                # Check validity
                valid = True
                # Check bounds
                if w <= 0 or h <= 0: valid = False
                for i in range(N):
                    if x[i] < r[i] - 1e-5 or x[i] > w - r[i] + 1e-5:
                        valid = False
                    if y[i] < r[i] - 1e-5 or y[i] > h - r[i] + 1e-5:
                        valid = False
                    if r[i] <= 0:
                        valid = False
                
                # Check overlaps
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                            min_dist = r[i] + r[j]
                            if dist_sq < min_dist**2 - 1e-5:
                                valid = False
                                break
                        if not valid: break
                
                if valid:
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack((x, y, r))
                        
        except Exception:
            continue

    if best_result is None:
        # Fallback: simple grid
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (j + 0.5) * (2.0 / side)
                    y = (i + 0.5) * (2.0 / side)
                    r = 0.45 * (2.0 / side)
                    cells.append([x, y, r])
        best_result = np.array(cells)
        
    return best_result
