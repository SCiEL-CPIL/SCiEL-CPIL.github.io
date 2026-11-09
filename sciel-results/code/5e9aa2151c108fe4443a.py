import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin from 150s limit
    
    best_solution = None
    best_score = 0.0
    best_ar = 1.0
    
    # We will try a few aspect ratios. 
    # Perimeter 4 => w+h=2.
    # Square is often optimal for dense packings, but elongated might allow specific patterns.
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.7, 1.1, 0.9, 1.3, 0.75]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 15:
            break
            
        h = 2.0 / (ar + 1.0)
        w = 2.0 * ar / (ar + 1.0)
        
        # Initialize parameters: [x0, y0, r0, x1, y1, r1, ...]
        # Start with a grid-like initialization to give the optimizer a good starting point
        # 5x5 grid covers 25 spots, we need 21.
        side = 5
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        r0 = np.zeros(N)
        
        # Fill first 21 spots of a 5x5 grid
        for i in range(N):
            row = i // side
            col = i % side
            x0[i] = (col + 0.5) * (w / side)
            y0[i] = (row + 0.5) * (h / side)
            # Initial radius slightly less than grid cell size to ensure validity
            r0[i] = (min(w, h) / side) * 0.4
        
        # Randomize slightly to break symmetry
        x0 += np.random.uniform(-0.02, 0.02, N)
        y0 += np.random.uniform(-0.02, 0.02, N)
        
        params = np.concatenate([x0, y0, r0])
        
        def objective(params):
            x = params[0::3]
            y = params[1::3]
            r = params[2::3]
            
            # Objective: Maximize sum(r) => Minimize -sum(r)
            obj = -np.sum(r)
            
            # Penalty for boundary violations
            # We want x >= r, x <= w-r, y >= r, y <= h-r
            # Violation amount: max(0, r - x), max(0, x + r - w), etc.
            pen = 0.0
            
            # X boundaries
            pen += np.sum(np.maximum(0, r - x)**2)
            pen += np.sum(np.maximum(0, x + r - w)**2)
            
            # Y boundaries
            pen += np.sum(np.maximum(0, r - y)**2)
            pen += np.sum(np.maximum(0, y + r - h)**2)
            
            # Overlaps
            # We want dist(i,j) >= r[i] + r[j]
            # Violation: max(0, (r[i] + r[j]) - dist(i,j))
            # Vectorized calculation for speed
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            
            min_dist = r[:, None] + r[None, :]
            
            # Zero out diagonal
            np.fill_diagonal(dist, np.inf)
            np.fill_diagonal(min_dist, 0)
            
            violations = np.maximum(0, min_dist - dist)
            pen += np.sum(violations**2)
            
            # Weight the penalty heavily to enforce constraints
            # As optimization proceeds, we could anneal this weight, but a high fixed weight
            # often works well with L-BFGS-B if the initial state is close to valid.
            # Let's use a large weight.
            weight = 100.0
            return obj + weight * pen
        
        # Run optimization
        # Use L-BFGS-B which handles bounds well.
        # Bounds: x in [0, w], y in [0, h], r in [0, min(w,h)/2]
        bounds = []
        max_r = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((0, max_r))
            
        # Multiple restarts with different random seeds for robustness
        # But we have time limits. Let's do 2-3 restarts per AR if time permits.
        
        num_restarts = 3
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Perturb initial state for subsequent restarts
            if restart > 0:
                params += np.random.uniform(-0.1, 0.1, 3*N)
                # Ensure bounds are respected in initialization
                params[0::3] = np.clip(params[0::3], 0, w)
                params[1::3] = np.clip(params[1::3], 0, h)
                params[2::3] = np.clip(params[2::3], 0, max_r)
            
            try:
                res = minimize(objective, params, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                continue
            
            if res.success or res.fun < 1e5: # Check if it converged reasonably
                x_opt = res.x[0::3]
                y_opt = res.x[1::3]
                r_opt = res.x[2::3]
                
                # Check validity strictly
                valid = True
                for i in range(N):
                    if x_opt[i] < r_opt[i] or x_opt[i] > w - r_opt[i] or y_opt[i] < r_opt[i] or y_opt[i] > h - r_opt[i]:
                        valid = False
                        break
                
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = x_opt[i] - x_opt[j]
                            dy = y_opt[i] - y_opt[j]
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < r_opt[i] + r_opt[j] - 1e-9: # Small tolerance
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    score = np.sum(r_opt)
                    if score > best_score:
                        best_score = score
                        best_solution = np.column_stack((x_opt, y_opt, r_opt))
                        best_ar = ar

    if best_solution is None:
        # Fallback: Simple grid
        side = 5
        x_list, y_list, r_list = [], [], []
        for i in range(side):
            for j in range(side):
                if len(x_list) >= N:
                    break
                x_list.append((j + 0.5) / side)
                y_list.append((i + 0.5) / side)
                r_list.append(0.5 / side - 0.01)
        best_solution = np.column_stack((x_list, y_list, r_list))
        
    return best_solution
