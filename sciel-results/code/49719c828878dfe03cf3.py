"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Initial guess: grid layout in a rectangle with w+h=2
    # Try a few aspect ratios
    best_result = None
    best_value = -np.inf
    
    # Try different aspect ratios w/h
    aspect_ratios = [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        w = 2.0 * ar / (1 + ar)
        h = 2.0 - w
        
        # Grid initialization
        cols = int(np.ceil(np.sqrt(N * w / h)))
        rows = int(np.ceil(N / cols))
        
        centers = []
        for i in range(cols):
            for j in range(rows):
                if len(centers) >= N:
                    break
                cx = (i + 0.5) * w / cols
                cy = (j + 0.5) * h / rows
                r = 0.5 * min(w / cols, h / rows) * 0.95
                centers.append([cx, cy, r])
        
        if len(centers) < N:
            continue
        
        x0 = np.array(centers[:N]).flatten()
        
        def objective(x):
            # Maximize sum of radii -> minimize negative
            cx = x[0::3]
            cy = x[1::3]
            r = x[2::3]
            
            # Bounding box
            min_x = np.min(cx - r)
            max_x = np.max(cx + r)
            min_y = np.min(cy - r)
            max_y = np.max(cy + r)
            W = max_x - min_x
            H = max_y - min_y
            
            # Perimeter constraint: W + H <= 2
            penalty = 0.0
            if W + H > 2.0:
                penalty += 1000 * (W + H - 2.0)
            
            # Non-overlap penalty
            for i in range(N):
                for j in range(i+1, N):
                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r[i] + r[j]:
                        penalty += 1000 * (r[i] + r[j] - dist)
            
            # Keep circles in reasonable bounds
            for i in range(N):
                if cx[i] - r[i] < -1.0:
                    penalty += 1000 * (-1.0 - (cx[i] - r[i]))
                if cx[i] + r[i] > 3.0:
                    penalty += 1000 * (cx[i] + r[i] - 3.0)
                if cy[i] - r[i] < -1.0:
                    penalty += 1000 * (-1.0 - (cy[i] - r[i]))
                if cy[i] + r[i] > 3.0:
                    penalty += 1000 * (cy[i] + r[i] - 3.0)
            
            return -np.sum(r) + penalty
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if time.time() > deadline:
                break
                
            if res.success or res.fun < best_value:
                x_opt = res.x
                cx = x_opt[0::3]
                cy = x_opt[1::3]
                r = x_opt[2::3]
                
                # Verify constraints
                min_x = np.min(cx - r)
                max_x = np.max(cx + r)
                min_y = np.min(cy - r)
                max_y = np.max(cy + r)
                W = max_x - min_x
                H = max_y - min_y
                
                valid = True
                if W + H > 2.0 + 1e-6:
                    valid = False
                
                for i in range(N):
                    for j in range(i+1, N):
                        dx = cx[i] - cx[j]
                        dy = cy[i] - cy[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < r[i] + r[j] - 1e-6:
                            valid = False
                
                if valid and np.all(r > 0):
                    value = np.sum(r)
                    if value > best_value:
                        best_value = value
                        result = np.column_stack([cx, cy, r])
                        best_result = result
                        
        except Exception as e:
            pass
    
    # Fallback: simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    return best_result

# EVOLVE-BLOCK-END
