"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Start with a grid layout in a square-ish rectangle
    # Perimeter = 2*(w+h) = 4, so w+h = 2
    # Try different aspect ratios
    
    best_value = 0
    best_result = None
    
    # Initial grid positions
    side = int(np.ceil(np.sqrt(N)))
    # Use a 5x5 grid (25 cells), take first 21
    cols, rows = 5, 5
    w, h = 1.0, 1.0  # square, perimeter = 4
    
    # Initial positions centered in cells
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    r0 = np.zeros(N)
    
    for i in range(N):
        col = i % cols
        row = i // cols
        x0[i] = (col + 0.5) * w / cols
        y0[i] = (row + 0.5) * h / rows
        r0[i] = min(w / cols, h / rows) * 0.45
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Bounds: circles must be inside rectangle
        # x[i] >= r[i], x[i] <= w - r[i]
        # y[i] >= r[i], y[i] <= h - r[i]
        penalty = 0
        for i in range(N):
            if x[i] < r[i]:
                penalty += (r[i] - x[i])**2
            if x[i] > w - r[i]:
                penalty += (x[i] - (w - r[i]))**2
            if y[i] < r[i]:
                penalty += (r[i] - y[i])**2
            if y[i] > h - r[i]:
                penalty += (y[i] - (h - r[i]))**2
        
        # Non-overlap: distance between centers >= sum of radii
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty += (min_dist - dist)**2
        
        # Maximize sum of radii -> minimize negative sum
        return -np.sum(r) + 100 * penalty
    
    def constraint_nonoverlap(params):
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                cons.append(dist_sq - min_dist*min_dist)
        # Boundary constraints
        for i in range(N):
            cons.append(x[i] - r[i])
            cons.append(w - r[i] - x[i])
            cons.append(y[i] - r[i])
            cons.append(h - r[i] - y[i])
        return cons
    
    def constraint_nonoverlap_simple(params):
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                cons.append(dist - r[i] - r[j])
        for i in range(N):
            cons.append(x[i] - r[i])
            cons.append(w - r[i] - x[i])
            cons.append(y[i] - r[i])
            cons.append(h - r[i] - y[i])
        return cons
    
    # Try multiple initial configurations and aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        # w + h = 2, w/h = ar => h = 2/(1+ar), w = 2*ar/(1+ar)
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Reset initial positions for this aspect ratio
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        r0 = np.zeros(N)
        
        for i in range(N):
            col = i % cols
            row = i // cols
            x0[i] = (col + 0.5) * w / cols
            y0[i] = (row + 0.5) * h / rows
            r0[i] = min(w / cols, h / rows) * 0.45
        
        params0 = np.concatenate([x0, y0, r0])
        
        bounds = []
        for i in range(N):
            bounds.append((0.0, w))
        for i in range(N):
            bounds.append((0.0, h))
        for i in range(N):
            bounds.append((0.0, min(w, h) * 0.45))
        
        try:
            result = minimize(objective, params0, method='SLSQP', bounds=bounds, 
                             options={'maxiter': 200, 'ftol': 1e-8})
            
            if result.success or result.fun < -best_value:
                x = result.x[:N]
                y = result.x[N:2*N]
                r = result.x[2*N:3*N]
                
                # Verify constraints
                valid = True
                for i in range(N):
                    if x[i] < r[i] - 1e-6 or x[i] > w - r[i] + 1e-6:
                        valid = False
                        break
                    if y[i] < r[i] - 1e-6 or y[i] > h - r[i] + 1e-6:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                            if dist < r[i] + r[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    value = np.sum(r)
                    if value > best_value:
                        best_value = value
                        best_result = np.column_stack([x, y, r])
                        
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    else:
        # Fallback to initial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
# EVOLVE-BLOCK-END
