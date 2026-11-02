import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s buffer
    
    best_val = -np.inf
    best_params = None
    
    # Try a few aspect ratios
    # W + H = 2
    # W in [0.6, 1.4]
    aspect_ratios = [1.0, 0.8, 1.2, 0.7, 1.3]
    
    # Initializations
    # 1. Grid 5x5 (first 21)
    # 2. Random
    
    inits = []
    
    # Grid init
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side))
    # Take first 21
    grid_centers = np.array(cells[:N])
    
    for W in aspect_ratios:
        H = 2.0 - W
        if H <= 0: continue
        
        # Scale grid centers to fit W, H
        x_init = grid_centers[:, 0] * W
        y_init = grid_centers[:, 1] * H
        
        # Initial radii: small
        r_init = np.full(N, 0.05)
        
        # Combine variables: W, x1..xN, y1..yN, r1..rN
        # Note: H is determined by W
        x0 = np.concatenate(([W], x_init, y_init, r_init))
        
        def objective(params):
            W = params[0]
            H = 2.0 - W
            if H <= 0 or W <= 0:
                return 1e6
            
            x = params[1:1+N]
            y = params[1+N:1+2*N]
            r = params[1+2*N:1+3*N]
            
            # Penalize negative radii
            if np.any(r <= 0):
                return 1e6 + np.sum(r[r<=0])
            
            # Boundary violations
            # x in [r, W-r]
            # y in [r, H-r]
            pen = 0.0
            pen += np.sum(np.maximum(0, r - x)**2)
            pen += np.sum(np.maximum(0, r - (W - x))**2)
            pen += np.sum(np.maximum(0, r - y)**2)
            pen += np.sum(np.maximum(0, r - (H - y))**2)
            
            # Overlap violations
            # d_ij >= r_i + r_j
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    overlap = r[i] + r[j] - dist
                    if overlap > 0:
                        pen += overlap**2
            
            return -np.sum(r) + 100.0 * pen
        
        # Bounds
        # W in [0.5, 1.5]
        # x in [0, W] -> [0, 1.5]
        # y in [0, H] -> [0, 1.5]
        # r in [0, 1.0]
        bounds = [(0.5, 1.5)]
        bounds += [(0, 1.5)] * N
        bounds += [(0, 1.5)] * N
        bounds += [(0, 1.0)] * N
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-8})
            if res.success:
                params = res.x
                W = params[0]
                H = 2.0 - W
                x = params[1:1+N]
                y = params[1+N:1+2*N]
                r = params[1+2*N:1+3*N]
                
                # Validate
                valid = True
                if W <= 0 or H <= 0:
                    valid = False
                if np.any(r <= 0):
                    valid = False
                if np.any(x < r) or np.any(x > W-r) or np.any(y < r) or np.any(y > H-r):
                    # Check if violations are small
                    if np.max(np.maximum(0, r-x)) > 1e-5 or np.max(np.maximum(0, r-(W-x))) > 1e-5 or np.max(np.maximum(0, r-y)) > 1e-5 or np.max(np.maximum(0, r-(H-y))) > 1e-5:
                        valid = False
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        if dist < r[i] + r[j] - 1e-5:
                            valid = False
                
                if valid:
                    val = np.sum(r)
                    if val > best_val:
                        best_val = val
                        best_params = (W, x, y, r)
                        
        except Exception:
            pass
            
        if time.time() > deadline:
            break

    if best_params is None:
        # Fallback to seed-like solution
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    W, x, y, r = best_params
    H = 2.0 - W
    
    # Final polish: ensure strict validity by shrinking radii slightly if needed
    # Check overlaps again
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if dist < r[i] + r[j]:
                # Shrink both
                shrink = (r[i] + r[j] - dist) / 2.0 + 1e-6
                r[i] -= shrink
                r[j] -= shrink
                if r[i] < 0: r[i] = 0
                if r[j] < 0: r[j] = 0
                
    # Check boundaries
    for i in range(N):
        if x[i] < r[i]:
            r[i] = x[i] - 1e-6
        if W - x[i] < r[i]:
            r[i] = W - x[i] - 1e-6
        if y[i] < r[i]:
            r[i] = y[i] - 1e-6
        if H - y[i] < r[i]:
            r[i] = H - y[i] - 1e-6
        if r[i] < 0:
            r[i] = 0
            
    result = np.column_stack((x, y, r))
    return result
