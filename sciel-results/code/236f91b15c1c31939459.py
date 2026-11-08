import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 21

def _objective_and_penalty(vars, W, H):
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    obj = -np.sum(r)
    penalty = 0.0
    
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                penalty += (min_dist - dist)**2
    
    for i in range(N):
        if x[i] < r[i]:
            penalty += (r[i] - x[i])**2
        if x[i] > W - r[i]:
            penalty += (x[i] - (W - r[i]))**2
        if y[i] < r[i]:
            penalty += (r[i] - y[i])**2
        if y[i] > H - r[i]:
            penalty += (y[i] - (H - r[i]))**2
        if r[i] < 0:
            penalty += r[i]**2
            
    return obj + 500.0 * penalty

def _validate(x, y, r, W, H):
    if np.any(r <= 0):
        return False
    for i in range(N):
        if x[i] < r[i] - 1e-6 or x[i] > W - r[i] + 1e-6:
            return False
        if y[i] < r[i] - 1e-6 or y[i] > H - r[i] + 1e-6:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if dist < r[i] + r[j] - 1e-6:
                return False
    return True

def _run_local_optimize(vars0, W, H, bounds, maxiter=2000):
    res = minimize(
        _objective_and_penalty, vars0, args=(W, H),
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    return res.x

def construct():
    start_time = time.time()
    time_budget = 115.0
    
    best_score = 0.0
    best_vars = None
    best_W = None
    best_H = None
    
    # Candidate aspect ratios (W/H). Perimeter = 2(W+H) <= 4 => W+H <= 2.
    # We'll fix H and compute W from aspect ratio, or fix sum.
    # Let's parameterize by ratio = W/H. Then W = ratio * H.
    # Perimeter constraint: 2*ratio*H + 2*H <= 4 => H <= 2 / (ratio + 1)
    # To maximize area, we use H = 2 / (ratio + 1) and W = ratio * H.
    # This gives the largest rectangle for a given aspect ratio.
    
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.33, 1.5, 1.73, 2.0, 2.5, 3.0]
    
    for ar in aspect_ratios:
        if time.time() - start_time > time_budget:
            break
            
        H = 2.0 / (ar + 1.0)
        W = ar * H
        
        if W < 0.1 or H < 0.1:
            continue
            
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0.0, W))
            bounds.append((0.0, H))
            bounds.append((0.0, min(W, H)/2.0))
        
        # Try multiple initializations
        trials_per_aspect = 3
        
        for trial in range(trials_per_aspect):
            if time.time() - start_time > time_budget:
                break
                
            np.random.seed(123 + trial * 10 + int(ar * 10))
            
            # Initialization 1: Grid
            if trial == 0:
                cols = 5
                rows = 5
                x0 = []
                y0 = []
                r0 = []
                for i in range(N):
                    col = i % cols
                    row = i // cols
                    xi = (col + 0.5) * W / cols + np.random.uniform(-W*0.05, W*0.05)
                    yi = (row + 0.5) * H / rows + np.random.uniform(-H*0.05, H*0.05)
                    xi = np.clip(xi, 0.01, W - 0.01)
                    yi = np.clip(yi, 0.01, H - 0.01)
                    x0.append(xi)
                    y0.append(yi)
                    r0.append(min(W, H) * 0.08)
                    
            # Initialization 2: Hexagonal
            elif trial == 1:
                x0 = []
                y0 = []
                r0 = []
                # Hex grid
                r_base = 0.05 * min(W, H)
                dx = 2 * r_base * 0.866
                dy = 2 * r_base * 0.75
                for row in range(8):
                    for col in range(8):
                        if len(x0) >= N:
                            break
                        xi = col * dx + (dx/2 if row % 2 == 1 else 0)
                        yi = row * dy
                        # Center it
                        xi = (xi - 3.5 * dx) + W/2
                        yi = (yi - 3.5 * dy) + H/2
                        xi = np.clip(xi, 0.01, W - 0.01)
                        yi = np.clip(yi, 0.01, H - 0.01)
                        x0.append(xi)
                        y0.append(yi)
                        r0.append(r_base)
                        
            # Initialization 3: Random
            else:
                x0 = np.random.uniform(0.05, W - 0.05, N)
                y0 = np.random.uniform(0.05, H - 0.05, N)
                r0 = np.full(N, 0.04 * min(W, H))
                
            vars0 = []
            for i in range(N):
                vars0.extend([x0[i], y0[i], r0[i]])
                
            # Run local optimization
            vars_res = _run_local_optimize(vars0, W, H, bounds)
            
            x = vars_res[0::3]
            y = vars_res[1::3]
            r = vars_res[2::3]
            
            if _validate(x, y, r, W, H):
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_vars = vars_res
                    best_W = W
                    best_H = H
                    
    # Final fallback: 1x1 grid
    if best_vars is None:
        W = 1.0
        H = 1.0
        side = 5
        cells = []
        for i in range(N):
            col = i % side
            row = i // side
            cells.append([ (col+0.5)/side, (row+0.5)/side, 0.5/side - 0.01 ])
        best_vars = np.array(cells).flatten()
        best_W = 1.0
        best_H = 1.0
        
    # One final polish step on the best candidate found
    if best_vars is not None and best_W is not None and best_H is not None:
        if time.time() - start_time < time_budget - 5:
            bounds = []
            for i in range(N):
                bounds.append((0.0, best_W))
                bounds.append((0.0, best_H))
                bounds.append((0.0, min(best_W, best_H)/2.0))
            
            vars_res = _run_local_optimize(best_vars, best_W, best_H, bounds, maxiter=5000)
            x = vars_res[0::3]
            y = vars_res[1::3]
            r = vars_res[2::3]
            
            if _validate(x, y, r, best_W, best_H):
                best_vars = vars_res
                
    # Format output: (x, y, r)
    result = []
    for i in range(N):
        result.append([best_vars[3*i], best_vars[3*i+1], best_vars[3*i+2]])
        
    return np.array(result)
