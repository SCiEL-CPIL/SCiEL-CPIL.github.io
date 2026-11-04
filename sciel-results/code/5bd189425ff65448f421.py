import numpy as np
import time
from scipy.optimize import minimize

N = 12
MIN_DIST = np.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    def get_initial_guess(seed):
        # Use a fixed seed for reproducibility in restarts? 
        # Or just vary the grid slightly.
        # Let's just use the standard grid for all restarts, but maybe randomize initial theta.
        xs = []
        ys = []
        for i in range(3):
            for j in range(4):
                x = j * MIN_DIST + (i % 2) * (MIN_DIST / 2)
                y = i * 1.5
                xs.append(x)
                ys.append(y)
        
        xs = np.array(xs)
        ys = np.array(ys)
        
        cx = (xs.max() + xs.min()) / 2
        cy = (ys.max() + ys.min()) / 2
        xs -= cx
        ys -= cy
        
        r_max = np.max(np.sqrt(xs**2 + ys**2))
        L_init = r_max / (np.sqrt(3) / 2) + 0.5
        theta_init = seed * 0.1 # Vary initial angle
        
        return np.concatenate([xs, ys, [L_init, theta_init]])

    def objective(v):
        if time.time() > deadline:
            return 1e10
        xs_v = v[0:12]
        ys_v = v[12:24]
        L = v[24]
        theta_out = v[25]
        
        obj = L
        
        # Overlap penalty
        for i in range(12):
            for j in range(i+1, 12):
                dx = xs_v[i] - xs_v[j]
                dy = ys_v[i] - ys_v[j]
                d = np.sqrt(dx**2 + dy**2)
                if d < MIN_DIST:
                    obj += 10000 * (MIN_DIST - d)**2
        
        # Containment penalty
        inradius = L * np.sqrt(3) / 2
        for k in range(6):
            angle_normal = theta_out + np.pi/6 + k * np.pi / 3
            nx = np.cos(angle_normal)
            ny = np.sin(angle_normal)
            for i in range(12):
                dist = xs_v[i]*nx + ys_v[i]*ny
                if dist > inradius:
                    obj += 10000 * (dist - inradius)**2
        
        return obj

    bounds = [(-20, 20)] * 24 + [(1.0, 20.0), (0.0, 2*np.pi)]
    
    best_obj = 1e10
    best_v = None
    
    # Run 3 restarts
    for restart in range(3):
        if time.time() > deadline - 20:
            break
            
        x0 = get_initial_guess(restart)
        
        try:
            # L-BFGS-B
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-8})
            if res.fun < best_obj:
                best_obj = res.fun
                best_v = res.x
                
            # Nelder-Mead refinement
            if time.time() < deadline - 5:
                res2 = minimize(objective, best_v, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6})
                if res2.fun < best_obj:
                    best_obj = res2.fun
                    best_v = res2.x
                    
        except Exception:
            pass

    if best_v is None:
        # Fallback to simple grid
        xs = []
        ys = []
        for i in range(3):
            for j in range(4):
                x = j * MIN_DIST + (i % 2) * (MIN_DIST / 2)
                y = i * 1.5
                xs.append(x)
                ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        cx = (xs.max() + xs.min()) / 2
        cy = (ys.max() + ys.min()) / 2
        xs -= cx
        ys -= cy
        r_max = np.max(np.sqrt(xs**2 + ys**2))
        L = r_max / (np.sqrt(3) / 2) + 0.1
        theta = 0.0
        best_v = np.concatenate([xs, ys, [L, theta]])

    xs_best = best_v[0:12]
    ys_best = best_v[12:24]
    L_best = best_v[24]
    theta_best = best_v[25]
    
    inner = []
    for i in range(12):
        inner.append([xs_best[i], ys_best[i], 0.0])
        
    center = (0.0, 0.0)
    side = L_best
    angle = np.degrees(theta_best)
    
    return inner, center, side, angle
