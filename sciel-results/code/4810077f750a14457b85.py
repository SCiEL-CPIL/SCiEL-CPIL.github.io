import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_val = -1.0
    best_x = None
    
    # Aspect ratios to try
    # w+h=2
    ws = np.arange(0.7, 1.31, 0.05)
    
    for w in ws:
        if time.time() > deadline:
            break
        h = 2.0 - w
        
        # Number of seeds
        n_seeds = 50
        if time.time() > deadline - 30:
            n_seeds = 10
            
        for seed in range(n_seeds):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(seed)
            
            # Initialization
            # Random points
            x0 = np.random.rand(N, 3)
            x0[:, 0] *= w
            x0[:, 1] *= h
            # Start with small radii to allow growth
            x0[:, 2] = 0.05
            
            bounds = []
            for i in range(N):
                bounds.append((0, w))
                bounds.append((0, h))
                bounds.append((0.01, min(w, h) / 2))
            
            def objective(v):
                v = v.reshape(N, 3)
                xs = v[:, 0]
                ys = v[:, 1]
                rs = v[:, 2]
                
                sum_r = np.sum(rs)
                
                # Overlap
                dx = xs[:, None] - xs[None, :]
                dy = ys[:, None] - ys[None, :]
                d = np.sqrt(dx**2 + dy**2)
                r_sum = rs[:, None] + rs[None, :]
                overlap = np.maximum(0, r_sum - d)
                np.fill_diagonal(overlap, 0)
                pen_overlap = np.sum(overlap**2)
                
                # Boundary
                vx_low = np.maximum(0, rs - xs)
                vx_high = np.maximum(0, xs - (w - rs))
                vy_low = np.maximum(0, rs - ys)
                vy_high = np.maximum(0, ys - (h - rs))
                
                pen_bound = np.sum(vx_low**2) + np.sum(vx_high**2) + np.sum(vy_low**2) + np.sum(vy_high**2)
                
                return -sum_r + 1000.0 * (pen_overlap + pen_bound)
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if res.success:
                    x_opt = res.x.reshape(N, 3)
                    
                    # Strict validation
                    valid = True
                    # Check bounds with small margin
                    margin = 1e-6
                    for i in range(N):
                        if x_opt[i, 0] < x_opt[i, 2] - margin or x_opt[i, 0] > w - x_opt[i, 2] + margin:
                            valid = False
                            break
                        if x_opt[i, 1] < x_opt[i, 2] - margin or x_opt[i, 1] > h - x_opt[i, 2] + margin:
                            valid = False
                            break
                    if valid:
                        for i in range(N):
                            for j in range(i+1, N):
                                dx = x_opt[i, 0] - x_opt[j, 0]
                                dy = x_opt[i, 1] - x_opt[j, 1]
                                d = np.sqrt(dx**2 + dy**2)
                                if d < x_opt[i, 2] + x_opt[j, 2] - margin:
                                    valid = False
                                    break
                            if not valid:
                                break
                    
                    if valid:
                        val = np.sum(x_opt[:, 2])
                        if val > best_val:
                            best_val = val
                            best_x = x_opt.copy()
            except Exception:
                pass

    if best_x is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_x
