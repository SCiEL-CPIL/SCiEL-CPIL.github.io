import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit

    best_score = -np.inf
    best_result = None

    def objective(params, a):
        """
        params: [x0, y0, r0, x1, y1, r1, ...]
        a: aspect ratio w/h
        """
        w = 2 * a / (1 + a)
        h = 2 / (1 + a)
        
        xs = params[0::3]
        ys = params[1::3]
        rs = params[2::3]
        
        # Boundary penalties
        pen = 0.0
        for i in range(N):
            # x bounds
            if xs[i] < rs[i]:
                pen += (rs[i] - xs[i])**2
            if xs[i] > w - rs[i]:
                pen += (xs[i] - (w - rs[i]))**2
            # y bounds
            if ys[i] < rs[i]:
                pen += (rs[i] - ys[i])**2
            if ys[i] > h - rs[i]:
                pen += (ys[i] - (h - ys[i]))**2
        
        # Overlap penalties
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                min_dist = rs[i] + rs[j]
                # Use squared distance to avoid sqrt if possible? 
                # But we need dist for comparison.
                dist = np.sqrt(dist_sq)
                if dist < min_dist:
                    pen += (min_dist - dist)**2
        
        # Objective: Maximize sum of radii -> Minimize -sum(rs) + penalty
        # Weight 10.0 is a guess, might need tuning
        return -np.sum(rs) + 10.0 * pen

    def check_validity(res, a):
        w = 2 * a / (1 + a)
        h = 2 / (1 + a)
        xs = res.x[0::3]
        ys = res.x[1::3]
        rs = res.x[2::3]
        
        # Check bounds
        for i in range(N):
            if xs[i] < rs[i] - 1e-5 or xs[i] > w - rs[i] + 1e-5:
                return False, 0.0
            if ys[i] < rs[i] - 1e-5 or ys[i] > h - rs[i] + 1e-5:
                return False, 0.0
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                if dist < rs[i] + rs[j] - 1e-5:
                    return False, 0.0
        
        return True, np.sum(rs)

    # Search over aspect ratios
    # Aspect ratio a = w/h. Since w+h=2, a=1 is square.
    # Try a range.
    aspect_ratios = np.linspace(1.0, 2.0, 8)
    
    # For each aspect ratio, do a few restarts
    n_restarts = 3
    
    for a in aspect_ratios:
        if time.time() - start_time > time_limit - 10:
            break
            
        w = 2 * a / (1 + a)
        h = 2 / (1 + a)
        
        for r in range(n_restarts):
            if time.time() - start_time > time_limit - 5:
                break
                
            # Initialize grid
            side = int(np.ceil(np.sqrt(N))) # 5
            # Create grid points
            # We want to distribute N circles in a side x side grid
            xs_init = []
            ys_init = []
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    # Center of cell
                    x_c = (i + 0.5) * (w / side)
                    y_c = (j + 0.5) * (h / side)
                    xs_init.append(x_c)
                    ys_init.append(y_c)
                    idx += 1
            
            xs_init = np.array(xs_init)
            ys_init = np.array(ys_init)
            
            # Initial radius
            # Max radius limited by cell size
            r_init = 0.4 * min(w / side, h / side)
            rs_init = np.full(N, r_init)
            
            # Add jitter
            xs_init += np.random.normal(0, 0.05, N)
            ys_init += np.random.normal(0, 0.05, N)
            
            params = np.zeros(3 * N)
            params[0::3] = xs_init
            params[1::3] = ys_init
            params[2::3] = rs_init
            
            # Optimize
            res = minimize(objective, params, args=(a,), method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-5})
            
            if res.success:
                valid, score = check_validity(res, a)
                if valid and score > best_score:
                    best_score = score
                    best_result = np.column_stack((res.x[0::3], res.x[1::3], res.x[2::3]))

    # Fallback to seed if optimization failed
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
