import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    # Aspect ratios to try
    # W + H = 2
    # ar = W/H
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.66, 1.8, 0.55]
    
    best_score = 0
    best_result = None

    for ar in aspect_ratios:
        H = 2.0 / (1.0 + ar)
        W = ar * H
        
        # Sequential Greedy Placement
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Place first circle at center
        x[0] = W / 2.0
        y[0] = H / 2.0
        r[0] = 0.01
        
        # Grid for finding next best spot
        grid_res = 40
        xs_grid = np.linspace(0.05, W - 0.05, grid_res)
        ys_grid = np.linspace(0.05, H - 0.05, grid_res)
        
        for i in range(1, N):
            best_x = W / 2.0
            best_y = H / 2.0
            best_min_dist = -1
            
            # Vectorized grid evaluation for speed
            XG, YG = np.meshgrid(xs_grid, ys_grid)
            
            # Distance to walls
            dist_walls = np.minimum(np.minimum(XG, W - XG), np.minimum(YG, H - YG))
            
            # Distance to existing circles
            min_dist_to_circles = np.full_like(XG, np.inf)
            for j in range(i):
                dist_sq = (XG - x[j])**2 + (YG - y[j])**2
                dist = np.sqrt(dist_sq)
                min_dist_to_circles = np.minimum(min_dist_to_circles, dist)
            
            # The "clearance" is the min of wall dist and circle dist
            clearance = np.minimum(dist_walls, min_dist_to_circles)
            
            # Find max clearance
            max_idx = np.argmax(clearance)
            best_min_dist = clearance.flat[max_idx]
            best_x = XG.flat[max_idx]
            best_y = YG.flat[max_idx]
            
            x[i] = best_x
            y[i] = best_y
            r[i] = best_min_dist / 2.0
            # Cap by walls
            r[i] = min(r[i], best_x, W - best_x, best_y, H - best_y)
            r[i] = max(r[i], 0.001)

        # Optimization
        def obj(v):
            return -np.sum(v[2*N:3*N])
            
        constraints = []
        for i in range(N):
            for j in range(i+1, N):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[i]-v[N+j])**2 + (v[N+i]-v[2*N+j])**2 - (v[2*N+i]+v[2*N+j])**2
                })
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: W - v[i] - v[2*N+i]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: H - v[N+i] - v[2*N+i]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*N+i]})
            
        x0 = np.concatenate([x, y, r])
        
        try:
            res = minimize(obj, x0, method='SLSQP', constraints=constraints, 
                           options={'maxiter': 50, 'ftol': 1e-5})
            
            if res.success:
                x_opt = res.x[:N]
                y_opt = res.x[N:2*N]
                r_opt = res.x[2*N:3*N]
                
                # Quick validation
                valid = True
                for i in range(N):
                    if r_opt[i] < 1e-6: valid = False; break
                    if x_opt[i] - r_opt[i] < -1e-4 or x_opt[i] + r_opt[i] > W + 1e-4: valid = False; break
                    if y_opt[i] - r_opt[i] < -1e-4 or y_opt[i] + r_opt[i] > H + 1e-4: valid = False; break
                    for j in range(i+1, N):
                        d2 = (x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2
                        rs = r_opt[i] + r_opt[j]
                        if d2 < rs**2 - 1e-4:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    score = np.sum(r_opt)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([x_opt, y_opt, r_opt])
                        
        except Exception:
            pass

    if best_result is None:
        # Fallback
        side = 5
        x = np.array([(i % side + 0.5) / side for i in range(N)])
        y = np.array([(i // side + 0.5) / side for i in range(N)])
        r = np.full(N, 0.5 / side * 0.8)
        best_result = np.column_stack([x, y, r])

    return best_result
