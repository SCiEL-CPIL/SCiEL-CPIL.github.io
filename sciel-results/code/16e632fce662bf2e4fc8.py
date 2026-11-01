import numpy as np
from scipy.optimize import minimize
import time

N = 32
START_TIME = time.time()
TIME_LIMIT = 110.0

def construct():
    global START_TIME
    start_time = time.time()
    
    # Initial grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side
                cells.append([x, y, r])
    init_grid = np.array(cells)
    
    def objective(params):
        if time.time() - start_time > TIME_LIMIT:
            return 1e10
            
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        obj = -np.sum(r)
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx*dx + dy*dy)
        
        min_dist = r[:, None] + r[None, :]
        diff = min_dist - dist
        smooth_max = 0.5 * (diff + np.sqrt(diff*diff + 1e-6))
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        overlap_pen = np.sum(smooth_max[mask])
        
        diff_x_neg = r - x
        diff_x_pos = x - (1.0 - r)
        pen_x_neg = 0.5 * (diff_x_neg + np.sqrt(diff_x_neg*diff_x_neg + 1e-6))
        pen_x_pos = 0.5 * (diff_x_pos + np.sqrt(diff_x_pos*diff_x_pos + 1e-6))
        
        diff_y_neg = r - y
        diff_y_pos = y - (1.0 - r)
        pen_y_neg = 0.5 * (diff_y_neg + np.sqrt(diff_y_neg*diff_y_neg + 1e-6))
        pen_y_pos = 0.5 * (diff_y_pos + np.sqrt(diff_y_pos*diff_y_pos + 1e-6))
        
        bound_pen = np.sum(pen_x_neg) + np.sum(pen_x_pos) + np.sum(pen_y_neg) + np.sum(pen_y_pos)
        
        mu = 100.0
        total_obj = obj + mu * (overlap_pen + bound_pen)
        
        return total_obj

    def make_valid(params):
        x = params[:N].copy()
        y = params[N:2*N].copy()
        r = params[2*N:3*N].copy()
        
        for _ in range(100):
            valid = True
            for i in range(N):
                if x[i] < r[i]:
                    r[i] = x[i] * 0.99
                    valid = False
                if 1.0 - x[i] < r[i]:
                    r[i] = (1.0 - x[i]) * 0.99
                    valid = False
                if y[i] < r[i]:
                    r[i] = y[i] * 0.99
                    valid = False
                if 1.0 - y[i] < r[i]:
                    r[i] = (1.0 - y[i]) * 0.99
                    valid = False
            
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist_sq = dx*dx + dy*dy
                        min_dist = r[i] + r[j]
                        if dist_sq < min_dist*min_dist:
                            factor = 0.99
                            r[i] *= factor
                            r[j] *= factor
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                break
                
        return np.concatenate([x, y, r])

    best_params = init_grid.flatten()
    best_score = np.sum(init_grid[:, 2])
    
    np.random.seed(42)
    num_restarts = 50
    
    for k in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT:
            break
            
        if k == 0:
            x0 = init_grid.flatten()
        else:
            x0 = init_grid.flatten().copy()
            noise_scale = 0.1
            x0[:N] += np.random.uniform(-noise_scale, noise_scale, N)
            x0[N:2*N] += np.random.uniform(-noise_scale, noise_scale, N)
            x0[2*N:3*N] = np.maximum(0.01, x0[2*N:3*N] + np.random.uniform(-0.05, 0.05, N))
            
            x0[:N] = np.clip(x0[:N], 0, 1)
            x0[N:2*N] = np.clip(x0[N:2*N], 0, 1)

        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12})
            # Check if this result is better
            # We need to make it valid first to compare fairly?
            # Or compare objective?
            # Objective includes penalty. Lower objective is better.
            # But we want max sum of radii.
            # If penalty is 0, objective = -sum(r).
            # So lower objective means higher sum(r) if penalty is 0.
            # If penalty > 0, it's a trade-off.
            # We should prioritize valid solutions with high sum(r).
            
            # Let's make it valid
            valid_params = make_valid(res.x)
            score = np.sum(valid_params[2*N:3*N])
            
            # Compare with best
            # We should also check if the valid_params is actually valid (make_valid should ensure it)
            # And compare score
            if score > best_score:
                best_score = score
                best_params = valid_params
        except Exception as e:
            pass

    # Final polish
    if time.time() - start_time < TIME_LIMIT - 20:
        try:
            res = minimize(objective, best_params, method='Powell', options={'maxiter': 1000, 'ftol': 1e-12})
            valid_params = make_valid(res.x)
            score = np.sum(valid_params[2*N:3*N])
            if score > best_score:
                best_score = score
                best_params = valid_params
        except Exception as e:
            pass

    x = best_params[:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    result = np.column_stack((x, y, r))
    return result
