import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    # Fallback solution
    s = (1.0 / 3.0) - 1e-6
    t = 1.0 - 3.0 * s
    centers_fb = np.zeros((n, 2))
    sides_fb = np.zeros(n)
    
    idx = 0
    for r in range(3):
        for c in range(3):
            centers_fb[idx, 0] = (c + 0.5) * s
            centers_fb[idx, 1] = (r + 0.5) * s
            sides_fb[idx] = s
            idx += 1
    centers_fb[idx, 0] = 1.0 - 0.5 * t
    centers_fb[idx, 1] = 1.0 - 0.5 * t
    sides_fb[idx] = t
    
    best_val = np.sum(sides_fb)
    best_centers = centers_fb
    best_sides = sides_fb
    
    def penalty_objective(params):
        centers = params[:n*2].reshape(n, 2)
        sides = params[n*2:]
        
        pen = 0.0
        tol = 1e-7
        
        # Bounds
        # cx - s/2 >= -tol => cx >= s/2 - tol
        # cx + s/2 <= 1 + tol => cx <= 1 + tol - s/2
        # s > 0
        
        # Vectorized bounds penalty
        left_viol = np.maximum(0, -tol - (centers[:,0] - sides/2))
        right_viol = np.maximum(0, (centers[:,0] + sides/2) - (1 + tol))
        bottom_viol = np.maximum(0, -tol - (centers[:,1] - sides/2))
        top_viol = np.maximum(0, (centers[:,1] + sides/2) - (1 + tol))
        side_viol = np.maximum(0, -sides)
        
        pen += np.sum(left_viol**2) * 1000
        pen += np.sum(right_viol**2) * 1000
        pen += np.sum(bottom_viol**2) * 1000
        pen += np.sum(top_viol**2) * 1000
        pen += np.sum(side_viol**2) * 1000
        
        # Overlaps
        # Compute pairwise distances
        # centers shape (n, 2)
        # dx[i,j] = |cx_i - cx_j|
        cx = centers[:, 0]
        cy = centers[:, 1]
        
        # Pairwise dx, dy
        # Using broadcasting
        # cx[:, None] - cx[None, :]
        dx = np.abs(cx[:, None] - cx[None, :])
        dy = np.abs(cy[:, None] - cy[None, :])
        
        # req_x[i,j] = (s_i + s_j)/2 - tol
        req_x = (sides[:, None] + sides[None, :]) / 2.0 - tol
        req_y = req_x # same formula
        
        # sep_x = dx - req_x
        # sep_y = dy - req_y
        sep_x = dx - req_x
        sep_y = dy - req_y
        
        # We want min(sep_x, sep_y) >= 0 for all i != j
        # Penalty for each pair: max(0, -min(sep_x, sep_y))^2
        
        # Soft min
        a = 10.0
        # Clip to avoid overflow
        sep_x_clamped = np.clip(sep_x, -10, 10)
        sep_y_clamped = np.clip(sep_y, -10, 10)
        
        # soft_min matrix
        # log(exp(-a*x) + exp(-a*y))
        # Use stable log-sum-exp
        # max_val = np.maximum(sep_x_clamped, sep_y_clamped)
        # log_sum_exp = max_val + np.log(np.exp(-a*(sep_x_clamped - max_val)) + np.exp(-a*(sep_y_clamped - max_val)))
        # soft_min = -1/a * log_sum_exp
        
        # Simpler:
        # min_val = np.minimum(sep_x_clamped, sep_y_clamped)
        # diff = np.abs(sep_x_clamped - sep_y_clamped)
        # soft_min = min_val - (1.0/a) * np.log(1.0 + np.exp(-a * diff))
        
        min_val = np.minimum(sep_x_clamped, sep_y_clamped)
        diff = np.abs(sep_x_clamped - sep_y_clamped)
        soft_min = min_val - (1.0/a) * np.log(1.0 + np.exp(-a * diff))
        
        # Penalty where soft_min < 0
        # We only care about i != j
        # Diagonal is 0, so soft_min(0,0) = 0.
        # We can just sum the whole matrix and divide by 2?
        # Or mask diagonal.
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        
        # Penalty
        viol = np.maximum(0, -soft_min)
        pen += np.sum(viol[mask]**2) * 1000
        
        sum_sides = np.sum(sides)
        return pen - sum_sides

    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-6, 1.0))
        
    x0 = np.concatenate([centers_fb.flatten(), sides_fb])
    
    start_time = time.time()
    deadline = start_time + 300 # 300s for optimization
    
    try:
        res = minimize(penalty_objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if time.time() < deadline:
            x_opt = res.x
            centers = x_opt[:n*2].reshape(n, 2)
            sides = x_opt[n*2:]
            
            # Verify
            valid = True
            tol = 1e-7
            for i in range(n):
                cx, cy = centers[i]
                s_i = sides[i]
                if cx - s_i/2 < -tol or cx + s_i/2 > 1 + tol:
                    valid = False
                if cy - s_i/2 < -tol or cy + s_i/2 > 1 + tol:
                    valid = False
                if s_i <= 0:
                    valid = False
            for i in range(n):
                for j in range(i+1, n):
                    dx = abs(centers[i,0] - centers[j,0])
                    dy = abs(centers[i,1] - centers[j,1])
                    req = (sides[i] + sides[j]) / 2.0 - tol
                    if dx < req and dy < req:
                        valid = False
                        
            if valid:
                val = np.sum(sides)
                if val > best_val:
                    best_val = val
                    best_centers = centers
                    best_sides = sides
    except Exception:
        pass
        
    return best_centers, best_sides
