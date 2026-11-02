import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def get_dims(t):
        t = np.clip(t, 0.15, 0.85)
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        return w, h

    def compute_penalty(t, circles, mu=10.0):
        """
        Computes the penalty for the given configuration.
        Penalty = sum of squared overlaps + sum of squared boundary violations.
        """
        w, h = get_dims(t)
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        
        # Boundary violations
        viol_left = np.maximum(0, r - x)
        viol_right = np.maximum(0, x - (w - r))
        viol_bottom = np.maximum(0, r - y)
        viol_top = np.maximum(0, y - (h - r))
        
        boundary_penalty = np.sum(viol_left**2) + np.sum(viol_right**2) + np.sum(viol_bottom**2) + np.sum(viol_top**2)
        
        # Overlaps
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                sum_r = r[i] + r[j]
                if dist_sq < sum_r**2:
                    dist = np.sqrt(dist_sq)
                    overlap = sum_r - dist
                    overlap_penalty += overlap**2
        
        return boundary_penalty + overlap_penalty

    def evaluate(t, circles):
        """
        Returns (score, penalty, is_valid)
        """
        penalty = compute_penalty(t, circles)
        score = np.sum(circles[:, 2])
        is_valid = penalty < 1e-7
        return score, penalty, is_valid

    def objective(vars):
        """
        Objective function for scipy minimize.
        vars: [t, x_0...x_N-1, y_0...y_N-1, r_0...r_N-1]
        Minimize penalty - mu * sum(r)
        """
        t = vars[0]
        circles = vars[1:].reshape(N, 3)
        
        # Ensure radii are positive
        circles[:, 2] = np.maximum(0.01, circles[:, 2])
        
        penalty = compute_penalty(t, circles)
        score = np.sum(circles[:, 2])
        
        return penalty - 10.0 * score

    def initial_grid(t):
        w, h = get_dims(t)
        circles = np.zeros((N, 3))
        # Use a 5x5 grid, fill 21
        side = 5
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                cx = (i + 0.5) * (w / side)
                cy = (j + 0.5) * (h / side)
                r = 0.45 * min(w, h) / side
                circles[count, 0] = cx
                circles[count, 1] = cy
                circles[count, 2] = r
                count += 1
        return circles

    best_score = -np.inf
    best_t = 0.5
    best_circles = None
    
    # Try multiple aspect ratios
    t_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    random.seed(42)
    np.random.seed(42)
    
    for t in t_values:
        if time.time() > deadline - 10:
            break
            
        circles = initial_grid(t)
        
        # Convert to flat array for scipy
        # vars = [t, x0, y0, r0, x1, y1, r1, ...]
        init_vars = np.concatenate(([t], circles.flatten()))
        
        # Optimization
        try:
            result = minimize(
                objective,
                init_vars,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 5000}
            )
        except Exception:
            continue
        
        opt_vars = result.x
        opt_t = opt_vars[0]
        opt_circles = opt_vars[1:].reshape(N, 3)
        opt_circles[:, 2] = np.maximum(0.01, opt_circles[:, 2])
        
        score, penalty, is_valid = evaluate(opt_t, opt_circles)
        
        if is_valid and score > best_score:
            best_score = score
            best_t = opt_t
            best_circles = opt_circles.copy()
            
        # Also try a local polish with SA if we have time
        # But L-BFGS-B is usually strong enough for this.
        
    # If we didn't find a valid solution, or score is low, try a fallback
    if best_circles is None or best_score < 1.0:
        # Fallback to initial grid at t=0.5
        t = 0.5
        circles = initial_grid(t)
        score, penalty, is_valid = evaluate(t, circles)
        if is_valid:
            best_score = score
            best_t = t
            best_circles = circles

    return best_circles
