import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    time_limit = 110.0

    def objective_smooth(x, beta):
        centers = x.reshape(N, 2)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        x_coords = centers[:, 0]
        y_coords = centers[:, 1]
        
        dist_left = 0.5 + x_coords
        dist_right = 0.5 - x_coords
        dist_bottom = 0.5 + y_coords
        dist_top = 0.5 - y_coords
        
        all_dists = np.concatenate([dists.ravel(), dist_left, dist_right, dist_bottom, dist_top])
        
        # Filter out very small/negative distances? 
        # Distances should be positive if inside.
        # If a circle goes outside, dist can be negative. 
        # We want to penalize this heavily.
        # The log-sum-exp of negative numbers is fine, but we want to push them away.
        # Actually, if dist < 0, it's invalid. 
        # We can clamp dists to a small positive number to avoid log(0) or log(neg) issues?
        # No, if dist is negative, we want the objective to be very bad (high value to minimize? No, we minimize).
        # Let's just use the raw distances.
        
        # To handle the case where dist is 0 or negative, we add a small epsilon
        # But if dist is negative, -beta * dist is positive, exp is large, sum is large, -log is negative.
        # We are minimizing. So large positive exp -> large sum -> large -log (negative).
        # Wait. 
        # f = -1/beta * log(sum(exp(-beta * d)))
        # If d is large positive, exp(-beta*d) is small. f is close to 0 (or -large if we consider the min).
        # If d is small, exp(-beta*d) is large. f is more negative.
        # We want to MAXIMIZE d. So we MINIMIZE -d.
        # So minimizing f is correct.
        
        # However, numerical stability:
        # sum(exp(-beta * d)) can overflow if beta is large and d is negative.
        # Let's shift d by its minimum to prevent overflow?
        # Standard trick: logsumexp(v) = max(v) + log(sum(exp(v - max(v))))
        # Here v = -beta * d.
        
        v = -beta * all_dists
        v_max = np.max(v)
        # If v_max is very large positive, exp(v - v_max) is safe.
        # sum_exp = np.sum(np.exp(v - v_max))
        # log_sum_exp = v_max + np.log(sum_exp)
        # obj = -1/beta * log_sum_exp
        
        # If v_max is -inf (all dists inf?), handle it.
        if np.isinf(v_max):
             return 0.0
            
        # Clip v to avoid overflow in exp
        v_clipped = np.clip(v - v_max, -700, 700)
        sum_exp = np.sum(np.exp(v_clipped))
        if sum_exp == 0:
            return 0.0
        log_sum_exp = v_max + np.log(sum_exp)
        
        return -1.0 / beta * log_sum_exp

    def get_initial_guess():
        # Hexagonal grid
        rows = int(np.ceil(np.sqrt(N * np.sqrt(3) / 2)))
        cols = int(np.ceil(N / rows))
        points = []
        for r in range(rows):
            for c in range(cols):
                x = c * 1.0
                y = r * 0.866
                if r % 2 == 1:
                    x += 0.5
                points.append([x, y])
        
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        scale_x = 0.9 / (max_x - min_x + 1e-6)
        scale_y = 0.9 / (max_y - min_y + 1e-6)
        scale = min(scale_x, scale_y)
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        scaled_points = []
        for p in points:
            nx = (p[0] - center_x) * scale
            ny = (p[1] - center_y) * scale
            scaled_points.append([nx, ny])
            
        scaled_points = scaled_points[:N]
        noise = np.random.uniform(-0.01, 0.01, size=(N, 2))
        init = np.array(scaled_points) + noise
        init[:, 0] = np.clip(init[:, 0], -0.49, 0.49)
        init[:, 1] = np.clip(init[:, 1], -0.49, 0.49)
        return init.flatten()

    x = get_initial_guess()
    beta = 10.0
    
    while beta < 10000 and (time.time() - start_time) < time_limit:
        res = minimize(lambda x: objective_smooth(x, beta), x, method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        x = res.x
        beta *= 2.0
        
    # Final evaluation with raw min to ensure we return a valid, good point
    # We don't need to re-optimize, just return x
    
    return x.reshape(N, 2)
