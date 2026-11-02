import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer for 120s limit
    
    best_score = -np.inf
    best_result = None
    
    # Function to calculate penalty and objective
    def objective(p):
        # p shape: (N, 3) -> [x, y, r]
        x = p[:, 0]
        y = p[:, 1]
        r = p[:, 2]
        
        # Sum of radii to maximize -> minimize negative
        obj = -np.sum(r)
        
        # Penalty for boundary violations
        # r <= x, r <= y, r <= 1-x, r <= 1-y
        pen_bound = 0.0
        # Vectorized penalty calculation
        # dist to left: x, to right: 1-x, to bottom: y, to top: 1-y
        d_left = x
        d_right = 1 - x
        d_bottom = y
        d_top = 1 - y
        
        # Violation is max(0, r - dist)
        v_left = np.maximum(0, r - d_left)
        v_right = np.maximum(0, r - d_right)
        v_bottom = np.maximum(0, r - d_bottom)
        v_top = np.maximum(0, r - d_top)
        
        pen_bound = np.sum(v_left**2) + np.sum(v_right**2) + np.sum(v_bottom**2) + np.sum(v_top**2)
        
        # Penalty for overlaps
        # d_ij >= r_i + r_j
        # Calculate pairwise distances
        # Using broadcasting for efficiency
        # x shape (N,), x[:, None] shape (N, 1), x[None, :] shape (1, N)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        
        r_sum = r[:, None] + r[None, :]
        
        # We only care about i < j, but summing all and dividing by 2 or just summing upper triangle is fine.
        # d[i,i] = 0, r_sum[i,i] = 2*r_i. max(0, 2r_i - 0) is large. We must exclude diagonal.
        # Let's zero out the diagonal before calculation or mask it.
        mask = ~np.eye(N, dtype=bool)
        
        # Violation is max(0, r_sum - d)
        # Where d < r_sum
        v_overlap = np.maximum(0, r_sum - d)
        # Zero out diagonal
        v_overlap[~mask] = 0
        
        pen_overlap = np.sum(v_overlap**2)
        
        return obj + 100.0 * (pen_bound + pen_overlap)

    def is_valid(p):
        x = p[:, 0]
        y = p[:, 1]
        r = p[:, 2]
        
        # Check bounds
        if np.any(x < 0) or np.any(y < 0) or np.any(x > 1) or np.any(y > 1):
            return False
        if np.any(r < 0):
            return False
        
        # Check wall constraints
        if np.any(r > x + 1e-9) or np.any(r > y + 1e-9) or np.any(r > 1 - x + 1e-9) or np.any(r > 1 - y + 1e-9):
            return False
            
        # Check overlap
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, None] + r[None, :]
        
        # Check if any distance is less than sum of radii
        # Exclude diagonal
        mask = ~np.eye(N, dtype=bool)
        if np.any(d[mask] < r_sum[mask] - 1e-9):
            return False
            
        return True

    def score(p):
        return np.sum(p[:, 2])

    # Initializations
    # 1. Grid
    side = int(np.ceil(np.sqrt(N)))
    grid_coords = []
    for i in range(side):
        for j in range(side):
            grid_coords.append(((i + 0.5) / side, (j + 0.5) / side))
    
    # 2. Random
    np.random.seed(42) # For reproducibility in restarts if needed, but we vary init
    
    restarts = 0
    while time.time() - start_time < time_limit:
        restarts += 1
        
        # Choose initialization strategy
        if restarts % 2 == 1:
            # Grid based with jitter
            init_coords = np.array(grid_coords[:N])
            init_r = np.full(N, 0.5 / side * 0.9) # Slightly smaller to allow movement
            # Jitter
            init_coords += np.random.normal(0, 0.05, init_coords.shape)
            # Clip to [0.1, 0.9] to keep away from walls initially? No, let optimizer handle it, but keep positive.
            init_coords = np.clip(init_coords, 0.1, 0.9)
        else:
            # Random placement
            init_coords = np.random.uniform(0.1, 0.9, (N, 2))
            init_r = np.full(N, 0.05) # Start small
            
        p0 = np.hstack([init_coords, init_r.reshape(-1, 1)])
        
        # Bounds for L-BFGS-B
        # x, y in [0, 1], r in [0, 0.5]
        bounds = [(0, 1), (0, 1), (0, 0.5)] * N
        
        try:
            res = minimize(objective, p0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            p_opt = res.x
            
            # Check if valid
            if is_valid(p_opt):
                s = score(p_opt)
                if s > best_score:
                    best_score = s
                    best_result = p_opt.copy()
                    
        except Exception:
            pass
            
        # If we have a very good score, we can break early?
        # Theoretical max for 26 circles is roughly sqrt(26)/2 * something? 
        # For 1 circle, sum=0.5. For 4, sum=1.0. For 9, sum=1.5.
        # For 26, maybe around 1.3 - 1.4?
        # If we get > 1.35, it's likely near optimal.
        if best_score > 1.35:
             break

    if best_result is None:
        # Fallback to seed if optimization failed completely
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
