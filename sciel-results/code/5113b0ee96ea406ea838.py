import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_result = None
    best_value = -1
    
    # We will try a few aspect ratios W x H where 2(W+H) = 4 => W+H = 2
    # Common aspect ratios for packing
    aspect_ratios = [1.0, 1.2, 1.5, 2.0] # W/H ratios
    
    for ratio in aspect_ratios:
        if time.time() > deadline:
            break
            
        # Calculate W and H
        # W/H = ratio => W = ratio * H
        # 2(W+H) = 4 => W+H = 2
        # ratio*H + H = 2 => H(1+ratio) = 2 => H = 2 / (1+ratio)
        H = 2.0 / (1.0 + ratio)
        W = 2.0 - H
        
        # Objective function: Minimize -sum(r) subject to constraints
        # We optimize positions (x, y) and a single radius scale factor 's'
        # All circles have radius r = s * base_r
        # To simplify, let's just optimize positions and a common radius r directly?
        # No, if we optimize 21 positions + 1 radius, that's 43 variables.
        # Penalty function:
        # Penalty = sum(max(0, (r_i + r_j - dist_ij)^2)) + sum(max(0, (r_i - x_i)^2)) + ...
        
        def objective(params):
            # params: [x_1, y_1, ..., x_N, y_N, r]
            # But if we use a single r, we just need to check if it fits.
            # Actually, it's better to allow different radii? 
            # The task says "maximize the sum of their radii". 
            # Usually, in these packing problems with equal area constraints or just "pack N circles", 
            # the optimal solution often has equal radii or near-equal. 
            # However, allowing different radii increases the search space significantly.
            # Given the "disjoint interiors" and "maximize sum of radii", 
            # and the fact that the seed likely used equal radii, let's stick to equal radii first for robustness.
            # If equal radii, we optimize 21*2 positions + 1 radius = 43 variables.
            
            x = params[:N]
            y = params[N:2*N]
            r = params[2*N]
            
            if r <= 0:
                return 1e10
                
            penalty = 0.0
            
            # Overlap penalty
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = 2.0 * r
                    if dist < min_dist:
                        penalty += (min_dist - dist)**2
            
            # Boundary penalty
            for i in range(N):
                if x[i] < r:
                    penalty += (r - x[i])**2
                elif x[i] > W - r:
                    penalty += (x[i] - (W - r))**2
                
                if y[i] < r:
                    penalty += (r - y[i])**2
                elif y[i] > H - r:
                    penalty += (y[i] - (H - r))**2
            
            # We want to maximize sum(r) = N*r. So minimize -N*r.
            # But we must ensure penalty is 0.
            # If penalty > 0, we return a large value.
            # A common trick: return penalty * large_constant - sum(r)
            # If penalty is 0, it minimizes -sum(r) (maximizes sum).
            # If penalty > 0, the large positive term dominates.
            
            return 100.0 * penalty - N * r

        # Initial guesses
        # 1. Grid initialization
        # 2. Random initialization
        
        initial_guesses = []
        
        # Grid Init
        # Approximate grid size
        # Try a few grid configurations
        grid_configs = [
            (4, 6, 7, 4), # 4+6+7+4 = 21
            (5, 5, 5, 5, 1), # 21
            (3, 7, 7, 4), # 21
        ]
        
        for config in grid_configs:
            rows = len(config)
            x_init = np.zeros(N)
            y_init = np.zeros(N)
            idx = 0
            # Scale to fit
            # Just place them evenly in the box
            for i in range(rows):
                count = config[i]
                for j in range(count):
                    x_init[idx] = (j + 0.5) * (W / count)
                    y_init[idx] = (i + 0.5) * (H / rows)
                    idx += 1
            
            # Initial radius guess
            # Area of box = W*H. Area of N circles = N*pi*r^2.
            # Max r approx sqrt(W*H / (N*pi)) * packing_efficiency
            r_guess = np.sqrt(W * H / (N * np.pi)) * 0.5
            r_init = np.array([r_guess])
            
            initial_guesses.append(np.concatenate([x_init, y_init, r_init]))
            
        # Random Inits
        np.random.seed(42)
        for _ in range(5):
            x_rand = np.random.uniform(0, W, N)
            y_rand = np.random.uniform(0, H, N)
            r_guess = np.sqrt(W * H / (N * np.pi)) * 0.3
            r_rand = np.array([r_guess])
            initial_guesses.append(np.concatenate([x_rand, y_rand, r_rand]))
        
        for x0 in initial_guesses:
            if time.time() > deadline:
                break
                
            try:
                # Use L-BFGS-B which is good for bound-constrained problems
                # Bounds for x: [0, W], y: [0, H], r: [0, min(W,H)/2]
                bounds = []
                for i in range(N):
                    bounds.append((0, W))
                    bounds.append((0, H))
                bounds.append((1e-6, min(W, H)/2))
                
                res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 200, 'ftol': 1e-10})
                
                if res.success or (res.fun < 1e-5): # Check if penalty is effectively 0
                    # Verify solution
                    x_opt = res.x[:N]
                    y_opt = res.x[N:2*N]
                    r_opt = res.x[2*N]
                    
                    valid = True
                    # Check overlaps
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                            if dist < 2.0 * r_opt - 1e-8:
                                valid = False
                                break
                        if not valid: break
                    if not valid:
                        continue
                    
                    # Check bounds
                    for i in range(N):
                        if x_opt[i] < r_opt - 1e-8 or x_opt[i] > W - r_opt + 1e-8:
                            valid = False
                            break
                        if y_opt[i] < r_opt - 1e-8 or y_opt[i] > H - r_opt + 1e-8:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    value = N * r_opt
                    if value > best_value:
                        best_value = value
                        result_array = np.column_stack((x_opt, y_opt, np.full(N, r_opt)))
                        best_result = result_array
                        
            except Exception as e:
                # Pass
                pass

    if best_result is None:
        # Fallback: Simple grid in a square 1x1 box (Perimeter 4)
        # 5x5 grid, take 21
        side = 5
        x = []
        y = []
        r = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x.append((j + 0.5) / side)
                y.append((i + 0.5) / side)
                r.append(0.5 / side - 1e-5)
                count += 1
            if count >= N:
                break
        return np.array([x, y, r]).T
        
    return best_result
