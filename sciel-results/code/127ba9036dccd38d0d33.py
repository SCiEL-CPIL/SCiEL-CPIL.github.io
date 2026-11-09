"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    best_sum = 0.0
    best_config = None
    
    # Helper to check validity and get sum
    def get_score(config, W, H):
        # config: (N, 3) array of [x, y, r]
        if len(config) != N:
            return 0.0, False
        # Check boundaries
        for i in range(N):
            x, y, r = config[i]
            if r < 0:
                return 0.0, False
            if x < r - 1e-6 or x > W - r + 1e-6:
                return 0.0, False
            if y < r - 1e-6 or y > H - r + 1e-6:
                return 0.0, False
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                x1, y1, r1 = config[i]
                x2, y2, r2 = config[j]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dist < (r1 + r2) - 1e-6:
                    return 0.0, False
        
        s = np.sum(config[:, 2])
        return s, True

    # Smooth objective using log-sum-exp for soft constraints
    def smooth_objective(vars, W, H, beta=50.0):
        config = vars.reshape(N, 3)
        x, y, r = config[:, 0], config[:, 1], config[:, 2]
        
        # Sum of radii (to maximize)
        sum_r = np.sum(r)
        
        # Boundary violations (soft)
        pen_x_low = np.maximum(r - x, 0)
        pen_x_high = np.maximum(x - (W - r), 0)
        pen_y_low = np.maximum(r - y, 0)
        pen_y_high = np.maximum(y - (H - r), 0)
        pen_r_neg = np.maximum(-r, 0)
        
        # Overlap violations (soft)
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                overlap_pen += np.maximum(min_dist - dist, 0)
        
        # Total penalty
        total_pen = np.sum(pen_x_low) + np.sum(pen_x_high) + np.sum(pen_y_low) + np.sum(pen_y_high) + np.sum(pen_r_neg) + overlap_pen
        
        # Use smooth approximation: -sum_r + beta * total_pen
        # But make it smoother by using soft-plus
        def soft_plus(v, beta=50.0):
            return (1.0/beta) * np.logaddexp(0, beta * v)
        
        smooth_pen = soft_plus(total_pen, beta)
        
        return -sum_r + 10.0 * smooth_pen

    # Initialize grid
    def init_grid(W, H):
        # Try to find best N x M grid
        best_init = None
        best_init_score = -1
        
        for M in range(1, 8):
            for N_cols in range(1, 8):
                if N_cols * M < N:
                    continue
                # Aspect ratio match
                ar_rect = W / H
                ar_grid = N_cols / M
                if abs(ar_rect - ar_grid) > 1.5:
                    continue
                
                # Calculate max radius for this grid
                # W = N_cols * 2r, H = M * 2r
                r_w = W / (2 * N_cols)
                r_h = H / (2 * M)
                r = min(r_w, r_h)
                
                # Create config
                config = np.zeros((N, 3))
                idx = 0
                for j in range(M):
                    for i in range(N_cols):
                        if idx >= N:
                            break
                        x = r + i * 2 * r
                        y = r + j * 2 * r
                        config[idx] = [x, y, r]
                        idx += 1
                    if idx >= N:
                        break
                
                if idx < N:
                    # Not enough spots, try to squeeze in or skip this grid
                    continue
                
                # Score: sum of radii
                s = np.sum(config[:, 2])
                if s > best_init_score:
                    best_init_score = s
                    best_init = config.copy()
        
        if best_init is None:
            # Fallback: simple square grid
            side = int(np.ceil(np.sqrt(N)))
            r = min(W, H) / (2 * side) * 0.9
            config = np.zeros((N, 3))
            idx = 0
            for j in range(side):
                for i in range(side):
                    if idx >= N:
                        break
                    x = r + i * 2 * r
                    y = r + j * 2 * r
                    config[idx] = [x, y, r]
                    idx += 1
            best_init = config
        return best_init

    # Loop over aspect ratios
    W_values = np.linspace(0.5, 1.5, 20)
    
    for W in W_values:
        if time.time() - start_time > time_limit:
            break
            
        H = 2.0 - W
        if H <= 0:
            continue
        
        init_config = init_grid(W, H)
        x0 = init_config.flatten()
        
        # Run optimization with smooth objective
        try:
            # Phase 1: Coarse optimization
            res1 = minimize(smooth_objective, x0, args=(W, H, 50.0), method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
            
            config1 = res1.x.reshape(N, 3)
            config1[:, 2] = np.maximum(config1[:, 2], 0)
            
            # Phase 2: Fine optimization with higher beta
            if time.time() - start_time < time_limit - 5.0:
                res2 = minimize(smooth_objective, config1.flatten(), args=(W, H, 200.0), method='Nelder-Mead', 
                               options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-6})
                config = res2.x.reshape(N, 3)
            else:
                config = config1
            
            config[:, 2] = np.maximum(config[:, 2], 0)
            
            s, valid = get_score(config, W, H)
            if valid and s > best_sum:
                best_sum = s
                best_config = config.copy()
        except Exception:
            continue

    # Final polish: if we have a good config, try to improve it further
    if best_config is not None and time.time() - start_time < time_limit - 10.0:
        # Find the W,H that gave best_config
        # Re-run with tighter tolerance on the best aspect ratio
        for W in W_values:
            H = 2.0 - W
            if H <= 0:
                continue
            
            # Check if this W,H matches our best config roughly
            config_test = best_config.copy()
            s_test, valid_test = get_score(config_test, W, H)
            if valid_test and abs(s_test - best_sum) < 1e-3:
                # Polish this configuration
                try:
                    res_polish = minimize(smooth_objective, config_test.flatten(), args=(W, H, 500.0), 
                                         method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-7})
                    config_polished = res_polish.x.reshape(N, 3)
                    config_polished[:, 2] = np.maximum(config_polished[:, 2], 0)
                    
                    s_polish, valid_polish = get_score(config_polished, W, H)
                    if valid_polish and s_polish > best_sum:
                        best_sum = s_polish
                        best_config = config_polished.copy()
                except Exception:
                    pass
                break

    if best_config is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_config
# EVOLVE-BLOCK-END
