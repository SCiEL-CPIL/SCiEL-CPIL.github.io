import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    best_sum_r = -np.inf
    best_config = None
    
    # Aspect ratios to try
    # Perimeter = 2(w+h) = 4 => w+h=2
    # w in [1, 2] is symmetric to h in [1, 2]
    # Let's try w in [0.5, 1.5]
    widths = np.linspace(0.5, 1.5, 11)
    
    def penalty_func(params, w, h, R):
        """
        Calculate penalty for a configuration.
        params: flat array of x, y coordinates (length 2*N)
        """
        x = params[:N]
        y = params[N:]
        
        # Boundary penalties
        # x_i must be in [R, w-R]
        # y_i must be in [R, h-R]
        pen_x = np.sum(np.maximum(0, R - x)**2) + np.sum(np.maximum(0, x - (w - R))**2)
        pen_y = np.sum(np.maximum(0, R - y)**2) + np.sum(np.maximum(0, y - (h - R))**2)
        
        # Overlap penalties
        # dist_ij < 2R
        # We want to penalize (2R - dist_ij)^2 if dist_ij < 2R
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx*dx + dy*dy
        # Avoid sqrt if possible, but we need dist for the condition
        # Actually, max(0, 2R - dist)^2 is smooth enough for Nelder-Mead?
        # dist = np.sqrt(dist_sq)
        # overlap = np.maximum(0, 2*R - dist)
        # pen_overlap = np.sum(overlap**2)
        
        # To be faster, we can approximate or just use sqrt. N=21 is small.
        dist = np.sqrt(dist_sq)
        # Mask for i < j to avoid double counting? 
        # The penalty sum over all i,j counts each pair twice. 
        # It's fine for optimization, just scales the penalty.
        # But self-dist is 0. 
        # We should ignore i=j.
        # dist[i,i] = 0. 2R - 0 = 2R. Penalty would be huge.
        # So we must mask diagonal.
        mask = ~np.eye(N, dtype=bool)
        overlap = np.where(mask, np.maximum(0, 2*R - dist), 0)
        pen_overlap = np.sum(overlap**2)
        
        return pen_x + pen_y + pen_overlap

    for w in widths:
        h = 2.0 - w
        
        # Max possible R is min(w, h) / 2
        R_max = min(w, h) / 2.0
        R_min = 0.01
        
        # Binary search for max R
        low = R_min
        high = R_max
        best_R_for_w = 0.0
        best_params_for_w = None
        
        # We want to find the largest R such that penalty can be minimized to ~0.
        # Let's do a coarse search first, then refine.
        # Or just binary search with a few iterations.
        
        # To speed up, we can try to find a feasible R by decreasing from R_max.
        # But binary search is safer.
        
        for _ in range(15): # 15 steps for binary search
            mid = (low + high) / 2.0
            
            # Initial guess: Grid
            side = 5
            # Create a 5x5 grid, take first 21
            xs_init = []
            ys_init = []
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    # Center of cell
                    x = (i + 0.5) * (w / side)
                    y = (j + 0.5) * (h / side)
                    # Clip to valid range [R, w-R]
                    x = np.clip(x, mid, w - mid)
                    y = np.clip(y, mid, h - mid)
                    xs_init.append(x)
                    ys_init.append(y)
                    idx += 1
            
            x0 = np.array(xs_init + ys_init)
            
            # Optimize
            # Nelder-Mead is robust for non-differentiable functions
            res = minimize(penalty_func, x0, args=(w, h, mid), method='Nelder-Mead', 
                           options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 200, 'maxfev': 200})
            
            if res.fun < 1e-5:
                # Feasible
                low = mid
                best_R_for_w = mid
                best_params_for_w = res.x
            else:
                high = mid
                
        # After binary search, best_R_for_w is the largest feasible R found
        # We should use the last successful configuration
        if best_params_for_w is not None:
            x_final = best_params_for_w[:N]
            y_final = best_params_for_w[N:]
            r_final = best_R_for_w
            
            # Verify validity strictly
            valid = True
            for i in range(N):
                if x_final[i] < r_final - 1e-6 or x_final[i] > w - r_final + 1e-6:
                    valid = False
                    break
                if y_final[i] < r_final - 1e-6 or y_final[i] > h - r_final + 1e-6:
                    valid = False
                    break
            
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x_final[i] - x_final[j]
                        dy = y_final[i] - y_final[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < 2*r_final - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                current_sum = N * r_final
                if current_sum > best_sum_r:
                    best_sum_r = current_sum
                    best_config = np.column_stack((x_final, y_final, np.full(N, r_final)))

    if best_config is None:
        # Fallback
        side = 5
        cells = []
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append([x, y, r])
                idx += 1
        return np.array(cells)
    
    return best_config
