"""Circle packing in perimeter-4 rectangle: Gradient-Based Optimization with Penalty Method."""
import numpy as np
import time
import scipy.optimize

N = 21

def _compute_penalty(params, W, H, mu=100.0):
    """
    Compute the penalty for a given configuration.
    params: array of shape (3N,). First N are x, next N are y, last N are r.
    W, H: dimensions of the rectangle.
    Returns: penalty value.
    """
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:]
    
    # Ensure radii are positive
    r = np.maximum(r, 0.0)
    
    penalty = 0.0
    
    # Wall violations
    # x_i must be >= r_i and <= W - r_i
    # y_i must be >= r_i and <= H - r_i
    wall_viol_x = np.maximum(0.0, r - x) + np.maximum(0.0, r - (W - x))
    wall_viol_y = np.maximum(0.0, r - y) + np.maximum(0.0, r - (H - y))
    wall_pen = np.sum(wall_viol_x**2) + np.sum(wall_viol_y**2)
    penalty += mu * wall_pen
    
    # Circle-circle violations
    # d_ij >= r_i + r_j
    # violation = max(0, r_i + r_j - d_ij)
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            viol = r[i] + r[j] - d
            if viol > 0:
                penalty += mu * viol**2
                
    # We want to MAXIMIZE sum(r). 
    # So we minimize -sum(r) + penalties.
    # However, if we just minimize -sum(r), it will push r to infinity unless constrained by penalties.
    # The penalty mu * viol^2 grows quadratically, while -sum(r) grows linearly.
    # So for large r, penalty dominates. This should find a local maximum of sum(r) subject to feasibility.
    
    objective_val = -np.sum(r) + penalty
    return objective_val

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_value = 0.0
    best_params = None
    best_w = 1.0
    
    rng = np.random.default_rng(42)
    
    # Try a few aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2]
    
    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.3 or w0 < 0.3:
            continue
            
        # Multi-start for this aspect ratio
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize
            # Grid initialization
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            
            for i in range(N):
                row, col = divmod(i, side)
                x[i] = (col + 0.5) / side * w0
                y[i] = (row + 0.5) / side * h0
                # Start with small radii to allow movement
                r[i] = 0.05
                
            # Add some randomness to escape symmetric local minima
            if restart > 0:
                x += rng.normal(0, 0.05, N)
                y += rng.normal(0, 0.05, N)
                # Keep within bounds
                x = np.clip(x, 0.05, w0 - 0.05)
                y = np.clip(y, 0.05, h0 - 0.05)
                
            params = np.concatenate([x, y, r])
            
            # Bounds
            # x: [0, w0], y: [0, h0], r: [0, min(w0,h0)/2]
            bounds = []
            for i in range(N):
                bounds.append((0.0, w0)) # x
            for i in range(N):
                bounds.append((0.0, h0)) # y
            for i in range(N):
                bounds.append((0.0, min(w0, h0)/2.0)) # r
                
            # Objective function
            def obj(p):
                return _compute_penalty(p, w0, h0, mu=50.0)
                
            # Run optimization
            try:
                res = scipy.optimize.minimize(
                    obj, 
                    params, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-8, 'gtol': 1e-5}
                )
                
                if res.success or res.fun < 1e-4:
                    final_params = res.x
                    # Extract
                    fx = final_params[:N]
                    fy = final_params[N:2*N]
                    fr = final_params[2*N:]
                    
                    # Check validity strictly
                    # If penalty is low, it should be valid.
                    # Let's verify
                    valid = True
                    for i in range(N):
                        if fx[i] < fr[i] - 1e-6 or fx[i] > w0 - fr[i] + 1e-6:
                            valid = False
                            break
                        if fy[i] < fr[i] - 1e-6 or fy[i] > h0 - fr[i] + 1e-6:
                            valid = False
                            break
                    if valid:
                        for i in range(N):
                            for j in range(i+1, N):
                                dx = fx[i] - fx[j]
                                dy = fy[i] - fy[j]
                                d = np.sqrt(dx*dx + dy*dy)
                                if d < fr[i] + fr[j] - 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                                
                    if valid:
                        val = np.sum(fr)
                        if val > best_value:
                            best_value = val
                            best_params = final_params.copy()
                            best_w = w0
            except Exception as e:
                pass

    # Fallback if no valid result found
    if best_params is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i+0.5)/side, (j+0.5)/side, 0.4/side)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Return best result
    x = best_params[:N]
    y = best_params[N:2*N]
    r = best_params[2*N:]
    
    return np.column_stack((x, y, r))
