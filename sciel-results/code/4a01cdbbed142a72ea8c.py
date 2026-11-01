import numpy as np
from scipy.optimize import minimize
import time

N = 21
START_TIME = time.time()
TIME_LIMIT = 140.0  # Leave 10s margin for 150s limit

def get_penalty(params, w, h):
    """
    params: array of shape (N*2,) for positions or (N*3,) for pos+rad
    w, h: dimensions of the box
    Returns: penalty value
    """
    n = len(params) // 2
    if params.shape[0] == n * 2:
        xy = params.reshape(n, 2)
        # Radii are not in params, this function is for position-only optimization
        # We need radii passed separately or this function is incomplete
        pass
    # This design is getting complicated. Let's make two specific cost functions.
    return 0.0

def construct():
    start = time.time()
    
    # Constants
    n = N
    w = 1.0
    h = 1.0
    
    # Initial guess: Grid
    side = int(np.ceil(np.sqrt(n)))
    # Create a grid of n points
    xs = []
    ys = []
    for i in range(side):
        for j in range(side):
            if len(xs) < n:
                xs.append((i + 0.5) / side)
                ys.append((j + 0.5) / side)
    
    # Initial radius
    r_init = 0.01
    
    # Function to calculate penalty for given positions and radii
    def calc_penalty(xy, radii, w, h):
        # xy: (n, 2)
        # radii: (n,)
        x = xy[:, 0]
        y = xy[:, 1]
        
        # Boundary penalty
        # x in [r, w-r], y in [r, h-r]
        pen_bound = 0.0
        for i in range(n):
            r = radii[i]
            if x[i] < r:
                pen_bound += (r - x[i])**2
            elif x[i] > w - r:
                pen_bound += (x[i] - (w - r))**2
            
            if y[i] < r:
                pen_bound += (r - y[i])**2
            elif y[i] > h - r:
                pen_bound += (y[i] - (h - r))**2
        
        # Overlap penalty
        pen_overlap = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    pen_overlap += (min_dist - dist)**2
        
        return pen_bound + pen_overlap

    # Optimization 1: Fix radii, optimize positions
    # We will iteratively increase radii
    
    # Initial positions
    xy = np.array(list(zip(xs, ys)))
    
    # We need to optimize positions to minimize penalty for a given radius vector
    def pos_cost(params):
        xy_curr = params.reshape(n, 2)
        # Radii are fixed in this closure
        return calc_penalty(xy_curr, current_radii, w, h)
    
    best_xy = xy.copy()
    current_radii = np.full(n, r_init)
    
    # Iterative expansion
    scale_factor = 1.05
    max_iter = 50
    tol = 1e-6
    
    for k in range(max_iter):
        if time.time() - start > TIME_LIMIT - 20:
            break
            
        # Optimize positions for current radii
        # Use Nelder-Mead
        res = minimize(pos_cost, best_xy.flatten(), method='Nelder-Mead', 
                       options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 1000})
        
        if res.success:
            best_xy = res.x.reshape(n, 2)
            
        # Check penalty
        pen = calc_penalty(best_xy, current_radii, w, h)
        
        # If penalty is small, scale up radii
        if pen < 1e-4:
            current_radii *= scale_factor
            # Optional: Add small random noise to help escape local minima
            # best_xy += np.random.normal(0, 0.001, best_xy.shape)
        else:
            # If penalty is high, try to reduce it further or stop
            # If we can't reduce it, we are stuck. 
            # Try a different restart or break.
            # For simplicity, if we fail to reduce penalty after scaling, we stop.
            # But we should try to minimize penalty first.
            # The minimize call above already tried.
            # If penalty is still high, increasing radii will make it worse.
            # So we stop expanding.
            break
            
    # Final Polish: Optimize positions AND radii
    # Objective: Maximize sum(radii) - Penalty * Large_Multiplier
    # Minimize: -sum(radii) + M * Penalty
    
    def full_cost(params):
        xy_curr = params[:n*2].reshape(n, 2)
        r_curr = params[n*2:]
        
        # Constrain radii to be positive
        r_curr = np.maximum(r_curr, 0.001)
        
        pen = calc_penalty(xy_curr, r_curr, w, h)
        sum_r = np.sum(r_curr)
        
        # M should be large enough to dominate the sum_r term if there is any overlap
        # Sum r is at most 21 * 0.5 = 10.5. Penalty is squared distance.
        # If overlap is 0.1, penalty is 0.01. We want 0.01 * M > 10.5 => M > 1000.
        M = 1000.0
        return -sum_r + M * pen

    # Initialize full params
    init_params = np.concatenate([best_xy.flatten(), current_radii])
    
    # Run final optimization
    # L-BFGS-B might be better here if we smooth the penalty, but Nelder-Mead is robust.
    # Let's try L-BFGS-B with bounds.
    # Bounds: x in [0, w], y in [0, h], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, w))
        bounds.append((0.0, h))
        bounds.append((0.0, 0.5))
        
    # Nelder-Mead doesn't support bounds well. 
    # Let's stick to Nelder-Mead and handle bounds via penalty (which we already do).
    # Or use COBYLA?
    # Let's try Nelder-Mead again for the full problem.
    
    res_full = minimize(full_cost, init_params, method='Nelder-Mead', 
                        options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 2000})
    
    if res_full.success:
        final_params = res_full.x
    else:
        final_params = init_params
        
    final_xy = final_params[:n*2].reshape(n, 2)
    final_radii = final_params[n*2:]
    
    # Ensure validity
    # Check penalty
    final_pen = calc_penalty(final_xy, final_radii, w, h)
    
    # If invalid, shrink radii slightly
    if final_pen > 1e-5:
        # Binary search for scale factor that makes it valid?
        # Or just reject and return previous valid state?
        # Let's try to shrink.
        scale = 1.0
        for _ in range(20):
            test_radii = final_radii * scale
            test_pen = calc_penalty(final_xy, test_radii, w, h)
            if test_pen < 1e-5:
                final_radii = test_radii
                break
            scale *= 0.95
        # Recalculate final value
        final_pen = calc_penalty(final_xy, final_radii, w, h)
        
    # Construct output array (n, 3)
    result = np.column_stack((final_xy[:, 0], final_xy[:, 1], final_radii))
    
    return result

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 21
START_TIME = time.time()
TIME_LIMIT = 140.0  # Leave 10s margin for 150s limit

def construct():
    start = time.time()
    
    # Constants
    n = N
    w = 1.0
    h = 1.0
    
    # Initial guess: Grid
    side = int(np.ceil(np.sqrt(n)))
    xs = []
    ys = []
    for i in range(side):
        for j in range(side):
            if len(xs) < n:
                xs.append((i + 0.5) / side)
                ys.append((j + 0.5) / side)
    
    # Initial radius
    r_init = 0.01
    
    # Function to calculate penalty for given positions and radii
    def calc_penalty(xy, radii, w, h):
        x = xy[:, 0]
        y = xy[:, 1]
        
        # Boundary penalty
        pen_bound = 0.0
        for i in range(n):
            r = radii[i]
            if x[i] < r:
                pen_bound += (r - x[i])**2
            elif x[i] > w - r:
                pen_bound += (x[i] - (w - r))**2
            
            if y[i] < r:
                pen_bound += (r - y[i])**2
            elif y[i] > h - r:
                pen_bound += (y[i] - (h - r))**2
        
        # Overlap penalty
        pen_overlap = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    pen_overlap += (min_dist - dist)**2
        
        return pen_bound + pen_overlap

    # Initial positions
    xy = np.array(list(zip(xs, ys)))
    
    # We need to optimize positions to minimize penalty for a given radius vector
    def pos_cost(params):
        xy_curr = params.reshape(n, 2)
        return calc_penalty(xy_curr, current_radii, w, h)
    
    best_xy = xy.copy()
    current_radii = np.full(n, r_init)
    
    # Iterative expansion
    scale_factor = 1.02
    max_iter = 100
    
    for k in range(max_iter):
        if time.time() - start > TIME_LIMIT - 30:
            break
            
        # Optimize positions for current radii
        res = minimize(pos_cost, best_xy.flatten(), method='Nelder-Mead', 
                       options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 500})
        
        if res.success:
            best_xy = res.x.reshape(n, 2)
            
        # Check penalty
        pen = calc_penalty(best_xy, current_radii, w, h)
        
        # If penalty is small, scale up radii
        if pen < 1e-4:
            current_radii *= scale_factor
        else:
            # If penalty is high, we might be stuck.
            # Try a few more iterations with higher tolerance?
            # Or break.
            # Let's break to save time for final polish.
            break
            
    # Final Polish: Optimize positions AND radii
    def full_cost(params):
        xy_curr = params[:n*2].reshape(n, 2)
        r_curr = params[n*2:]
        
        # Constrain radii to be positive
        r_curr = np.maximum(r_curr, 0.001)
        
        pen = calc_penalty(xy_curr, r_curr, w, h)
        sum_r = np.sum(r_curr)
        
        M = 1000.0
        return -sum_r + M * pen

    # Initialize full params
    init_params = np.concatenate([best_xy.flatten(), current_radii])
    
    # Run final optimization
    res_full = minimize(full_cost, init_params, method='Nelder-Mead', 
                        options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 1000})
    
    if res_full.success:
        final_params = res_full.x
    else:
        final_params = init_params
        
    final_xy = final_params[:n*2].reshape(n, 2)
    final_radii = final_params[n*2:]
    
    # Ensure validity
    final_pen = calc_penalty(final_xy, final_radii, w, h)
    
    if final_pen > 1e-5:
        scale = 1.0
        for _ in range(20):
            test_radii = final_radii * scale
            test_pen = calc_penalty(final_xy, test_radii, w, h)
            if test_pen < 1e-5:
                final_radii = test_radii
                break
            scale *= 0.95
        final_pen = calc_penalty(final_xy, final_radii, w, h)
        
    result = np.column_stack((final_xy[:, 0], final_xy[:, 1], final_radii))
    
    return result
# EVOLVE-BLOCK-END
