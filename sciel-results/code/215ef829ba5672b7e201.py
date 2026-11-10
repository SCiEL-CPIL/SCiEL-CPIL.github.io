# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def moving_sofa_2d() -> np.ndarray:
    """
    Moves a sofa through an L-shaped corridor.
    Strategy: Directly optimize the (x, y) coordinates for each intermediate angle step (a=1..89).
    This removes the polynomial constraint of the current program, allowing the path to be
    any arbitrary curve. We use Nelder-Mead optimization on the 178-dimensional space of
    intermediate positions, with a fast vectorized evaluation function.
    """
    
    # 1. Setup Grid
    x_grid = np.arange(-3, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid)
    P = np.column_stack((X.ravel(), Y.ravel()))
    N_points = P.shape[0]
    wx = P[:, 0]
    wy = P[:, 1]
    
    # Angles for intermediate steps: 1 to 89
    angles = np.arange(1, 90, 1.0)
    n_steps = len(angles)
    n_params = 2 * n_steps  # x and y for each step
    
    # Pre-compute rotation matrices for each angle
    # R(-a) is used in the problem description: "R(-a)(w - (x, y))"
    # Wait, the problem says: "A point w is inside placement (x, y, a) if R(-a)(w - (x, y)) lies in the base"
    # So we rotate the point w relative to (x,y) by -a.
    # Let's pre-compute cos and sin for each angle.
    cos_a = np.cos(np.deg2rad(angles))  # shape (n_steps,)
    sin_a = np.sin(np.deg2rad(angles))  # shape (n_steps,)
    
    # For each step i, we have:
    # u = dx * cos(a_i) + dy * sin(a_i)
    # v = -dx * sin(a_i) + dy * cos(a_i)
    # where dx = wx - x_i, dy = wy - y_i
    
    def evaluate_area(params):
        """
        params: array of shape (178,), [x1, y1, x2, y2, ..., x89, y89]
        Returns the area (number of points * 0.0001).
        """
        # Reshape to (n_steps, 2)
        xy = params.reshape(n_steps, 2)
        x_path = xy[:, 0]
        y_path = xy[:, 1]
        
        mask = np.ones(N_points, dtype=bool)
        
        for i in range(n_steps):
            x_i = x_path[i]
            y_i = y_path[i]
            c = cos_a[i]
            s = sin_a[i]
            
            dx = wx - x_i
            dy = wy - y_i
            
            u = dx * c + dy * s
            v = -dx * s + dy * c
            
            in_r1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_r2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            
            in_corridor = in_r1 | in_r2
            
            mask &= in_corridor
            
            # Early exit
            if not np.any(mask):
                return 0.0
        
        return float(np.sum(mask)) * 0.0001

    # Initial guess
    # Based on the current best score ~1.27, the path is roughly:
    # x goes 0 -> -2, y goes 0 -> 0 with a bump.
    # Let's start with a simple curve.
    # x(t) = -2*t, y(t) = 0.5 * sin(pi*t) where t = a/90
    # But we only have intermediate steps a=1..89.
    # Let's use a sine curve for y and linear for x.
    
    t = angles / 90.0
    x_init = -2.0 * t
    y_init = 0.5 * np.sin(np.pi * t)
    
    # Flatten
    initial_params = np.column_stack((x_init, y_init)).ravel()
    
    # Optimize
    best_score = 0.0
    best_params = initial_params
    
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Number of restarts
    n_restarts = 10
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial guess
        if r == 0:
            params = initial_params
        else:
            # Random perturbation
            noise = np.random.normal(0, 0.1, n_params)
            params = initial_params + noise
        
        # Optimize
        # Nelder-Mead is robust for small dimensional problems, but 178 dims is large.
        # Let's try it anyway, with a small maxiter.
        try:
            res = scipy.optimize.minimize(
                lambda p: -evaluate_area(p), 
                params, 
                method='Nelder-Mead',
                options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x
        except Exception as e:
            pass
    
    # Convert best_params to the required output format
    # The evaluator expects a list of points.
    # We need to return the intermediate steps (a=1..89).
    
    xy_best = best_params.reshape(n_steps, 2)
    x_best = xy_best[:, 0]
    y_best = xy_best[:, 1]
    
    # Ensure the path is smooth? The problem doesn't require it, but it might help.
    # However, the evaluator interpolates between consecutive rows.
    # If we return a=1, 2, ..., 89, the evaluator will add (0,0,0) before and (-2,0,90) after.
    # The interpolation between (0,0,0) and (x1, y1, 1) is linear.
    # This is fine.
    
    # Create the result array
    result_path = np.column_stack((x_best, y_best, angles))
    
    # If result is empty (shouldn't happen), return a dummy
    if len(result_path) == 0:
        result_path = np.array([[-1.0, 0.0, 45.0]])
        
    return result_path

# EVOLVE-BLOCK-END
