import numpy as np
import time
def moving_sofa_2d() -> np.ndarray:
    # Grid definition
    x_grid = np.arange(-3.0, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = len(X_flat)
    
    # Precompute grid points as column vectors for broadcasting
    X_col = X_flat[:, np.newaxis] # (P, 1)
    Y_col = Y_flat[:, np.newaxis] # (P, 1)
    
    def evaluate_path_vectorized(path):
        # path is (n_ctrl, 3)
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        
        # Generate all placements
        placements = []
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            
            ts = np.linspace(0, 1, m+1)
            for t in ts:
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = p1[2] + t * (p2[2] - p1[2])
                placements.append((x, y, a))
        
        # Avoid duplicates at segment boundaries? 
        # The evaluator says "adds linear interpolations... in m equal steps".
        # Usually this implies inclusive endpoints.
        # If we have segment 1: t=0..1 and segment 2: t=0..1, the point at t=1 of seg1 is t=0 of seg2.
        # Duplicates don't change the boolean AND result, so it's safe to keep them.
        
        if not placements:
            return 0.0
            
        # Convert to numpy arrays
        # placements is a list of tuples (x, y, a)
        # We want arrays of shape (1, M) for broadcasting against (P, 1)
        px = np.array([p[0] for p in placements])[np.newaxis, :] # (1, M)
        py = np.array([p[1] for p in placements])[np.newaxis, :] # (1, M)
        pa = np.array([p[2] for p in placements])[np.newaxis, :] # (1, M)
        
        # Rotate grid points by -a
        theta = -np.deg2rad(pa) # (1, M)
        c = np.cos(theta) # (1, M)
        s = np.sin(theta) # (1, M)
        
        # w = (X, Y) - (x, y)
        wx = X_col - px # (P, M)
        wy = Y_col - py # (P, M)
        
        # u = wx * c - wy * s
        # v = wx * s + wy * c
        u = wx * c - wy * s # (P, M)
        v = wx * s + wy * c # (P, M)
        
        # Check conditions
        # R1: 0 <= u <= 1, -3 <= v <= 1
        in_r1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        # R2: -3 <= u <= 1, 0 <= v <= 1
        in_r2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        
        inside_any = in_r1 | in_r2 # (P, M)
        
        # A point is valid if it is inside ALL placements
        # i.e., all(inside_any[j, :]) for each j
        valid_points = np.all(inside_any, axis=1) # (P,)
        
        return np.sum(valid_points) * 0.0001

    # Optimization
    n_ctrl = 12
    ctrl_angles = np.linspace(5.0, 85.0, n_ctrl)
    
    def make_path_from_params(params):
        path = np.zeros((n_ctrl, 3))
        path[:, 2] = ctrl_angles
        path[:, 0] = params[0::2]
        path[:, 1] = params[1::2]
        return path

    # Initial guess
    current_params = np.zeros(2 * n_ctrl)
    for i in range(n_ctrl):
        t = ctrl_angles[i] / 90.0
        current_params[2*i] = -2.0 * t
        current_params[2*i+1] = 0.0
        
    # Bounds for differential evolution
    # x: -3 to 1, y: -2 to 2
    bounds = []
    for i in range(n_ctrl):
        bounds.append((-3.0, 1.0))
        bounds.append((-2.0, 2.0))
        
    def objective(params):
        path = make_path_from_params(params)
        return -evaluate_path_vectorized(path)
        
    import scipy.optimize as so
    
    start_time = time.time()
    time_limit = 580.0
    deadline = start_time + time_limit
    
    # Use differential evolution for global search
    # maxiter is the number of generations
    # popsize is the population size
    # We need to be careful with the time.
    # DE can be slow if the population is large.
    # Let's try a moderate population.
    
    try:
        result = so.differential_evolution(
            objective, 
            bounds, 
            maxiter=50, 
            popsize=15, 
            tol=1e-5, 
            seed=42,
            # Callback to check time? DE doesn't support callback easily for stopping.
            # We can wrap the objective to check time and raise an exception? 
            # Or just run it and hope it finishes.
            # With vectorized eval, 50 iters * 15 pop * 135000 points...
            # Each eval is ~10-20ms.
            # 50 * 15 = 750 evals. 750 * 20ms = 15s.
            # This is very fast. We can increase maxiter.
        )
        
        if result.fun < 0: # Score is negative of objective
             best_params = result.x
        else:
             best_params = current_params
             
    except Exception as e:
        print(f"DE failed: {e}")
        best_params = current_params
        
    # Polish with Nelder-Mead
    if time.time() < deadline - 30:
        def nm_objective(params):
            path = make_path_from_params(params)
            return -evaluate_path_vectorized(path)
            
        result_nm = so.minimize(
            nm_objective, 
            best_params, 
            method='Nelder-Mead', 
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        if result_nm.fun < objective(best_params):
            best_params = result_nm.x
            
    # Final check
    final_path = make_path_from_params(best_params)
    if not np.all(np.isfinite(final_path)):
        final_path = make_path_from_params(current_params)
        
    return final_path
