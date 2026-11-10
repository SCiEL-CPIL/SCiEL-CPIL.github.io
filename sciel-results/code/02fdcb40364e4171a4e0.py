import numpy as np
import time
import scipy.optimize as so

def moving_sofa_2d() -> np.ndarray:
    """
    Optimizes the path of the moving sofa to maximize the area of the intersection.
    """
    start_time = time.time()
    time_limit = 600.0 # Hard limit
    safety_margin = 30.0 # Stop 30s before limit
    
    # Sofa points definition
    # i = 0..449, j = 0..299
    # x = -3 + 0.01*i
    # y = -1.5 + 0.01*j
    i_vals = np.arange(-3.0, 1.5, 0.01)
    j_vals = np.arange(-1.5, 1.5, 0.01)
    P_x, P_y = np.meshgrid(i_vals, j_vals, indexing='xy')
    P_x = P_x.flatten()
    P_y = P_y.flatten()
    N_pts = len(P_x)
    
    # Function to evaluate the area for a given path (list of [x, y, angle])
    def evaluate_path(path):
        # Densify the path as the evaluator does
        # Prepend (0,0,0) and append (-2,0,90)
        full_path = [(0.0, 0.0, 0.0)] + path + [(-2.0, 0.0, 90.0)]
        
        densified = []
        for i in range(len(full_path) - 1):
            x1, y1, a1 = full_path[i]
            x2, y2, a2 = full_path[i+1]
            
            # m = max(1, ceil(|a2 - a1|))
            da = abs(a2 - a1)
            m = max(1, int(np.ceil(da)))
            
            # Linear interpolation in (x, y, a)
            # The evaluator says "linear interpolations of (x, y, a) in m equal steps"
            # This usually means t goes from 0 to 1 in m steps? Or m-1 intervals?
            # "in m = ... equal steps" usually implies m intervals, so m+1 points?
            # Or does it mean m points?
            # Let's look at the phrasing: "adds linear interpolations ... in m equal steps".
            # Standard interpretation: t_k = k/m for k=0..m.
            # However, the endpoints are already in full_path.
            # If we add points for k=1..m-1, we have m+1 total points including endpoints.
            # Let's assume the evaluator generates the full dense path.
            # To be safe, we simulate what the evaluator likely does.
            # If m=1, no intermediate points.
            # If m=2, 1 intermediate point at t=0.5.
            
            if m > 1:
                for k in range(1, m):
                    t = k / m
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    a = a1 + t * (a2 - a1)
                    densified.append((x, y, a))
        
        # Append the last point
        densified.append(full_path[-1])
        # Prepend the first point
        densified = [full_path[0]] + densified
        
        # Now check intersection
        # in_sofa is True for all points initially
        in_sofa = np.ones(N_pts, dtype=bool)
        
        # Process each placement
        for x, y, a in densified:
            theta = np.deg2rad(a)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            
            # R(-a) (w - (x,y))
            # w = (wx, wy)
            # dx = wx - x
            # dy = wy - y
            # u = dx * cos(-a) - dy * sin(-a) = dx * cos(a) + dy * sin(a)
            # v = dx * sin(-a) + dy * cos(-a) = -dx * sin(a) + dy * cos(a)
            
            dx = P_x - x
            dy = P_y - y
            
            u = dx * cos_t + dy * sin_t
            v = -dx * sin_t + dy * cos_t
            
            # Base corridor:
            # {0 <= u <= 1, -3 <= v <= 1} OR {-3 <= u <= 1, 0 <= v <= 1}
            cond1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            cond2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            mask = cond1 | cond2
            
            in_sofa &= mask
            
            # Early exit if no points left
            if not np.any(in_sofa):
                return 0.0
        
        return np.sum(in_sofa) * 0.0001

    # Parameterization
    # We optimize the intermediate points.
    # Let's use 10 intermediate angles for more flexibility
    angles = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0, 88.0])
    n_angles = len(angles)
    n_params = 2 * n_angles
    
    # Initial guess from baseline
    # Baseline:
    # [-0.1, 0.1, 15.0],
    # [-0.2, 0.2, 20.0],
    # [-0.3, 0.3, 34.0],
    # [-0.3, 0.3, 40.0],
    # [-0.4, 0.3, 50.0],
    # [-0.5, 0.2, 55.0],
    # [-0.6, 0.2, 75.0],
    # [-0.7, 0.3, 86.0]
    
    # Interpolate x, y for our target angles from the baseline points
    base_x = [-0.1, -0.2, -0.3, -0.3, -0.4, -0.5, -0.6, -0.7]
    base_y = [0.1, 0.2, 0.3, 0.3, 0.3, 0.2, 0.2, 0.3]
    base_a = [15.0, 20.0, 34.0, 40.0, 50.0, 55.0, 75.0, 86.0]
    
    init_x = np.interp(angles, base_a, base_x)
    init_y = np.interp(angles, base_a, base_y)
    
    p0 = np.concatenate([init_x, init_y])
    
    # Objective function for optimizer (minimize negative area)
    def objective(params):
        x_vals = params[:n_angles]
        y_vals = params[n_angles:]
        
        # Construct path
        path = []
        for i in range(n_angles):
            path.append([x_vals[i], y_vals[i], angles[i]])
        
        area = evaluate_path(path)
        return -area

    # Optimize
    best_area = objective(p0)
    best_params = p0
    
    # Use Powell method, robust for non-smooth
    # Set maxiter and maxfun to prevent runaway, but we have time guard
    max_iter = 2000
    max_fun = 2000
    
    def time_guard():
        return time.time() - start_time > (time_limit - safety_margin)
        
    try:
        # We might run multiple optimizations or just one long one.
        # Powell is good.
        res = so.minimize(objective, p0, method='Powell', 
                          options={'maxiter': max_iter, 'maxfev': max_fun, 'disp': False})
        
        if not time_guard():
            if res.fun < best_area:
                best_area = -res.fun
                best_params = res.x
    except Exception as e:
        pass
        
    # If time remains, try a second optimization with more waypoints or different method
    if not time_guard():
        # Try Nelder-Mead
        try:
            res2 = so.minimize(objective, best_params, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'maxfev': 1000, 'disp': False})
            if not time_guard():
                if res2.fun < best_area:
                    best_area = -res2.fun
                    best_params = res2.x
        except Exception:
            pass

    # Construct final path
    x_vals = best_params[:n_angles]
    y_vals = best_params[n_angles:]
    
    final_path = []
    for i in range(n_angles):
        final_path.append([x_vals[i], y_vals[i], angles[i]])
        
    path_arr = np.array(final_path, dtype=np.float64)
    
    return path_arr
