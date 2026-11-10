import numpy as np
import time
import scipy.optimize as so

def moving_sofa_2d() -> np.ndarray:
    """
    Optimizes the path of the moving sofa to maximize the area of the intersection.
    """
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 40.0

    # Sofa points definition
    i_vals = np.arange(-3.0, 1.5, 0.01)
    j_vals = np.arange(-1.5, 1.5, 0.01)
    P_x, P_y = np.meshgrid(i_vals, j_vals, indexing='xy')
    P_x = P_x.flatten()
    P_y = P_y.flatten()
    N_pts = len(P_x)

    # Precompute base corridor check function for a single placement
    # We will vectorize over placements if possible, but here we iterate placements
    # and check all points.
    
    def check_placement(x, y, a_deg):
        """
        Returns boolean mask of sofa points inside the placement.
        """
        theta = np.deg2rad(a_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        dx = P_x - x
        dy = P_y - y
        
        # R(-a) rotation
        u = dx * cos_t + dy * sin_t
        v = -dx * sin_t + dy * cos_t
        
        # Base corridor:
        # {0 <= u <= 1, -3 <= v <= 1} OR {-3 <= u <= 1, 0 <= v <= 1}
        cond1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        cond2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        
        return cond1 | cond2

    def evaluate_path(path):
        """
        Evaluates the area of the intersection of sofa points with all placements in the path.
        path is a list of [x, y, angle]
        """
        # Densify the path as the evaluator does
        # Prepend (0,0,0) and append (-2,0,90)
        full_path = [(0.0, 0.0, 0.0)] + [tuple(p) for p in path] + [(-2.0, 0.0, 90.0)]
        
        densified = [full_path[0]]
        
        for i in range(len(full_path) - 1):
            x1, y1, a1 = full_path[i]
            x2, y2, a2 = full_path[i+1]
            
            da = abs(a2 - a1)
            m = max(1, int(np.ceil(da)))
            
            if m > 1:
                for k in range(1, m):
                    t = k / m
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    a = a1 + t * (a2 - a1)
                    densified.append((x, y, a))
            
            densified.append(full_path[i+1])
        
        # Check intersection
        # in_sofa is True for all points initially
        in_sofa = np.ones(N_pts, dtype=bool)
        
        for x, y, a in densified:
            mask = check_placement(x, y, a)
            in_sofa &= mask
            
            # Early exit if no points left
            if not np.any(in_sofa):
                return 0.0
        
        return np.sum(in_sofa) * 0.0001

    # Parameterization
    # We optimize the intermediate points.
    # Let's use 12 intermediate angles for more flexibility
    # Angles should be strictly increasing from 0 to 90
    angles = np.array([8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0, 72.0, 78.0, 83.0, 87.0])
    n_angles = len(angles)
    n_params = 2 * n_angles

    # Initial guess
    # Based on known good solutions for the moving sofa problem
    # We start with a reasonable guess that is valid
    # x, y typically move from (0,0) to (-2,0) while rotating
    
    # A simple linear interpolation in x,y for the initial guess
    init_x = np.linspace(-0.2, -1.8, n_angles)
    init_y = np.zeros(n_angles)
    
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
    best_params = p0.copy()
    
    def time_guard():
        return time.time() - start_time > (time_limit - safety_margin)

    # Strategy:
    # 1. Coarse search with Nelder-Mead or Powell
    # 2. Fine-tune with coordinate descent or L-BFGS-B (if smooth enough, but it's not)
    # Powell is robust for non-smooth.
    
    # First, try Powell
    try:
        res = so.minimize(objective, p0, method='Powell', 
                          options={'maxiter': 1000, 'maxfev': 5000, 'disp': False})
        if not time_guard():
            if res.fun < best_area:
                best_area = -res.fun
                best_params = res.x
    except Exception:
        pass

    # Second, try Nelder-Mead for local refinement
    if not time_guard():
        try:
            res2 = so.minimize(objective, best_params, method='Nelder-Mead', 
                               options={'maxiter': 2000, 'maxfev': 10000, 'disp': False})
            if not time_guard():
                if res2.fun < best_area:
                    best_area = -res2.fun
                    best_params = res2.x
        except Exception:
            pass

    # Third, manual coordinate descent for fine tuning
    if not time_guard():
        current_params = best_params.copy()
        current_area = -best_area
        step_size = 0.05
        for _ in range(50): # 50 iterations of coordinate descent
            if time_guard():
                break
            improved = False
            for i in range(n_params):
                if time_guard():
                    break
                original_val = current_params[i]
                
                # Try positive step
                current_params[i] = original_val + step_size
                area_pos = -objective(current_params)
                if area_pos > current_area:
                    current_area = area_pos
                    improved = True
                else:
                    # Try negative step
                    current_params[i] = original_val - step_size
                    area_neg = -objective(current_params)
                    if area_neg > current_area:
                        current_area = area_neg
                        improved = True
                    else:
                        # Revert
                        current_params[i] = original_val
                
                if not improved:
                    # Maybe reduce step size? Or just keep going
                    pass
            
            if not improved:
                # Reduce step size
                step_size *= 0.7
                if step_size < 0.001:
                    break

        if current_area > best_area:
            best_area = current_area
            best_params = current_params

    # Construct final path
    x_vals = best_params[:n_angles]
    y_vals = best_params[n_angles:]
    
    final_path = []
    for i in range(n_angles):
        final_path.append([x_vals[i], y_vals[i], angles[i]])
        
    path_arr = np.array(final_path, dtype=np.float64)
    
    # Sanity check: ensure finite
    if not np.all(np.isfinite(path_arr)):
        # Fallback to a known valid simple path
        fallback_x = np.linspace(-0.5, -1.5, n_angles)
        fallback_y = np.zeros(n_angles)
        final_path = []
        for i in range(n_angles):
            final_path.append([fallback_x[i], fallback_y[i], angles[i]])
        path_arr = np.array(final_path, dtype=np.float64)

    return path_arr

