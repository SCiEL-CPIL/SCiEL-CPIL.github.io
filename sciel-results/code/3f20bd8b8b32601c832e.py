import numpy as np
import time
from scipy.optimize import differential_evolution

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # 20s margin

    # Grid definition
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = X_flat.size

    P_start = np.array([0.0, 0.0, 0.0])
    P_end = np.array([-2.0, 0.0, 90.0])

    def densify_path(path):
        full_path = np.vstack([P_start, path, P_end])
        dense_points = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            delta_a = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(delta_a)))
            for k in range(1, m + 1):
                t = k / m
                p_int = p1 + t * (p2 - p1)
                dense_points.append(p_int)
        return np.array(dense_points)

    def check_inside_fast(placements):
        # Adaptive mask
        valid_mask = np.ones(n_points, dtype=bool)
        valid_indices = np.arange(n_points)
        
        for x, y, a_deg in placements:
            a_rad = np.deg2rad(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            # Only check valid points
            Xv = X_flat[valid_indices]
            Yv = Y_flat[valid_indices]
            
            dx = Xv - x
            dy = Yv - y
            
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a
            
            in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            inside = in_rect1 | in_rect2
            
            if not np.any(inside):
                return 0
            
            # Update valid indices
            valid_indices = valid_indices[inside]
            
        return len(valid_indices)

    def score_path(path):
        dense = densify_path(path)
        count = check_inside_fast(dense)
        return count * 0.0001

    # Initial best path (from seed)
    best_path = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)
    best_score = score_path(best_path)

    # Define optimization function
    # We will optimize the x,y positions at 7 fixed angles: 10, 20, 30, 45, 60, 75, 85
    # Angles 0 and 90 are fixed by the evaluator, so we optimize 7 intermediate points
    # That's 7 points * 2 = 14 variables.
    angles_opt = [10.0, 20.0, 30.0, 45.0, 60.0, 75.0, 85.0]
    n_wp = len(angles_opt)
    
    def path_from_params(params):
        # params: [x1, y1, x2, y2, ..., x7, y7]
        path = np.zeros((n_wp, 3))
        for i, a in enumerate(angles_opt):
            path[i, 0] = params[2*i]
            path[i, 1] = params[2*i+1]
            path[i, 2] = a
        return path

    def objective(params):
        if time.time() > deadline:
            return 0
        path = path_from_params(params)
        return -score_path(path) # Minimize negative score

    # Bounds for x, y
    bounds = [(-1.5, 0.5), (-0.5, 1.5)] * n_wp
    
    # Run Differential Evolution
    t0 = time.time()
    test_params = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
    score_path(path_from_params(test_params))
    t1 = time.time()
    avg_eval_time = t1 - t0
    
    time_left = deadline - time.time()
    max_evals = int(time_left / avg_eval_time)
    max_iter = max(10, max_evals // 70) 
    popsize = 15
    
    print(f"Starting DE with max_iter={max_iter}, popsize={popsize}")
    
    best_params = None
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=max_iter, 
            popsize=popsize,
            seed=42,
            polish=False,
            init='sobol'
        )
        
        if result.fun < 0:
            best_params = result.x
            best_path_de = path_from_params(best_params)
            best_score_de = -result.fun
            if best_score_de > best_score:
                best_path = best_path_de
                best_score = best_score_de
    except Exception as e:
        print(f"DE failed: {e}")

    # Local refinement using Nelder-Mead simplex
    if time.time() < deadline - 30:
        print("Starting local refinement with Nelder-Mead")
        from scipy.optimize import minimize
        
        if best_params is not None:
            current_params = best_params
        else:
            # Project seed to 7-point path
            angles_seed = best_path[:, 2]
            x_interp = np.interp(angles_opt, angles_seed, best_path[:, 0])
            y_interp = np.interp(angles_opt, angles_seed, best_path[:, 1])
            current_params = np.column_stack((x_interp, y_interp)).flatten()
            
        current_score = -objective(current_params)
        
        # Nelder-Mead with tighter tolerance
        try:
            nm_result = minimize(
                objective,
                current_params,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            if nm_result.fun < current_score:
                current_params = nm_result.x
                current_score = nm_result.fun
        except Exception as e:
            print(f"Nelder-Mead failed: {e}")
        
        # Coordinate ascent polish
        if time.time() < deadline - 15:
            print("Starting coordinate ascent polish")
            step_size = 0.05
            for coord in range(len(current_params)):
                if time.time() > deadline - 10:
                    break
                for direction in [-1, 1]:
                    while True:
                        if time.time() > deadline - 10:
                            break
                        new_params = current_params.copy()
                        new_params[coord] += direction * step_size
                        # Clip to bounds
                        new_params[coord] = np.clip(new_params[coord], bounds[coord][0], bounds[coord][1])
                        new_score = -objective(new_params)
                        if new_score < current_score:  # Lower is better for objective
                            current_params = new_params
                            current_score = new_score
                        else:
                            break
                step_size *= 0.5
                if step_size < 1e-4:
                    break
        
        final_path = path_from_params(current_params)
        final_score = score_path(final_path)
        if final_score > best_score:
            best_path = final_path
            best_score = final_score

    return best_path
