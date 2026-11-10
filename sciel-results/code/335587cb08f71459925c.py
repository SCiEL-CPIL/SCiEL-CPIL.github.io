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
    # We will optimize the x,y positions at 7 fixed angles: 0, 15, 30, 45, 60, 75, 90
    # Angles 0 and 90 are fixed by the evaluator, so we only optimize 15, 30, 45, 60, 75
    # That's 5 points * 2 = 10 variables.
    # Wait, the evaluator prepends (0,0,0) and appends (-2,0,90).
    # If we provide points at 15, 30, 45, 60, 75, the path is:
    # (0,0,0) -> (x1,y1,15) -> (x2,y2,30) -> ... -> (x5,y5,75) -> (-2,0,90)
    # This is a good parameterization.
    
    angles_opt = [15.0, 30.0, 45.0, 60.0, 75.0]
    
    def path_from_params(params):
        # params: [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
        path = np.zeros((5, 3))
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
    # x is roughly in [-1.5, 0.5]
    # y is roughly in [-0.5, 1.5]
    bounds = [(-1.5, 0.5), (-0.5, 1.5)] * 5
    
    # Run Differential Evolution
    # maxiter and popsize tuned for time budget
    # Let's start with a small pop and few iterations, then expand if time permits.
    
    # Time check
    t0 = time.time()
    # Quick test
    test_params = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
    score_path(path_from_params(test_params))
    t1 = time.time()
    avg_eval_time = t1 - t0
    
    time_left = deadline - time.time()
    # Estimate max evaluations
    max_evals = int(time_left / avg_eval_time)
    # DE overhead is roughly 2-3x the number of function evaluations
    max_iter = max(10, max_evals // 50) 
    popsize = 20
    
    print(f"Starting DE with max_iter={max_iter}, popsize={popsize}")
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=max_iter, 
            popsize=popsize,
            seed=42,
            polish=False, # Polish is slow, we'll do our own
            init='sobol'
        )
        
        if result.fun < 0: # If we found a better path
            best_params = result.x
            best_path_de = path_from_params(best_params)
            best_score_de = -result.fun
            if best_score_de > best_score:
                best_path = best_path_de
                best_score = best_score_de
    except Exception as e:
        print(f"DE failed: {e}")

    # Local refinement on the best path
    # Use a simple hill-climbing on the 10 parameters
    if time.time() < deadline - 30:
        print("Starting local refinement")
        current_params = best_path[:, :2].flatten() if best_path.shape[0] == 5 else best_params
        # If best_path is the seed (8 points), we need to project it to 5 points or just use the DE result
        # Let's assume best_path is from DE or we convert seed to 5-point path
        if best_path.shape[0] != 5:
            # Project seed to 5-point path by sampling
            angles_seed = best_path[:, 2]
            # Interpolate x,y at angles_opt
            x_interp = np.interp(angles_opt, angles_seed, best_path[:, 0])
            y_interp = np.interp(angles_opt, angles_seed, best_path[:, 1])
            current_params = np.column_stack((x_interp, y_interp)).flatten()
            
        current_score = -objective(current_params)
        step_size = 0.1
        rng = np.random.default_rng(42)
        
        for i in range(100):
            if time.time() > deadline - 10:
                break
            # Random perturbation
            perturbation = rng.normal(0, step_size, size=current_params.shape)
            new_params = current_params + perturbation
            # Clip to bounds
            new_params[0::2] = np.clip(new_params[0::2], -1.5, 0.5)
            new_params[1::2] = np.clip(new_params[1::2], -0.5, 1.5)
            
            new_score = -objective(new_params)
            if new_score > current_score:
                current_params = new_params
                current_score = new_score
                step_size *= 0.9
            else:
                step_size *= 0.95
                
        final_path = path_from_params(current_params)
        final_score = score_path(final_path)
        if final_score > best_score:
            best_path = final_path
            best_score = final_score

    return best_path
