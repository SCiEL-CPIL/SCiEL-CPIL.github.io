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
        valid_indices = np.arange(n_points)
        for x, y, a_deg in placements:
            a_rad = np.deg2rad(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
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

    # Optimization parameters
    angles_opt = [15.0, 30.0, 45.0, 60.0, 75.0]
    
    def path_from_params(params):
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
        return -score_path(path)

    # Bounds for x, y
    bounds = [(-1.5, 0.5), (-0.5, 1.5)] * 5
    
    # Run Differential Evolution
    t0 = time.time()
    test_params = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
    score_path(path_from_params(test_params))
    t1 = time.time()
    avg_eval_time = t1 - t0
    
    time_left = deadline - time.time()
    max_evals = int(time_left / avg_eval_time)
    max_iter = max(10, max_evals // 50) 
    popsize = 20
    
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
        pass

    # Local refinement using L-BFGS-B with a smooth surrogate
    # The score function is a step function (count * 0.0001). 
    # We can approximate the "margin" of the points to make it differentiable.
    # For each placement, a point is inside if min(dist_to_boundary) >= 0.
    # The score is proportional to the number of points with min_dist >= 0 across all placements.
    # A smooth surrogate for the "count" is the sum of sigmoid(min_dist / sigma) for each point?
    # Or sum of sigmoid(min_dist_i) where min_dist_i is the min over all placements.
    # Let's define f(params) = sum_j sigmoid(min_k dist(j, k) / sigma)
    # This is smooth and increasing. Maximizing this should increase the count.
    
    def smooth_surrogate(params):
        if time.time() > deadline:
            return 0
        path = path_from_params(params)
        dense = densify_path(path)
        
        # Compute distance to boundary for each point and each placement
        # dist = min over placements of (distance to boundary of that placement)
        # If dist > 0, point is inside.
        
        # To be efficient, we only compute for a subset of points? 
        # No, we need the sum.
        # Let's try to vectorize the distance calculation.
        
        # Actually, the current check_inside_fast is very fast.
        # L-BFGS-B requires gradients. We can use finite differences or provide gradients.
        # Providing gradients for the smooth surrogate is complex.
        # Let's use `minimize` with method='L-BFGS-B' and let it estimate gradients?
        # Or use 'COBYLA' which doesn't need gradients?
        # L-BFGS-B is better for high dimensions (10D).
        
        # Let's implement the gradient of the smooth surrogate.
        # Surrogate: S(params) = sum_j sigmoid( min_k D(j, k) / sigma )
        # D(j, k) is the distance from point j to the boundary of placement k.
        # If point is outside, D < 0. If inside, D > 0.
        
        # This is getting complicated to code correctly and fast enough for 100+ iterations.
        # Alternative: Use the existing score_path (integer count) and do a random search refinement (hill climbing) which is simpler and robust.
        # The inspiration 1 used random search refinement.
        # Let's stick to the random search refinement from inspiration 1 but make it more aggressive.
        
        return 0 # Placeholder, we won't use this for now.

    # Local refinement on the best path
    # Use a simple hill-climbing on the 10 parameters
    if time.time() < deadline - 30:
        current_params = best_path[:, :2].flatten() if best_path.shape[0] == 5 else None
        if current_params is None:
            # Project seed to 5-point path
            angles_seed = best_path[:, 2]
            x_interp = np.interp(angles_opt, angles_seed, best_path[:, 0])
            y_interp = np.interp(angles_opt, angles_seed, best_path[:, 1])
            current_params = np.column_stack((x_interp, y_interp)).flatten()
            
        current_score = score_path(path_from_params(current_params))
        step_size = 0.1
        rng = np.random.default_rng(42)
        
        # Aggressive random search
        max_refine_evals = 2000
        refine_start = time.time()
        refine_budget = deadline - 30 - refine_start
        
        # Estimate eval time again
        t0 = time.time()
        score_path(path_from_params(current_params))
        t1 = time.time()
        avg_refine_time = t1 - t0
        if avg_refine_time == 0:
            avg_refine_time = 0.01
        max_refine_evals = min(max_refine_evals, int(refine_budget / avg_refine_time))
        
        for i in range(max_refine_evals):
            if time.time() > deadline - 10:
                break
            perturbation = rng.normal(0, step_size, size=current_params.shape)
            new_params = current_params + perturbation
            new_params[0::2] = np.clip(new_params[0::2], -1.5, 0.5)
            new_params[1::2] = np.clip(new_params[1::2], -0.5, 1.5)
            
            new_score = score_path(path_from_params(new_params))
            if new_score > current_score:
                current_params = new_params
                current_score = new_score
                step_size *= 0.95
            else:
                step_size *= 0.98
                
        final_path = path_from_params(current_params)
        final_score = score_path(final_path)
        if final_score > best_score:
            best_path = final_path
            best_score = final_score

    return best_path
