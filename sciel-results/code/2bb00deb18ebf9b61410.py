import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import warnings

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # 50s margin before 630s kill

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
        """Add linear interpolation between consecutive placements as per evaluator."""
        if len(path) == 0:
            return np.array([])
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

    def count_inside(placements):
        """Count grid points inside all placements."""
        if len(placements) == 0:
            return n_points
        
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
            
            in_rect1 = (u >= -1e-9) & (u <= 1 + 1e-9) & (v >= -3 - 1e-9) & (v <= 1 + 1e-9)
            in_rect2 = (u >= -3 - 1e-9) & (u <= 1 + 1e-9) & (v >= -1e-9) & (v <= 1 + 1e-9)
            inside = in_rect1 | in_rect2
            
            if not np.any(inside):
                return 0
            
            valid_indices = valid_indices[inside]
            if len(valid_indices) == 0:
                return 0
        
        return len(valid_indices)

    def score_path(path):
        dense = densify_path(path)
        count = count_inside(dense)
        return count * 0.0001

    def get_path_from_ctrl(ctrl_pts, n_samples=30):
        """
        Generate a path from control points using cubic B-splines.
        ctrl_pts: (n_ctrl, 3) array.
        Returns: (n_samples, 3) array of (x, y, a).
        """
        if len(ctrl_pts) < 4:
            # Fallback to linear if too few points
            t = np.linspace(0, 1, n_samples)
            path = np.zeros((n_samples, 3))
            for i in range(n_samples):
                path[i] = ctrl_pts[0] + t[i] * (ctrl_pts[-1] - ctrl_pts[0])
            return path

        # Separate x, y, a
        x_ctrl = ctrl_pts[:, 0]
        y_ctrl = ctrl_pts[:, 1]
        a_ctrl = ctrl_pts[:, 2]
        
        t_ctrl = np.linspace(0, 1, len(ctrl_pts))
        
        try:
            cs_x = CubicSpline(t_ctrl, x_ctrl)
            cs_y = CubicSpline(t_ctrl, y_ctrl)
            cs_a = CubicSpline(t_ctrl, a_ctrl)
            
            t_samples = np.linspace(0, 1, n_samples)
            x_path = cs_x(t_samples)
            y_path = cs_y(t_samples)
            a_path = cs_a(t_samples)
            
            path = np.column_stack([x_path, y_path, a_path])
            return path
        except Exception:
            # Fallback to linear if spline fails
            t = np.linspace(0, 1, n_samples)
            path = np.zeros((n_samples, 3))
            for i in range(n_samples):
                if len(ctrl_pts) == 1:
                    path[i] = ctrl_pts[0]
                else:
                    idx = t[i] * (len(ctrl_pts) - 1)
                    i0 = int(idx)
                    i1 = min(i0 + 1, len(ctrl_pts) - 1)
                    frac = idx - i0
                    path[i] = (1 - frac) * ctrl_pts[i0] + frac * ctrl_pts[i1]
            return path

    # Optimization Setup
    n_ctrl = 6 # Number of control points
    ctrl_start = P_start
    ctrl_end = P_end
    
    # Variables to optimize: intermediate control points (n_ctrl - 2) * 3
    n_vars = (n_ctrl - 2) * 3
    
    bounds = []
    for _ in range(n_ctrl - 2):
        bounds.append((-2.5, 1.0)) # x
        bounds.append((-1.5, 1.5)) # y
        bounds.append((0.0, 90.0)) # a

    def objective(params):
        if time.time() > deadline - 10:
            return 0 # Penalty
        
        # Reconstruct control points
        ctrl_pts = np.zeros((n_ctrl, 3))
        ctrl_pts[0] = ctrl_start
        ctrl_pts[-1] = ctrl_end
        
        for i in range(n_ctrl - 2):
            ctrl_pts[i+1, 0] = params[3*i]
            ctrl_pts[i+1, 1] = params[3*i+1]
            ctrl_pts[i+1, 2] = params[3*i+2]
            
        # Generate path
        path = get_path_from_ctrl(ctrl_pts, n_samples=25)
        
        # Score
        score = score_path(path)
        return -score

    # Initial best
    best_score = 0
    best_params = None
    best_path = None
    
    # Initial guess: linear path
    init_ctrl = np.linspace(ctrl_start, ctrl_end, n_ctrl)
    init_params = init_ctrl[1:-1].flatten()
    
    # Timing check
    t0 = time.time()
    objective(init_params)
    t1 = time.time()
    avg_eval_time = t1 - t0
    
    time_left = deadline - time.time()
    max_evals = max(50, int(time_left * 0.8 / avg_eval_time))
    
    # Multi-start optimization
    np.random.seed(42)
    n_restarts = min(20, max(5, max_evals // 50))
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
            
        if restart == 0:
            current_params = init_params.copy()
        else:
            if best_params is not None and np.random.random() < 0.6:
                # Perturb best
                current_params = best_params + np.random.randn(n_vars) * 0.2
            else:
                # Random start
                current_params = np.random.uniform(
                    [b[0] for b in bounds],
                    [b[1] for b in bounds]
                )
            current_params = np.clip(current_params, [b[0] for b in bounds], [b[1] for b in bounds])
            
        try:
            result = minimize(
                objective,
                current_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
            )
            
            lbfgs_score = -result.fun
            if lbfgs_score > best_score:
                best_score = lbfgs_score
                best_params = result.x
                # Generate path for this best
                ctrl_pts = np.zeros((n_ctrl, 3))
                ctrl_pts[0] = ctrl_start
                ctrl_pts[-1] = ctrl_end
                for i in range(n_ctrl - 2):
                    ctrl_pts[i+1, 0] = result.x[3*i]
                    ctrl_pts[i+1, 1] = result.x[3*i+1]
                    ctrl_pts[i+1, 2] = result.x[3*i+2]
                
                best_path = get_path_from_ctrl(ctrl_pts, n_samples=25)
                
        except Exception:
            pass

    # Refine with Nelder-Mead if time permits
    if time.time() < deadline - 60 and best_params is not None:
        try:
            nm_result = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            nm_score = -nm_result.fun
            if nm_score > best_score:
                best_score = nm_score
                best_params = nm_result.x
                ctrl_pts = np.zeros((n_ctrl, 3))
                ctrl_pts[0] = ctrl_start
                ctrl_pts[-1] = ctrl_end
                for i in range(n_ctrl - 2):
                    ctrl_pts[i+1, 0] = best_params[3*i]
                    ctrl_pts[i+1, 1] = best_params[3*i+1]
                    ctrl_pts[i+1, 2] = best_params[3*i+2]
                best_path = get_path_from_ctrl(ctrl_pts, n_samples=25)
        except Exception:
            pass

    # Final validation and return
    if best_path is None:
        # Fallback: simple linear path
        best_path = np.linspace(P_start, P_end, 10)[1:-1]
        
    # Ensure the path is valid and finite
    if not np.all(np.isfinite(best_path)):
        best_path = np.linspace(P_start, P_end, 10)[1:-1]
        
    # Final check
    final_score = score_path(best_path)
    if final_score <= 0:
        # If score is 0, try a very simple path
        best_path = np.array([
            [0.0, 0.0, 5.0],
            [-0.5, 0.5, 15.0],
            [-1.0, 0.5, 30.0],
            [-1.5, 0.0, 60.0],
            [-1.8, 0.0, 80.0]
        ])
        final_score = score_path(best_path)
        if final_score <= 0:
             # Absolute fallback
             best_path = np.array([[-1.0, 0.0, 45.0]])

    return best_path
