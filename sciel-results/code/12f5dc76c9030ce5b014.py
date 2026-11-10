# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    # Grid definition
    x_grid = np.arange(-3.0, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = len(X_flat)
    
    # Precompute for fast evaluation
    # Base corridor: union of R1 and R2
    # R1: 0 <= u <= 1, -3 <= v <= 1
    # R2: -3 <= u <= 1, 0 <= v <= 1
    
    def check_inside(x, y, a_deg):
        theta = -np.deg2rad(a_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        wx = X_flat - x
        wy = Y_flat - y
        u = wx * c - wy * s
        v = wx * s + wy * c
        in_r1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        in_r2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        return in_r1 | in_r2
    
    def evaluate_path(path):
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        mask = np.ones(n_points, dtype=bool)
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
                inside = check_inside(x, y, a)
                mask &= inside
                if not mask.any():
                    return 0.0
        return np.sum(mask) * 0.0001
    
    # Use 6 control points for a smoother path
    # Fixed angles: 15, 30, 45, 60, 75, 85
    angles = np.array([15.0, 30.0, 45.0, 60.0, 75.0, 85.0])
    
    def make_path_from_params(params):
        # params: [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6]
        path = np.zeros((len(angles), 3))
        for i, a in enumerate(angles):
            path[i, 0] = params[2*i]
            path[i, 1] = params[2*i+1]
            path[i, 2] = a
        return path
    
    # Initial guess based on geometric intuition
    # The inner corner should move roughly along a curve from (0,0) to (-2,0)
    # while rotating from 0 to 90 degrees
    initial_params = np.array([
        -0.1, 0.1,   # a=15
        -0.2, 0.2,   # a=30
        -0.35, 0.25, # a=45
        -0.5, 0.2,   # a=60
        -0.65, 0.15, # a=75
        -0.8, 0.1,   # a=85
    ])
    
    start_time = time.time()
    time_limit = 590.0
    
    # First, evaluate initial path
    current_path = make_path_from_params(initial_params)
    current_score = evaluate_path(current_path)
    best_params = initial_params.copy()
    best_score = current_score
    
    # Use Nelder-Mead optimization
    def objective(params):
        if time.time() - start_time > time_limit - 10:
            return -best_score  # Return worst possible to stop
        path = make_path_from_params(params)
        return -evaluate_path(path)  # Negate for minimization
    
    # Run Nelder-Mead
    try:
        result = minimize(
            objective,
            initial_params,
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-5,
                'fatol': 1e-7,
                'adaptive': True
            }
        )
        if result.fun < -best_score:
            best_params = result.x
            best_score = -result.fun
    except Exception as e:
        pass
    
    # Local refinement with smaller steps
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005]
    for step in step_sizes:
        if time.time() - start_time > time_limit - 20:
            break
        improved = True
        while improved and time.time() - start_time < time_limit - 20:
            improved = False
            for dim in range(len(best_params)):
                for direction in [1, -1]:
                    new_params = best_params.copy()
                    new_params[dim] += direction * step
                    path = make_path_from_params(new_params)
                    val = evaluate_path(path)
                    if val > best_score:
                        best_score = val
                        best_params = new_params
                        improved = True
    
    final_path = make_path_from_params(best_params)
    return final_path

# EVOLVE-BLOCK-END
