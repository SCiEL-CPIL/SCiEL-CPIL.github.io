# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize

def moving_sofa_2d() -> np.ndarray:
    """
    Find a path for the corridor such that the intersection of all placements
    (including the start (0,0,0) and end (-2,0,90)) has maximum area.
    
    We use a parametric path model and optimize its parameters using Nelder-Mead.
    The path is defined by t in [0, 1]:
    x(t) = -2 * t
    y(t) = c1 * sin(pi * t) + c2 * t
    angle(t) = 90 * t
    
    We optimize c1 and c2 to maximize the intersection area.
    """
    
    def evaluate_path(path):
        # Add start and end
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        
        # Interpolate
        interpolated = []
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(m + 1):
                t = k / m
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = p1[2] + t * (p2[2] - p1[2])
                interpolated.append([x, y, a])
        
        interpolated = np.array(interpolated)
        
        # Grid points
        # Using a slightly coarser grid for speed during optimization, 
        # then refining? No, the score is based on the specific grid in the contract.
        # We must use the exact grid.
        xs = np.arange(-3.0, 1.5, 0.01)
        ys = np.arange(-1.5, 1.5, 0.01)
        X, Y = np.meshgrid(xs, ys)
        X = X.flatten()
        Y = Y.flatten()
        
        in_all = np.ones(len(X), dtype=bool)
        
        for x, y, a_deg in interpolated:
            a_rad = np.radians(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = X - x
            dy = Y - y
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            in_arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            in_corridor = in_arm1 | in_arm2
            
            in_all &= in_corridor
            
            # Early exit if no points remain
            if not np.any(in_all):
                return 0.0
        
        return np.sum(in_all) * 0.0001

    def get_path_from_params(params):
        c1, c2 = params
        n_points = 10 # Number of intermediate waypoints
        t_vals = np.linspace(0.1, 0.9, n_points)
        
        x_vals = -2.0 * t_vals
        y_vals = c1 * np.sin(np.pi * t_vals) + c2 * t_vals
        a_vals = 90.0 * t_vals
        
        path = np.column_stack((x_vals, y_vals, a_vals))
        return path

    # Objective function for minimizer (we want to maximize score, so minimize negative score)
    def objective(params):
        path = get_path_from_params(params)
        score = evaluate_path(path)
        return -score

    # Initial guess
    # Based on previous best, y was around 0.2-0.3.
    # y(t) = c1 sin(pi t) + c2 t
    # If c2=0, y is a hump. If c1=0.2, max y is 0.2.
    # Let's try c1=0.2, c2=0.0
    initial_params = [0.2, 0.0]
    
    # Bounds for c1 and c2
    # y should probably stay within [-0.5, 0.5] to keep the sofa in the corridor
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    
    # Use Nelder-Mead
    # To speed up, we might reduce the grid size during optimization?
    # The contract says "The sofa is the set of grid points...". The evaluator uses that exact grid.
    # We must use the exact grid to get the correct score.
    # 145000 points * ~100 placements * 1000 iterations might be too slow.
    # 145000 * 100 * 1000 = 14.5 billion operations. Too slow.
    
    # Optimization Strategy:
    # 1. Use a coarse grid for the optimization loop.
    # 2. Use the fine grid only for the final evaluation and return.
    
    def evaluate_path_coarse(path, step=0.02):
        # Same as evaluate_path but with coarser grid
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        
        interpolated = []
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(m + 1):
                t = k / m
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = p1[2] + t * (p2[2] - p1[2])
                interpolated.append([x, y, a])
        
        interpolated = np.array(interpolated)
        
        xs = np.arange(-3.0, 1.5, step)
        ys = np.arange(-1.5, 1.5, step)
        X, Y = np.meshgrid(xs, ys)
        X = X.flatten()
        Y = Y.flatten()
        
        in_all = np.ones(len(X), dtype=bool)
        
        for x, y, a_deg in interpolated:
            a_rad = np.radians(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = X - x
            dy = Y - y
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            in_arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            in_corridor = in_arm1 | in_arm2
            
            in_all &= in_corridor
            if not np.any(in_all):
                return 0.0
        
        # Scale the score to approximate the fine grid score?
        # Area scales with step^2. Fine step is 0.01, coarse is 0.02.
        # Number of points scales with 1/step^2.
        # Score = count * 0.0001.
        # If we use step 0.02, count is 1/4 of fine grid.
        # So score_coarse * 4 ~ score_fine.
        return np.sum(in_all) * 0.0001 * (0.01/step)**2

    def objective_coarse(params):
        path = get_path_from_params(params)
        score = evaluate_path_coarse(path)
        return -score

    # Optimize with coarse grid
    result = scipy.optimize.minimize(objective_coarse, initial_params, method='Nelder-Mead', 
                                      options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
    
    best_params = result.x
    best_path = get_path_from_params(best_params)
    
    # Final evaluation with fine grid to ensure we return a good path
    # We might do a local search around the best params with fine grid?
    # Just use the best path from coarse optimization.
    
    return best_path

# EVOLVE-BLOCK-END
