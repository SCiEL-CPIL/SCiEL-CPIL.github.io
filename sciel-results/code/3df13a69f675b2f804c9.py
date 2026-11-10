import numpy as np
from scipy.optimize import differential_evolution
import time

def moving_sofa_2d() -> np.ndarray:
    """
    Optimizes the path of the L-shaped corridor to maximize the area of the
    intersection of all corridor placements (the "sofa").
    """
    # Grid definition
    # i = 0..449, j = 0..299
    # x = -3 + 0.01i, y = -1.5 + 0.01j
    x_grid = np.arange(-3, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid)
    points = np.column_stack((X.ravel(), Y.ravel()))
    
    def point_in_base_corridor(u, v):
        # Base: {0 <= u <= 1, -3 <= v <= 1} U {-3 <= u <= 1, 0 <= v <= 1}
        cond1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        cond2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        return cond1 | cond2

    def evaluate_path(path):
        # path: (n, 3) array
        start = np.array([[0.0, 0.0, 0.0]])
        end = np.array([[-2.0, 0.0, 90.0]])
        full_path = np.vstack([start, path, end])
        
        # Densify
        dense_path = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            if m > 1:
                ts = np.linspace(0, 1, m+1)[1:-1]
                for t in ts:
                    dense_path.append(p1 + t * (p2 - p1))
            dense_path.append(p2)
            
        dense_path = np.array(dense_path)
        
        mask = np.ones(len(points), dtype=bool)
        
        for p in dense_path:
            x, y, a_deg = p
            a_rad = np.deg2rad(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = points[:, 0] - x
            dy = points[:, 1] - y
            
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            cond = point_in_base_corridor(u, v)
            mask &= cond
            if not mask.any():
                return 0.0
                
        return np.sum(mask)

    n_points = 12
    t = np.linspace(0.05, 0.95, n_points)
    init_x = -2 * t
    init_y = np.zeros(n_points)
    init_a = 90 * t
    
    x0 = np.column_stack((init_x, init_y, init_a)).ravel()
    
    bounds = []
    for i in range(n_points):
        bounds.append((-3.0, 1.0))
        bounds.append((-3.0, 1.0))
        bounds.append((0.0, 90.0))
        
    def objective(params):
        p = params.reshape(n_points, 3)
        idx = np.argsort(p[:, 2])
        p_sorted = p[idx]
        area = evaluate_path(p_sorted)
        return -area

    start_time = time.time()
    try:
        # Use a time limit or maxiter to stay within 600s
        # 600s is generous.
        result = differential_evolution(objective, bounds, maxiter=200, popsize=15, seed=42, polish=False, tol=1e-4)
    except Exception:
        result = None

    if result and result.success:
        best_params = result.x
        best_path = best_params.reshape(n_points, 3)
        idx = np.argsort(best_path[:, 2])
        best_path = best_path[idx]
        return best_path.astype(np.float64)
    else:
        init_path = np.column_stack((init_x, init_y, init_a))
        return init_path.astype(np.float64)
