import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # 20s safety margin

    # 1. Define the grid for evaluation
    # Grid points: (-3 + 0.01i, -1.5 + 0.01j), i=0..449, j=0..299
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    grid_points = np.column_stack((gx.ravel(), gy.ravel()))
    N_pts = grid_points.shape[0]
    
    # 2. Define the base corridor mask function
    # Base corridor: union of {0 <= u <= 1, -3 <= v <= 1} and {-3 <= u <= 1, 0 <= v <= 1}
    def in_base_corridor(u, v):
        rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        return rect1 | rect2

    # 3. Fast evaluation function
    # Takes a path (N, 3) array of (x, y, angle_degrees)
    # Returns the area of the intersection
    def evaluate_path(path):
        # path shape (N, 3)
        # We need to intersect all placements.
        # Start with all True
        mask = np.ones(N_pts, dtype=bool)
        
        for x, y, a_deg in path:
            a_rad = np.deg2rad(a_deg)
            c, s = np.cos(a_rad), np.sin(a_rad)
            
            # Rotate grid points relative to (x, y) by -a
            # w' = R(-a)(w - (x, y))
            # R(-a) = [[c, s], [-s, c]]
            dx = grid_points[:, 0] - x
            dy = grid_points[:, 1] - y
            
            u = c * dx + s * dy
            v = -s * dx + c * dy
            
            current_mask = in_base_corridor(u, v)
            mask &= current_mask
            
            # Early exit if mask becomes empty
            if not np.any(mask):
                return 0.0
                
        return np.sum(mask) * 0.0001

    # 4. Define the parametric path generator
    # Path is a circular arc:
    # x(theta) = Cx + R * cos(theta + delta)
    # y(theta) = Cy + R * sin(theta + delta)
    # theta is the corridor angle in radians.
    # We generate points for theta in [theta_start, theta_end]
    
    # The evaluator adds (0,0,0) and (-2,0,90).
    # My points should be in between.
    # Let's use angles 10 to 80 degrees for my generated points.
    # This allows the evaluator to interpolate 0->10 and 80->90.
    angles_deg = np.linspace(10, 80, 8)
    angles_rad = np.deg2rad(angles_deg)
    
    def generate_path(params):
        Cx, Cy, R, delta = params
        delta_rad = np.deg2rad(delta)
        
        xs = Cx + R * np.cos(angles_rad + delta_rad)
        ys = Cy + R * np.sin(angles_rad + delta_rad)
        
        path = np.column_stack((xs, ys, angles_deg))
        return path

    def objective(params):
        path = generate_path(params)
        area = evaluate_path(path)
        return -area

    # 5. Search Strategy
    best_params = None
    best_score = -np.inf
    
    # Coarse grid search
    # Ranges:
    # Cx: [-2, 0]
    # Cy: [-2, 0]
    # R: [1.0, 4.0]
    # delta: [0, 90]
    
    # We have ~580s.
    # Let's do a coarse grid of 10x10x10x10 = 10,000 evaluations.
    # Each evaluation is ~8 rotations of 135k points.
    # Vectorized NumPy should handle this in ~0.01-0.05s per eval.
    # 10,000 evals * 0.05s = 500s. This is tight.
    # Let's reduce grid size or optimize evaluation.
    
    # Optimization: Precompute rotations? No, angle changes.
    # Optimization: Use fewer points for coarse search?
    # Let's use a subsampled grid for the coarse search to speed it up.
    # Subsample by 4 (take every 4th point in both dims).
    # 135k points -> ~8k points. 16x faster.
    
    xs_sub = xs[::4]
    ys_sub = ys[::4]
    gx_sub, gy_sub = np.meshgrid(xs_sub, ys_sub, indexing='ij')
    grid_points_sub = np.column_stack((gx_sub.ravel(), gy_sub.ravel()))
    N_pts_sub = grid_points_sub.shape[0]
    
    def evaluate_path_sub(path):
        mask = np.ones(N_pts_sub, dtype=bool)
        for x, y, a_deg in path:
            a_rad = np.deg2rad(a_deg)
            c, s = np.cos(a_rad), np.sin(a_rad)
            dx = grid_points_sub[:, 0] - x
            dy = grid_points_sub[:, 1] - y
            u = c * dx + s * dy
            v = -s * dx + c * dy
            current_mask = in_base_corridor(u, v)
            mask &= current_mask
            if not np.any(mask):
                return 0.0
        return np.sum(mask) * (0.0001 * 16) # Scale area back up (4x4=16)

    def objective_sub(params):
        path = generate_path(params)
        area = evaluate_path_sub(path)
        return -area

    # Grid search
    cx_range = np.linspace(-2, 0, 8)
    cy_range = np.linspace(-2, 0, 8)
    r_range = np.linspace(1.0, 4.0, 8)
    delta_range = np.linspace(0, 90, 8)
    
    best_sub_score = -np.inf
    best_sub_params = None
    
    for Cx in cx_range:
        for Cy in cy_range:
            for R in r_range:
                for delta in delta_range:
                    if time.time() > deadline - 10:
                        break
                    params = [Cx, Cy, R, delta]
                    score = -objective_sub(params)
                    if score > best_sub_score:
                        best_sub_score = score
                        best_sub_params = params
                        
        if time.time() > deadline - 10:
            break
        if time.time() > deadline - 10:
            break
        if time.time() > deadline - 10:
            break

    if best_sub_params is None:
        # Fallback to baseline if search failed
        return np.array([
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ], dtype=np.float64)

    # Refine best candidate using full grid and local optimization
    current_params = np.array(best_sub_params)
    current_score = -objective(current_params)
    
    # Local optimization (Nelder-Mead)
    try:
        res = minimize(objective, current_params, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-5, 'adaptive': True})
        if res.fun < current_score:
            current_params = res.x
            current_score = -res.fun
    except Exception:
        pass

    # Check time and maybe do a few more local optimizations from nearby grid points
    if time.time() < deadline - 30:
        # Try a few other good candidates from the grid
        candidates = []
        for Cx in cx_range:
            for Cy in cy_range:
                for R in r_range:
                    for delta in delta_range:
                        if np.linalg.norm([Cx, Cy, R, delta] - best_sub_params) > 0.5:
                            candidates.append([Cx, Cy, R, delta])
        # Shuffle and try top 5 closest to best
        candidates.sort(key=lambda p: np.linalg.norm(np.array(p) - np.array(best_sub_params)))
        for p in candidates[:5]:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(objective, np.array(p), method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-5, 'adaptive': True})
                if -res.fun > current_score:
                    current_params = res.x
                    current_score = -res.fun
            except Exception:
                pass

    final_path = generate_path(current_params)
    return final_path
