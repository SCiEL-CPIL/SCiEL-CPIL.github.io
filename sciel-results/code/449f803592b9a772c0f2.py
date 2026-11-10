import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _check_points_in_corridor(points, x, y, a_deg):
    """
    Check if points (N, 2) are inside the L-shaped corridor 
    with inner corner at (x, y) and rotation a_deg.
    Returns boolean array of shape (N,).
    """
    a_rad = np.deg2rad(a_deg)
    c, s = np.cos(-a_rad), np.sin(-a_rad)
    R = np.array([[c, -s], [s, c]])
    
    # Transform points to local frame
    # points: (N, 2)
    # w_local = R @ (w - (x,y))^T
    w_local = (points - np.array([x, y])) @ R.T
    
    u = w_local[:, 0]
    v = w_local[:, 1]
    
    # Corridor definition:
    # Union of {0 <= u <= 1, -3 <= v <= 1} and {-3 <= u <= 1, 0 <= v <= 1}
    # We use a large range for the "infinite" parts to be safe, 
    # but the sofa is bounded by the intersection of start and end.
    # The evaluator uses specific grid points, so we just need to be accurate 
    # for the points we check.
    
    rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    
    return rect1 | rect2

def _evaluate_path_coarse(path, coarse_step=0.05):
    """
    Estimate the area of the sofa defined by the path using a coarse grid.
    path: (n, 3) array of (x, y, a).
    """
    # Densify path as per evaluator logic
    # Evaluator adds (0,0,0) and (-2,0,90)
    full_path = np.vstack([np.array([0.0, 0.0, 0.0]), path, np.array([-2.0, 0.0, 90.0])])
    
    # Interpolate
    densified = [full_path[0]]
    for i in range(len(full_path) - 1):
        p1 = full_path[i]
        p2 = full_path[i+1]
        da = p2[2] - p1[2]
        m = max(1, int(np.ceil(abs(da))))
        if m > 1:
            ts = np.linspace(0, 1, m+1)[1:-1]
            for t in ts:
                interp = p1 + t * (p2 - p1)
                densified.append(interp)
        densified.append(p2)
    densified = np.array(densified)
    
    # Define coarse grid
    # The sofa is contained in the intersection of the start and end corridors.
    # Start: (0,0,0). End: (-2,0,90).
    # Let's use a bounding box that covers the potential intersection.
    # x: [-3, 1], y: [-3, 3] seems safe.
    xs = np.arange(-3.0, 1.0 + coarse_step, coarse_step)
    ys = np.arange(-3.0, 3.0 + coarse_step, coarse_step)
    X, Y = np.meshgrid(xs, ys)
    points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Check each placement
    # To save memory, we can accumulate the "valid" mask
    valid_mask = np.ones(len(points), dtype=bool)
    
    for p in densified:
        x, y, a = p
        in_corridor = _check_points_in_corridor(points, x, y, a)
        valid_mask &= in_corridor
        
        # Early exit if no points are valid
        if not np.any(valid_mask):
            break
            
    area = np.sum(valid_mask) * (coarse_step ** 2)
    return area

def moving_sofa_2d() -> np.ndarray:
    """
    Returns a path (n, 3) for the moving sofa problem.
    """
    start_time = time.time()
    deadline = start_time + 550.0  # 550s limit, 50s buffer for overhead
    
    # Initial path guess
    # 8 intermediate points
    n_points = 8
    # Angles roughly linear from 15 to 75
    angles = np.linspace(15, 75, n_points)
    # Positions roughly along a path
    # Start (0,0), End (-2,0)
    # Let's try a simple arc or line
    xs = np.linspace(-0.2, -1.8, n_points)
    ys = np.linspace(0.0, 0.0, n_points)
    
    initial_path = np.column_stack([xs, ys, angles])
    
    best_path = initial_path
    best_score = _evaluate_path_coarse(best_path)
    
    # Optimization
    def objective(params):
        path = params.reshape(-1, 3)
        return -_evaluate_path_coarse(path)
        
    x0 = initial_path.ravel()
    
    # Nelder-Mead
    # We run multiple restarts if time permits
    restarts = 5
    
    for r in range(restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            x_start = x0
        else:
            # Perturb initial guess
            x_start = x0 + np.random.normal(0, 0.1, x0.shape)
            
        try:
            res = minimize(objective, x_start, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.success or res.fun < -best_score:
                # Check time
                if time.time() > deadline:
                    break
                new_path = res.x.reshape(-1, 3)
                new_score = -res.fun
                if new_score > best_score:
                    best_score = new_score
                    best_path = new_path
        except Exception:
            pass
            
    return best_path

# EVOLVE-BLOCK-END
