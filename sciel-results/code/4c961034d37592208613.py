# EVOLVE-BLOCK-START
import numpy as np
from time import time
from scipy.optimize import minimize

def _check_point_in_base(u, v):
    """Check if point (u,v) is in the base L-shaped corridor."""
    # Horizontal arm: 0 <= u <= 1, -3 <= v <= 1
    h = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    # Vertical arm: -3 <= u <= 1, 0 <= v <= 1
    v_arm = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    return h | v_arm

def _compute_sofa_area(path, grid_x, grid_y):
    """
    Compute the number of grid points inside all corridor placements.
    path: (n, 3) array of (x, y, angle_deg)
    grid_x, grid_y: 1D arrays of grid coordinates
    Returns: count of points inside all placements
    """
    # Build full grid
    gx, gy = np.meshgrid(grid_x, grid_y)
    gx = gx.ravel()
    gy = gy.ravel()
    points = np.column_stack([gx, gy])  # (N, 2)
    
    # Start with all points as candidates
    inside_all = np.ones(len(points), dtype=bool)
    
    # Evaluator adds (0,0,0) before and (-2,0,90) after
    all_placements = np.vstack([
        np.array([[0.0, 0.0, 0.0]]),
        path,
        np.array([[-2.0, 0.0, 90.0]])
    ])
    
    # Interpolate between consecutive placements
    for i in range(len(all_placements) - 1):
        x1, y1, a1 = all_placements[i]
        x2, y2, a2 = all_placements[i+1]
        m = max(1, int(np.ceil(abs(a2 - a1))))
        if m == 1:
            # Only the endpoint (a2) is added; a1 is already covered
            t = np.array([1.0])
        else:
            t = np.linspace(0, 1, m+1)[1:]  # skip t=0 since a1 is previous placement
        for ti in t:
            xi = x1 + ti * (x2 - x1)
            yi = y1 + ti * (y2 - y1)
            ai = a1 + ti * (a2 - a1)
            # Rotate points by -ai degrees around (xi, yi)
            rad = np.radians(-ai)
            c = np.cos(rad)
            s = np.sin(rad)
            dx = points[:, 0] - xi
            dy = points[:, 1] - yi
            u = c * dx - s * dy
            v = s * dx + c * dy
            mask = _check_point_in_base(u, v)
            inside_all &= mask
            if not np.any(inside_all):
                return 0
    
    return int(np.sum(inside_all))

def _objective(params, grid_x, grid_y):
    """Negative area for minimization."""
    # params: [x1, y1, a1, x2, y2, a2, ..., xn, yn, an]
    n = len(params) // 3
    path = params.reshape(n, 3)
    # Ensure angles are in [0, 90] roughly
    path[:, 2] = np.clip(path[:, 2], 0, 90)
    count = _compute_sofa_area(path, grid_x, grid_y)
    return -count

def moving_sofa_2d() -> np.ndarray:
    start_time = time()
    deadline = start_time + 600  # 600s deadline
    
    # Grid points
    grid_x = np.arange(-3, 1.5, 0.01)
    grid_y = np.arange(-1.5, 1.5, 0.01)
    
    # Baseline path
    baseline = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)
    
    best_path = baseline.copy()
    best_count = _compute_sofa_area(best_path, grid_x, grid_y)
    
    # Try different numbers of control points
    for n_ctrl in [6, 8, 10]:
        if time() > deadline - 120:
            break
        
        # Initialize with evenly spaced angles
        angles = np.linspace(10, 85, n_ctrl)
        xs = np.linspace(-0.1, -0.7, n_ctrl)
        ys = np.linspace(0.1, 0.3, n_ctrl)
        init = np.column_stack([xs, ys, angles]).ravel()
        
        # Nelder-Mead optimization
        try:
            res = minimize(
                _objective, init,
                args=(grid_x, grid_y),
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 1e-3,
                    'fatol': 1e-3,
                    'adaptive': True
                }
            )
            if time() > deadline - 120:
                break
            
            opt_path = res.x.reshape(n_ctrl, 3)
            opt_path[:, 2] = np.clip(opt_path[:, 2], 0, 90)
            opt_count = _compute_sofa_area(opt_path, grid_x, grid_y)
            
            if opt_count > best_count:
                best_count = opt_count
                best_path = opt_path.copy()
        except Exception:
            pass
    
    # Local refinement with more iterations if time permits
    if time() < deadline - 60 and len(best_path) > 0:
        n_ctrl = len(best_path)
        init = best_path.ravel()
        try:
            res = minimize(
                _objective, init,
                args=(grid_x, grid_y),
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-4,
                    'fatol': 1e-4,
                    'adaptive': True
                }
            )
            if time() < deadline - 30:
                opt_path = res.x.reshape(n_ctrl, 3)
                opt_path[:, 2] = np.clip(opt_path[:, 2], 0, 90)
                opt_count = _compute_sofa_area(opt_path, grid_x, grid_y)
                if opt_count > best_count:
                    best_count = opt_count
                    best_path = opt_path.copy()
        except Exception:
            pass
    
    return best_path

# EVOLVE-BLOCK-END
