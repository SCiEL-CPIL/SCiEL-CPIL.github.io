# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _sofa_grid():
    """Create the sofa point grid once."""
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    return gx.ravel(), gy.ravel()


def _corridor_mask(x, y, a_deg, px, py):
    """
    Boolean mask: True where point (px, py) is inside the L-shaped corridor
    with inner corner at (x, y), rotated by a_deg degrees.
    
    Base corridor (a=0): union of {0<=u<=1, -3<=v<=1} and {-3<=u<=1, 0<=v<=1}
    """
    rad = np.deg2rad(a_deg)
    c = np.cos(rad)
    s = np.sin(rad)
    # R(-a) applied to (w - (x,y)):
    # u = c*(px-x) + s*(py-y)
    # v = -s*(px-x) + c*(py-y)
    dx = px - x
    dy = py - y
    u = c * dx + s * dy
    v = -s * dx + c * dy
    
    # Arm 1: 0<=u<=1, -3<=v<=1
    m1 = (u >= -1e-9) & (u <= 1 + 1e-9) & (v >= -3 - 1e-9) & (v <= 1 + 1e-9)
    # Arm 2: -3<=u<=1, 0<=v<=1
    m2 = (u >= -3 - 1e-9) & (u <= 1 + 1e-9) & (v >= -1e-9) & (v <= 1 + 1e-9)
    
    return m1 | m2


def _compute_sofa_area(path, px, py):
    """
    Given a full path (with start (0,0,0) and end (-2,0,90) prepended/appended,
    and densified), compute the number of sofa points inside every placement.
    """
    n_points = len(px)
    # Start with all points as candidates
    inside = np.ones(n_points, dtype=bool)
    
    for i in range(len(path)):
        x, y, a = path[i]
        mask = _corridor_mask(x, y, a, px, py)
        inside = inside & mask
        # Early exit if no points remain
        if not np.any(inside):
            break
    
    return np.count_nonzero(inside) * 0.0001


def _densify_path(rows):
    """
    Densify the path: between consecutive rows, add linear interpolations
    in m = max(1, ceil(|a2-a1|)) equal steps.
    """
    result = [rows[0]]
    for i in range(len(rows) - 1):
        x1, y1, a1 = rows[i]
        x2, y2, a2 = rows[i + 1]
        m = max(1, int(np.ceil(abs(a2 - a1))))
        for k in range(1, m + 1):
            t = k / m
            result.append([x1 + t * (x2 - x1),
                           y1 + t * (y2 - y1),
                           a1 + t * (a2 - a1)])
    return np.array(result)


def _full_path_from_params(params):
    """
    Convert spline parameters to a full densified path including
    start (0,0,0) and end (-2,0,90).
    
    params: array of shape (n_ctrl, 3) giving control points (x, y, a)
    for a cubic Bezier-like path. We use a simple approach:
    the path goes from (0,0,0) to (-2,0,90) through the control points.
    """
    n_ctrl = len(params)
    if n_ctrl == 0:
        # Direct path
        raw = np.array([[0, 0, 0], [-2, 0, 90]])
    else:
        # Build path: start, then control points, then end
        raw = np.vstack([np.array([0, 0, 0.0]), params, np.array([-2, 0, 90.0])])
    
    return _densify_path(raw)


def _objective(params, px, py):
    """Negative area (we minimize negative area = maximize area)."""
    path = _full_path_from_params(params)
    area = _compute_sofa_area(path, px, py)
    return -area


def moving_sofa_2d() -> np.ndarray:
    """
    Optimize the corridor path to maximize the sofa area.
    
    Uses a parametric representation with cubic Bezier-like control points,
    optimized via differential evolution followed by local refinement.
    """
    deadline = time.time() + 580  # 20s safety margin
    
    px, py = _sofa_grid()
    
    # Start with the baseline path as initial guess
    baseline = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ])
    
    # Evaluate baseline
    full_baseline = _densify_path(np.vstack([np.array([0,0,0.0]), baseline, np.array([-2,0,90.0])]))
    best_area = _compute_sofa_area(full_baseline, px, py)
    best_params = baseline.copy()
    
    # Strategy 1: Try different numbers of control points with differential evolution
    for n_ctrl in [3, 4, 5]:
        if time.time() > deadline - 120:
            break
        
        # Bounds for control points
        # x: roughly -1.5 to 0.5
        # y: roughly -0.5 to 1.0
        # a: roughly 5 to 85
        bounds = [(-1.5, 0.5), (-0.5, 1.0), (5.0, 85.0)] * n_ctrl
        
        def obj_flat(flat_params):
            if time.time() > deadline - 10:
                return 0
            params = flat_params.reshape(-1, 3)
            return _objective(params, px, py)
        
        # Use differential evolution with limited iterations
        x0 = np.zeros(n_ctrl * 3)
        # Initialize near baseline positions
        if n_ctrl <= len(baseline):
            for i in range(n_ctrl):
                x0[i*3] = baseline[i, 0]
                x0[i*3+1] = baseline[i, 1]
                x0[i*3+2] = baseline[i, 2]
        
        try:
            result = differential_evolution(
                obj_flat, bounds,
                seed=42,
                maxiter=30,
                tol=0.01,
                popsize=15,
                polish=False,
                update_interval=5,
            )
        except Exception:
            continue
        
        if result.success or hasattr(result, 'x'):
            params = result.x.reshape(-1, 3)
            # Ensure angles are in reasonable range
            params[:, 2] = np.clip(params[:, 2], 5, 85)
            
            area = -_objective(params, px, py)
            if area > best_area:
                best_area = area
                best_params = params.copy()
    
    # Strategy 2: Local refinement with Nelder-Mead on the best params
    if time.time() < deadline - 60 and len(best_params) > 0:
        try:
            result = minimize(
                lambda p: _objective(p.reshape(-1, 3), px, py),
                best_params.ravel(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 0.01, 'fatol': 1e-6}
            )
            if hasattr(result, 'x'):
                params = result.x.reshape(-1, 3)
                params[:, 2] = np.clip(params[:, 2], 5, 85)
                area = -_objective(params, px, py)
                if area > best_area:
                    best_area = area
                    best_params = params.copy()
        except Exception:
            pass
    
    # Strategy 3: Try a few random perturbations of the baseline
    if time.time() < deadline - 30:
        rng = np.random.RandomState(123)
        for trial in range(50):
            if time.time() > deadline - 15:
                break
            # Random path with 5-8 points
            n = rng.randint(4, 9)
            # Angles roughly increasing from ~10 to ~80
            angles = np.sort(rng.uniform(8, 82, n))
            xs = -0.05 - 0.08 * angles / 90 + rng.uniform(-0.1, 0.1, n)
            ys = 0.1 + 0.2 * np.sin(angles / 90 * np.pi / 2) + rng.uniform(-0.1, 0.1, n)
            params = np.column_stack([xs, ys, angles])
            
            area = -_objective(params, px, py)
            if area > best_area:
                best_area = area
                best_params = params.copy()
    
    # Final: return best_params
    if len(best_params) == 0:
        # Fallback to baseline
        return baseline
    
    return best_params


# EVOLVE-BLOCK-END
