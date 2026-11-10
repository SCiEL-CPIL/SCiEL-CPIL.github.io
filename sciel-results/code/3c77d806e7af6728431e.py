# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import math


def _evaluate_path(path: np.ndarray) -> float:
    """
    Evaluate the raw value (area estimate) for a given path.
    The evaluator prepends (0,0,0) and appends (-2,0,90), then densifies.
    We replicate this to compute the intersection area.
    
    Returns:
        raw_value: number of sofa points * 0.0001
    """
    # Prepend and append
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([-2.0, 0.0, 90.0])
    
    # Combine path
    all_points = np.vstack([start, path, end])
    
    # Densify: between consecutive rows, add linear interpolations in m = max(1, ceil(|a2 - a1|)) steps
    densified = [all_points[0]]
    for i in range(len(all_points) - 1):
        p1 = all_points[i]
        p2 = all_points[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        if m == 1:
            densified.append(p2)
        else:
            for k in range(1, m + 1):
                t = k / m
                interp = p1 + t * (p2 - p1)
                densified.append(interp)
    
    densified = np.array(densified)
    
    # Sofa grid points
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    
    # For each placement, check which points are inside
    # Base corridor: union of {0 <= u <= 1, -3 <= v <= 1} and {-3 <= u <= 1, 0 <= v <= 1}
    # A point w is inside placement (x, y, a) if R(-a)(w - (x, y)) lies in the base
    
    n_points = len(points)
    inside_all = np.ones(n_points, dtype=bool)
    
    for (x, y, a) in densified:
        a_rad = np.radians(a)
        cos_a = np.cos(-a_rad)
        sin_a = np.sin(-a_rad)
        
        # Transform points to local frame
        dx = points[:, 0] - x
        dy = points[:, 1] - y
        
        u = cos_a * dx - sin_a * dy
        v = sin_a * dx + cos_a * dy
        
        # Check if (u, v) is in base corridor
        # Base: (0 <= u <= 1 and -3 <= v <= 1) or (-3 <= u <= 1 and 0 <= v <= 1)
        in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        in_base = in_rect1 | in_rect2
        
        inside_all &= in_base
        
        # Early exit if no points are inside
        if not np.any(inside_all):
            break
    
    return np.sum(inside_all) * 0.0001


def _objective(params: np.ndarray) -> float:
    """
    Objective function for optimization. Returns negative area (to minimize).
    params: control points for the spline.
    """
    # params shape: (n_ctrl, 2) for x,y of control points, and n_ctrl angles
    n_ctrl = len(params) // 3
    
    # Extract x, y, angle control points
    ctrl_xy = params[:2 * n_ctrl].reshape(n_ctrl, 2)
    ctrl_angles = params[2 * n_ctrl:3 * n_ctrl]
    
    # Create spline for x(t), y(t), angle(t) where t in [0, 1]
    try:
        t = np.linspace(0, 1, n_ctrl)
        cs_x = CubicSpline(t, ctrl_xy[:, 0], bc_type='natural')
        cs_y = CubicSpline(t, ctrl_xy[:, 1], bc_type='natural')
        cs_a = CubicSpline(t, ctrl_angles, bc_type='natural')
    except Exception:
        return 1e10  # Invalid spline
    
    # Sample the spline at several points to get a discrete path
    n_samples = 10
    t_samples = np.linspace(0, 1, n_samples)
    
    # Get x, y, angle at sample points
    x_vals = cs_x(t_samples)
    y_vals = cs_y(t_samples)
    a_vals = cs_a(t_samples)
    
    # Create path
    path = np.column_stack([x_vals, y_vals, a_vals])
    
    # Evaluate
    area = _evaluate_path(path)
    
    # Return negative area (we want to maximize)
    return -area


def moving_sofa_2d() -> np.ndarray:
    """
    The 2D moving-sofa problem: find the largest-area rigid shape that can be moved
    through an L-shaped corridor of width 1 that makes a single 90-degree turn.
    
    Uses a cubic B-spline parameterized by arc length to represent the path,
    optimized via Nelder-Mead.
    
    Returns:
        path: np.ndarray of shape (n, 3); each row is (x, y, angle_degrees)
    """
    # Initial guess: smooth path from near start to near end
    # Start angle ~15, end angle ~86 (interior points)
    n_ctrl = 5
    
    # Initial control points
    x0 = np.array([-0.1, -0.2, -0.35, -0.5, -0.65])
    y0 = np.array([0.1, 0.2, 0.3, 0.25, 0.3])
    a0 = np.array([15.0, 30.0, 45.0, 65.0, 85.0])
    
    params0 = np.concatenate([x0, y0, a0])
    
    # Optimize using Nelder-Mead
    result = minimize(
        _objective,
        params0,
        method='Nelder-Mead',
        options={
            'maxiter': 200,
            'xatol': 1e-4,
            'fatol': 1e-6,
            'adaptive': True
        }
    )
    
    params_opt = result.x
    n_ctrl_opt = len(params_opt) // 3
    ctrl_xy = params_opt[:2 * n_ctrl_opt].reshape(n_ctrl_opt, 2)
    ctrl_angles = params_opt[2 * n_ctrl_opt:3 * n_ctrl_opt]
    
    # Create spline
    t = np.linspace(0, 1, n_ctrl_opt)
    cs_x = CubicSpline(t, ctrl_xy[:, 0], bc_type='natural')
    cs_y = CubicSpline(t, ctrl_xy[:, 1], bc_type='natural')
    cs_a = CubicSpline(t, ctrl_angles, bc_type='natural')
    
    # Sample the spline to get the final path
    n_samples = 8
    t_samples = np.linspace(0, 1, n_samples)
    
    x_vals = cs_x(t_samples)
    y_vals = cs_y(t_samples)
    a_vals = cs_a(t_samples)
    
    path = np.column_stack([x_vals, y_vals, a_vals])
    
    # Ensure all values are finite
    if not np.all(np.isfinite(path)):
        # Fallback to baseline
        path = np.array(
            [
                [-0.1, 0.1, 15.0],
                [-0.2, 0.2, 20.0],
                [-0.3, 0.3, 34.0],
                [-0.3, 0.3, 40.0],
                [-0.4, 0.3, 50.0],
                [-0.5, 0.2, 55.0],
                [-0.6, 0.2, 75.0],
                [-0.7, 0.3, 86.0],
            ],
            dtype=np.float64,
        )
    
    return path


# EVOLVE-BLOCK-END
