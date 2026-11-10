# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
import time

def _base_corridor_points(u, v):
    """
    Check if a point (u, v) is inside the base L-shaped corridor.
    Base corridor: union of {0 <= u <= 1, -3 <= v <= 1} and {-3 <= u <= 1, 0 <= v <= 1}
    """
    in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    return in_rect1 | in_rect2

def _compute_sofa_area(path, sofa_x, sofa_y):
    """
    Compute the area of the sofa (intersection of all corridor placements).
    path: (n, 3) array of (x, y, a) where a is in degrees.
    sofa_x, sofa_y: 1D arrays of grid point coordinates.
    """
    n = len(path)
    # Precompute sofa grid
    X, Y = np.meshgrid(sofa_x, sofa_y, indexing='ij')
    
    # For each placement, check which sofa points are inside
    # A point is in the sofa if it's inside ALL placements
    inside_all = np.ones((len(sofa_x), len(sofa_y)), dtype=bool)
    
    for i in range(n):
        x, y, a_deg = path[i]
        a_rad = np.deg2rad(a_deg)
        
        # Rotation matrix R(-a)
        cos_a = np.cos(-a_rad)
        sin_a = np.sin(-a_rad)
        
        # Transform sofa points to local frame: w_local = R(-a)(w - (x,y))
        # w = (X, Y), so w - (x,y) = (X-x, Y-y)
        # R(-a) = [[cos(-a), -sin(-a)], [sin(-a), cos(-a)]]
        # But we want R(-a) applied to (w - (x,y))
        # Actually, the problem says: R(-a)(w - (x, y)) lies in the base
        # Let me re-read: "A point w is inside placement (x, y, a) if R(-a)(w - (x, y)) lies in the base"
        # R is rotation. So R(-a) is rotation by -a degrees.
        
        # Let's compute: u = X - x, v = Y - y
        # Then rotate by -a: 
        # u_local = cos(-a)*u - sin(-a)*v = cos(a)*u + sin(a)*v
        # v_local = sin(-a)*u + cos(-a)*v = -sin(a)*u + cos(a)*v
        
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        u = X - x
        v = Y - y
        
        u_local = cos_a * u + sin_a * v
        v_local = -sin_a * u + cos_a * v
        
        # Check if (u_local, v_local) is in base corridor
        in_rect1 = (u_local >= 0) & (u_local <= 1) & (v_local >= -3) & (v_local <= 1)
        in_rect2 = (u_local >= -3) & (u_local <= 1) & (v_local >= 0) & (v_local <= 1)
        in_base = in_rect1 | in_rect2
        
        inside_all &= in_base
    
    # Count points inside all placements
    count = np.sum(inside_all)
    area = count * 0.0001
    return area

def _build_dense_path(control_points, n_steps=200):
    """
    Build a dense path from control points using cubic B-spline interpolation.
    control_points: (m, 3) array of (x, y, a) control points.
    n_steps: number of steps to interpolate.
    """
    m = len(control_points)
    if m < 2:
        return control_points
    
    # Create parameter values for control points
    t = np.linspace(0, 1, m)
    
    # Interpolate each coordinate separately
    try:
        cs_x = CubicSpline(t, control_points[:, 0])
        cs_y = CubicSpline(t, control_points[:, 1])
        cs_a = CubicSpline(t, control_points[:, 2])
        
        t_dense = np.linspace(0, 1, n_steps)
        x_dense = cs_x(t_dense)
        y_dense = cs_y(t_dense)
        a_dense = cs_a(t_dense)
        
        dense_path = np.column_stack([x_dense, y_dense, a_dense])
        return dense_path
    except Exception:
        # Fallback to linear interpolation
        t_dense = np.linspace(0, 1, n_steps)
        x_dense = np.interp(t_dense, t, control_points[:, 0])
        y_dense = np.interp(t_dense, t, control_points[:, 1])
        a_dense = np.interp(t_dense, t, control_points[:, 2])
        dense_path = np.column_stack([x_dense, y_dense, a_dense])
        return dense_path

def _objective(params, sofa_x, sofa_y, n_steps=100):
    """
    Objective function for optimization: negative sofa area.
    params: flattened array of control points.
    """
    m = len(sofa_x)  # This is wrong, let me fix
    # Actually, we need to know the number of control points
    # Let's use a fixed number
    n_ctrl = 6
    params = params.reshape(n_ctrl, 3)
    
    dense_path = _build_dense_path(params, n_steps)
    area = _compute_sofa_area(dense_path, sofa_x, sofa_y)
    return -area

def moving_sofa_2d() -> np.ndarray:
    """
    The 2D moving-sofa problem: find the largest-area rigid shape that can be moved
    through an L-shaped corridor of width 1 that makes a single 90-degree turn.
    
    Uses cubic B-spline parameterization with SLSQP optimization.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    # Sofa grid
    sofa_x = np.arange(-3, 1.5, 0.01)
    sofa_y = np.arange(-1.5, 1.5, 0.01)
    
    # Initial guess for control points
    # The path should go from (0,0,0) to (-2,0,90)
    # Let's start with a reasonable initial path
    n_ctrl = 6
    init_params = np.array([
        [0.0, 0.0, 0.0],
        [-0.3, 0.2, 20.0],
        [-0.6, 0.3, 45.0],
        [-0.9, 0.3, 60.0],
        [-1.3, 0.2, 75.0],
        [-1.7, 0.0, 90.0],
    ], dtype=np.float64).flatten()
    
    best_params = init_params.copy()
    best_area = _compute_sofa_area(_build_dense_path(init_params.reshape(n_ctrl, 3), 100), sofa_x, sofa_y)
    
    # Define bounds for each parameter
    # x: roughly -2 to 0, y: -1 to 1, a: 0 to 90
    bounds = [
        (-2.0, 0.0),  # x0
        (-1.0, 1.0),  # y0
        (0.0, 90.0),  # a0
        (-2.0, 0.0),  # x1
        (-1.0, 1.0),  # y1
        (0.0, 90.0),  # a1
        (-2.0, 0.0),  # x2
        (-1.0, 1.0),  # y2
        (0.0, 90.0),  # a2
        (-2.0, 0.0),  # x3
        (-1.0, 1.0),  # y3
        (0.0, 90.0),  # a3
        (-2.0, 0.0),  # x4
        (-1.0, 1.0),  # y4
        (0.0, 90.0),  # a4
        (-2.0, 0.0),  # x5
        (-1.0, 1.0),  # y5
        (0.0, 90.0),  # a5
    ]
    
    # Optimization
    try:
        result = minimize(
            _objective,
            init_params,
            args=(sofa_x, sofa_y, 100),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-6}
        )
        
        if result.success and result.fun < -best_area:
            best_params = result.x
            best_area = -result.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        # Not enough time for more, return current best
        dense_path = _build_dense_path(best_params.reshape(n_ctrl, 3), 200)
        # Add endpoints
        start_point = np.array([[0.0, 0.0, 0.0]])
        end_point = np.array([[-2.0, 0.0, 90.0]])
        full_path = np.vstack([start_point, dense_path, end_point])
        return full_path
    
    # Try with more control points if time allows
    n_ctrl2 = 8
    init_params2 = np.array([
        [0.0, 0.0, 0.0],
        [-0.2, 0.1, 12.0],
        [-0.4, 0.25, 28.0],
        [-0.6, 0.3, 45.0],
        [-0.8, 0.3, 58.0],
        [-1.1, 0.25, 70.0],
        [-1.5, 0.1, 82.0],
        [-1.8, 0.0, 90.0],
    ], dtype=np.float64).flatten()
    
    best_params2 = init_params2.copy()
    best_area2 = _compute_sofa_area(_build_dense_path(init_params2.reshape(n_ctrl2, 3), 100), sofa_x, sofa_y)
    
    bounds2 = []
    for i in range(n_ctrl2):
        bounds2.append((-2.0, 0.0))  # x
        bounds2.append((-1.0, 1.0))  # y
        bounds2.append((0.0, 90.0))  # a
    
    try:
        result2 = minimize(
            lambda p: _objective(p, sofa_x, sofa_y, 100),
            init_params2,
            method='SLSQP',
            bounds=bounds2,
            options={'maxiter': 50, 'ftol': 1e-6}
        )
        
        if result2.success and result2.fun < -best_area2:
            best_params2 = result2.x
            best_area2 = -result2.fun
    except Exception:
        pass
    
    # Choose the better result
    if best_area2 > best_area:
        best_params = best_params2
        n_ctrl = n_ctrl2
    
    # Final dense path with 200 steps
    dense_path = _build_dense_path(best_params.reshape(n_ctrl, 3), 200)
    
    # Add endpoints
    start_point = np.array([[0.0, 0.0, 0.0]])
    end_point = np.array([[-2.0, 0.0, 90.0]])
    full_path = np.vstack([start_point, dense_path, end_point])
    
    return full_path

# EVOLVE-BLOCK-END
