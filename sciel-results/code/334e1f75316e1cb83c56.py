# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _corridor_mask(x, y, a_deg, grid_x, grid_y):
    """
    Return a boolean array of shape (450, 300) where True means the grid point is inside
    the L-shaped corridor placement with inner corner at (x, y) rotated by a_deg degrees.
    
    Base corridor (a=0): union of {0<=u<=1, -3<=v<=1} and {-3<=u<=1, 0<=v<=1}.
    A point w is inside if R(-a)(w - (x,y)) lies in the base.
    """
    a = np.deg2rad(a_deg)
    ca, sa = np.cos(a), np.sin(a)
    
    # Transform grid points to corridor local frame
    # w = (grid_x, grid_y), p = w - (x, y)
    # R(-a) * p: rotation by -a
    # R(-a) = [[cos(a), sin(a)], [-sin(a), cos(a)]]
    dx = grid_x - x
    dy = grid_y - y
    
    u = ca * dx + sa * dy
    v = -sa * dx + ca * dy
    
    # Base corridor: union of two rectangles
    # Rect 1: 0 <= u <= 1, -3 <= v <= 1
    # Rect 2: -3 <= u <= 1, 0 <= v <= 1
    in_r1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    in_r2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    
    return in_r1 | in_r2


def _compute_sofa_area(path):
    """
    Given a path (n, 3) array, compute the area of the sofa (intersection of all corridor placements).
    The evaluator prepends (0,0,0) and appends (-2,0,90), and densifies with linear interpolation.
    """
    # Prepend start and append end
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([-2.0, 0.0, 90.0])
    full_path = np.vstack([start, path, end])
    
    # Densify: between consecutive rows, add linear interpolations
    # m = max(1, ceil(|a2 - a1|)) equal steps
    densified = [full_path[0]]
    for i in range(len(full_path) - 1):
        p1 = full_path[i]
        p2 = full_path[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        if m > 1:
            for j in range(1, m):
                t = j / m
                interp = p1 + t * (p2 - p1)
                densified.append(interp)
        densified.append(p2)
    
    densified = np.array(densified)
    
    # Grid points
    grid_x = np.arange(-3, 1.5, 0.01)
    grid_y = np.arange(-1.5, 1.5, 0.01)
    GX, GY = np.meshgrid(grid_x, grid_y, indexing='ij')
    
    # Initialize mask to all True
    sofa_mask = np.ones(GX.shape, dtype=bool)
    
    for x, y, a in densified:
        mask = _corridor_mask(x, y, a, GX, GY)
        sofa_mask &= mask
    
    return np.sum(sofa_mask) * 0.0001


def _bezier_path(t_vals, p0, p1, p2, p3):
    """
    Evaluate a cubic Bezier curve at parameter values t_vals (array).
    Returns (n, 2) array of (x, y) positions.
    """
    t = t_vals[:, np.newaxis]
    # Cubic Bezier: B(t) = (1-t)^3 * P0 + 3(1-t)^2 * t * P1 + 3(1-t) * t^2 * P2 + t^3 * P3
    b0 = (1 - t)**3
    b1 = 3 * (1 - t)**2 * t
    b2 = 3 * (1 - t) * t**2
    b3 = t**3
    
    # p0, p1, p2, p3 are (x, y) pairs
    px = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
    py = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
    
    return np.column_stack([px, py])


def _objective(params):
    """
    Objective function for optimization.
    params: array of 8 values: [p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y]
    These define a cubic Bezier curve for the inner corner position.
    The angle is linearly interpolated from 0 to 90 degrees.
    """
    # Unpack
    p0 = np.array([params[0], params[1]])
    p1 = np.array([params[2], params[3]])
    p2 = np.array([params[4], params[5]])
    p3 = np.array([params[6], params[7]])
    
    # Use 8 control points along the Bezier curve (t = 0, 1/7, 2/7, ..., 1)
    # But we need to exclude t=0 and t=1 since those are fixed by the evaluator
    # Actually, the evaluator prepends (0,0,0) and appends (-2,0,90).
    # Our path should be intermediate points.
    # Let's use t values in (0, 1) for our control points.
    # We'll use 8 points: t = 1/9, 2/9, ..., 8/9
    t_vals = np.linspace(1/9, 8/9, 8)
    
    # Evaluate Bezier
    positions = _bezier_path(t_vals, p0, p1, p2, p3)
    
    # Angle is linearly interpolated from 0 to 90
    # At t=0, angle=0; at t=1, angle=90
    # So at t_i, angle = 90 * t_i
    angles = 90.0 * t_vals
    
    # Build path
    path = np.column_stack([positions[:, 0], positions[:, 1], angles])
    
    # Compute area
    area = _compute_sofa_area(path)
    
    # We want to maximize area, so return negative for minimization
    return -area


def moving_sofa_2d() -> np.ndarray:
    """
    The 2D moving-sofa problem: find the largest-area rigid shape that can be moved
    through an L-shaped corridor of width 1 that makes a single 90-degree turn.
    
    Returns:
        path: np.ndarray of shape (n, 3); each row is (x, y, angle_degrees) for the
            corridor's inner corner.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second deadline
    
    # Initial guess: a smooth curve from near (0,0) to near (-2,0)
    # The inner corner starts at (0,0) with angle 0 and ends at (-2,0) with angle 90.
    # A reasonable path might be an arc or a smooth curve.
    
    # Let's start with a simple arc-like path
    # p0 = (0, 0), p3 = (-2, 0)
    # p1 and p2 are control points
    # Initial guess: p1 = (-0.5, 0.5), p2 = (-1.5, 0.5)
    x0 = np.array([0.0, 0.0, -0.5, 0.5, -1.5, 0.5, -2.0, 0.0])
    
    best_params = x0.copy()
    best_area = _compute_sofa_area(_path_from_params(best_params))
    
    # Try a few different initializations
    inits = [
        np.array([0.0, 0.0, -0.5, 0.5, -1.5, 0.5, -2.0, 0.0]),
        np.array([0.0, 0.0, -1.0, 0.0, -1.0, 0.0, -2.0, 0.0]),
        np.array([0.0, 0.0, -0.5, 1.0, -1.5, 1.0, -2.0, 0.0]),
        np.array([0.0, 0.0, -0.3, 0.3, -1.7, 0.3, -2.0, 0.0]),
        np.array([0.0, 0.0, -0.5, 0.2, -1.5, 0.2, -2.0, 0.0]),
        np.array([0.0, 0.0, -0.7, 0.7, -1.3, 0.7, -2.0, 0.0]),
    ]
    
    for x0 in inits:
        if time.time() > deadline - 60.0:
            break
        
        # Evaluate initial
        init_area = _compute_sofa_area(_path_from_params(x0))
        if init_area > best_area:
            best_area = init_area
            best_params = x0.copy()
        
        # Run L-BFGS-B
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 200, 'disp': False}
            )
            if result.fun < -best_area:
                best_area = -result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Return the best path
    path = _path_from_params(best_params)
    return path


def _path_from_params(params):
    """
    Convert 8 parameters (p0, p1, p2, p3 for Bezier) to a path array.
    """
    p0 = np.array([params[0], params[1]])
    p1 = np.array([params[2], params[3]])
    p2 = np.array([params[4], params[5]])
    p3 = np.array([params[6], params[7]])
    
    t_vals = np.linspace(1/9, 8/9, 8)
    positions = _bezier_path(t_vals, p0, p1, p2, p3)
    angles = 90.0 * t_vals
    
    return np.column_stack([positions[:, 0], positions[:, 1], angles])


# EVOLVE-BLOCK-END
