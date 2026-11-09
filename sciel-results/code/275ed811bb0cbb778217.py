import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    """
    Returns a path (n, 3) for the moving sofa problem.
    """
    start_time = time.time()
    deadline = 600.0
    
    # Grid definition
    # i = 0..449, j = 0..299
    # x_grid = -3 + 0.01i, y_grid = -1.5 + 0.01j
    # Note: np.arange(-3, 1.5, 0.01) gives 450 points.
    # np.arange(-1.5, 1.5, 0.01) gives 300 points.
    x_g = np.arange(-3.0, 1.5, 0.01)
    y_g = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_g, y_g)
    # Flatten for vectorized ops
    Wx = X.flatten()
    Wy = Y.flatten()
    N_pts = len(Wx)
    
    # Base corridor check function
    # Returns boolean array of shape (N_pts,) for a given placement (x, y, a_deg)
    def is_inside_base(x_c, y_c, a_deg):
        a_rad = np.deg2rad(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # Rotate points relative to center (x_c, y_c) by -a
        # R(-a) = [[cos, sin], [-sin, cos]]
        # u = cos*(Wx - x_c) + sin*(Wy - y_c)
        # v = -sin*(Wx - x_c) + cos*(Wy - y_c)
        dx = Wx - x_c
        dy = Wy - y_c
        
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        # Base corridor:
        # Part 1: 0 <= u <= 1, -3 <= v <= 1
        # Part 2: -3 <= u <= 1, 0 <= v <= 1
        in_p1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        in_p2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        
        return in_p1 | in_p2

    # Function to compute area for a given path
    # path: (n, 3) array of (x, y, a)
    def compute_area(path):
        # Build list of all placements
        # Start: (0, 0, 0)
        # End: (-2, 0, 90)
        # Path: path[0], ..., path[n-1]
        
        placements = []
        # Add start
        placements.append((0.0, 0.0, 0.0))
        
        prev_x, prev_y, prev_a = 0.0, 0.0, 0.0
        
        # Add path points with interpolation
        for i in range(len(path)):
            curr_x, curr_y, curr_a = path[i]
            
            # Interpolate from prev to curr
            # m = max(1, ceil(|a2 - a1|))
            da = abs(curr_a - prev_a)
            m = max(1, int(np.ceil(da)))
            
            # We need m steps. This means m+1 points including start and end?
            # "adds linear interpolations ... in m equal steps"
            # Usually this means m segments, so m+1 points.
            # But the endpoints are already in the list?
            # The evaluator adds interpolations *between* consecutive rows.
            # If we have row A and row B, we add points between them.
            # The prompt says: "Between consecutive rows it adds linear interpolations ... in m equal steps".
            # This implies the rows themselves are included, and the interpolations fill the gaps.
            # So if m=1, there are no intermediate points, just A and B.
            # If m=2, there is 1 intermediate point? Or 2 steps means 3 points?
            # "m equal steps" usually means m intervals.
            # Let's assume m intervals, so m-1 intermediate points.
            # Wait, if m=1, 1 step, 0 intermediate points.
            # Let's generate m-1 intermediate points.
            
            if m > 1:
                for k in range(1, m):
                    t = k / m
                    ix = prev_x + t * (curr_x - prev_x)
                    iy = prev_y + t * (curr_y - prev_y)
                    ia = prev_a + t * (curr_a - prev_a)
                    placements.append((ix, iy, ia))
            
            placements.append((curr_x, curr_y, curr_a))
            prev_x, prev_y, prev_a = curr_x, curr_y, curr_a
        
        # Add end
        placements.append((-2.0, 0.0, 90.0))
        
        # Check all placements
        # To save memory, we can accumulate the intersection
        inside_all = np.ones(N_pts, dtype=bool)
        
        for px, py, pa in placements:
            inside = is_inside_base(px, py, pa)
            inside_all &= inside
            # Early exit if empty
            if not np.any(inside_all):
                return 0.0
        
        return np.sum(inside_all) * 0.0001

    # Initial path guess
    # Based on the provided baseline
    init_path = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)
    
    # Objective function for minimization (we want to maximize area)
    def objective(params):
        # params is a flat array of length 3 * n
        # We need to ensure angles are monotonic? 
        # The evaluator handles interpolation. If angles are not monotonic, it might behave weirdly.
        # Let's constrain the angles to be fixed or monotonic.
        # To simplify optimization, let's fix the angles to be uniformly spaced or follow a specific curve.
        # Let's say we have N=8 points.
        # Angles: 10, 20, 30, 40, 50, 60, 70, 80.
        # We only optimize x and y.
        
        n_pts_path = 8
        # Fixed angles
        angles = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.float64)
        
        # Extract x, y
        xy = params[:2 * n_pts_path].reshape(n_pts_path, 2)
        
        # Construct path
        path = np.zeros((n_pts_path, 3))
        path[:, 0] = xy[:, 0]
        path[:, 1] = xy[:, 1]
        path[:, 2] = angles
        
        area = compute_area(path)
        return -area # Minimize negative area

    # Initial parameters
    # Extract x, y from init_path
    init_params = init_path[:, :2].flatten()
    
    # Optimization
    best_path = init_path
    best_area = compute_area(init_path)
    
    try:
        # Use Nelder-Mead
        # Limits on time
        result = minimize(
            objective, 
            init_params, 
            method='Nelder-Mead', 
            options={'maxiter': 500, 'xatol': 1e-3, 'fatol': 1e-5}
        )
        
        if result.fun < -best_area: # If we found a better area (more negative objective)
            # Reconstruct path
            n_pts_path = 8
            angles = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.float64)
            xy = result.x[:2 * n_pts_path].reshape(n_pts_path, 2)
            new_path = np.zeros((n_pts_path, 3))
            new_path[:, 0] = xy[:, 0]
            new_path[:, 1] = xy[:, 1]
            new_path[:, 2] = angles
            best_path = new_path
            best_area = -result.fun
            
    except Exception as e:
        pass # Keep initial path if optimization fails

    # Check time
    if time.time() > deadline - 10:
        return best_path

    # Try a second round with slightly different angles or more points?
    # For now, return the best found.
    
    return best_path
