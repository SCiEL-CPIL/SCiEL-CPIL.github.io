# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    """
    Strategy: Direct Numerical Optimization of the Path Objective.
    
    The current program uses the analytic Pogossnitsky-Scheybman curve.
    The evaluator uses a discrete grid and linear interpolation between waypoints.
    This discrete nature means the continuous optimum is not the discrete optimum.
    Furthermore, the "sofa" is defined as the intersection of all placements.
    
    We will formulate the problem as a direct optimization over a set of waypoints.
    Objective: Maximize the number of grid points in the intersection of all placements.
    
    To make this tractable:
    1. We parameterize the path by a set of N waypoints (x_i, y_i, theta_i).
    2. We enforce theta_i to be strictly increasing (monotonic rotation) to avoid
       degenerate loops, or simply fix theta_i to be evenly spaced and optimize (x_i, y_i).
       Given the problem structure, fixing theta_i to be evenly spaced from 0 to 90
       is a strong prior that likely holds near the optimum.
    3. The objective function is the count of grid points that are inside ALL placements
       (including the interpolated ones).
    4. We use a derivative-free optimizer (Nelder-Mead or Powell) because the objective
       is non-smooth (counting function).
    
    To speed up the objective evaluation:
    - Pre-compute the grid.
    - Vectorize the check for a single waypoint.
    - The bottleneck is the number of interpolated waypoints.
      If we use N=10 waypoints, there are 9 intervals.
      Each interval has m = ceil(|dtheta|) steps. If dtheta=10, m=10.
      Total placements = 10 (waypoints) + 9*10 (interpolated) = 100 placements.
      Checking 100 placements against ~135k grid points (450x300) is 13.5M operations.
      With numpy, this is fast (~0.1s).
      Nelder-Mead might need 100-200 evaluations. Total time ~20-40s. Feasible.
    
    We will start from the analytic path as the initial guess.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    # Grid definition
    u_vals = np.arange(-3, 1.5, 0.01)
    v_vals = np.arange(-1.5, 1.5, 0.01)
    U, V = np.meshgrid(u_vals, v_vals, indexing='ij')
    grid_x = U.ravel()
    grid_y = V.ravel()
    n_points = len(grid_x)

    # Base corridor mask in local coordinates (u, v)
    # Base: {0 <= u <= 1, -3 <= v <= 1} U {-3 <= u <= 1, 0 <= v <= 1}
    # Note: The grid points are in the global frame.
    # For a placement (x, y, a), a point p is inside if R(-a)(p - (x,y)) is in Base.
    
    def is_inside_base(u, v):
        m1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        m2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        return m1 | m2

    # Pre-compute base mask for the grid points? 
    # No, because the grid points move relative to the base frame when the placement changes.
    # However, we can vectorize the transformation.
    
    def compute_intersection_count(path_params):
        """
        path_params: array of shape (N, 2) containing (x_i, y_i) for N waypoints.
        Thetas are fixed to be linspace(0, 90, N).
        Returns the count of grid points inside all placements.
        """
        N = len(path_params)
        xs = path_params[:, 0]
        ys = path_params[:, 1]
        thetas_deg = np.linspace(0, 90, N)
        
        # Generate all placements (waypoints + interpolations)
        # The evaluator inserts linear interpolations.
        # We must replicate this exactly.
        
        placements = []
        for i in range(N - 1):
            x1, y1, a1 = xs[i], ys[i], thetas_deg[i]
            x2, y2, a2 = xs[i+1], ys[i+1], thetas_deg[i+1]
            
            m = max(1, int(np.ceil(abs(a2 - a1))))
            
            # The evaluator says: "adds linear interpolations ... in m equal steps"
            # This usually means m-1 intermediate points, or m points total?
            # "in m equal steps" typically implies dividing the interval into m segments,
            # which yields m+1 points including endpoints, or m-1 intermediates.
            # Let's look at the standard interpretation:
            # If m=1, no intermediates.
            # If m=2, 1 intermediate.
            # The code likely does:
            # for k in range(1, m): ...
            # So there are m-1 intermediate points.
            
            for k in range(1, m):
                t = k / m
                x_int = x1 + t * (x2 - x1)
                y_int = y1 + t * (y2 - y1)
                a_int = a1 + t * (a2 - a1)
                placements.append((x_int, y_int, a_int))
        
        # Add the waypoints themselves
        # The evaluator puts start (0,0,0) and end (-2,0,90) and our rows.
        # Our rows are the waypoints.
        # So the full list of placements is:
        # (0,0,0) -> [interpolations to wp1] -> wp1 -> [interpolations to wp2] -> wp2 ... -> wpN -> [interpolations to end] -> (-2,0,90)
        
        # Wait, the task says: "The evaluator puts (0, 0, 0) before your rows and (-2, 0, 90) after them."
        # And "Between consecutive rows it adds linear interpolations".
        # So the sequence is:
        # P0 = (0,0,0)
        # P1 = wp1
        # ...
        # PN = wpN
        # P_{N+1} = (-2,0,90)
        
        # We need to check all points in this sequence, including the interpolated ones between P0-P1, P1-P2, ..., PN-P_{N+1}.
        
        all_placements = []
        
        # Segment P0 -> wp1
        x1, y1, a1 = 0.0, 0.0, 0.0
        x2, y2, a2 = xs[0], ys[0], thetas_deg[0]
        m = max(1, int(np.ceil(abs(a2 - a1))))
        for k in range(1, m):
            t = k / m
            all_placements.append((x1 + t*(x2-x1), y1 + t*(y2-y1), a1 + t*(a2-a1)))
        all_placements.append((x2, y2, a2)) # wp1
        
        # Intermediate segments
        for i in range(N - 1):
            x1, y1, a1 = xs[i], ys[i], thetas_deg[i]
            x2, y2, a2 = xs[i+1], ys[i+1], thetas_deg[i+1]
            m = max(1, int(np.ceil(abs(a2 - a1))))
            for k in range(1, m):
                t = k / m
                all_placements.append((x1 + t*(x2-x1), y1 + t*(y2-y1), a1 + t*(a2-a1)))
            all_placements.append((x2, y2, a2)) # wp_{i+1}
        
        # Segment wpN -> End
        x1, y1, a1 = xs[-1], ys[-1], thetas_deg[-1]
        x2, y2, a2 = -2.0, 0.0, 90.0
        m = max(1, int(np.ceil(abs(a2 - a1))))
        for k in range(1, m):
            t = k / m
            all_placements.append((x1 + t*(x2-x1), y1 + t*(y2-y1), a1 + t*(a2-a1)))
        all_placements.append((x2, y2, a2)) # End point
        
        # Now check intersection
        # Initialize mask with True
        valid_mask = np.ones(n_points, dtype=bool)
        
        # For each placement, compute which points are inside, and AND with valid_mask
        # If valid_mask becomes all False, we can break early.
        
        for x, y, a_deg in all_placements:
            a_rad = np.radians(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            # Transform grid points to local frame
            # q = R(-a)(p - c)
            # u = cos_a * (px - x) + sin_a * (py - y)
            # v = -sin_a * (px - x) + cos_a * (py - y)
            
            px = grid_x - x
            py = grid_y - y
            
            u = cos_a * px + sin_a * py
            v = -sin_a * px + cos_a * py
            
            # Check base corridor condition
            inside = is_inside_base(u, v)
            
            valid_mask &= inside
            
            # Early exit if no points remain
            if not np.any(valid_mask):
                return 0
        
        return np.count_nonzero(valid_mask)

    # Initial guess: Analytic path
    # x(t) = -1 + cos(t) - sin(t)
    # y(t) = -1 + sin(t) + cos(t)
    # for t in [0, pi/2]
    
    # We need to choose N.
    # If N is small, optimization is easier but path is coarser.
    # If N is large, path is finer but optimization is harder.
    # Let's try N=10.
    
    N = 10
    thetas_deg_init = np.linspace(0, 90, N)
    thetas_rad_init = np.radians(thetas_deg_init)
    
    x_init = -1.0 + np.cos(thetas_rad_init) - np.sin(thetas_rad_init)
    y_init = -1.0 + np.sin(thetas_rad_init) + np.cos(thetas_rad_init)
    
    # Note: The first waypoint should be close to (0,0) and the last close to (-2,0)?
    # The analytic path starts at t=0: x=-1+1-0=0, y=-1+0+1=0. So (0,0).
    # Ends at t=pi/2: x=-1+0-1=-2, y=-1+1+0=0. So (-2,0).
    # So the analytic path matches the start and end points perfectly.
    
    initial_params = np.column_stack((x_init, y_init))
    
    # Objective function to minimize (negative of count)
    def objective(params):
        if time.time() > deadline - 10:
            # Return a bad value to stop optimization
            return 1e9
        try:
            count = compute_intersection_count(params)
            return -count
        except Exception:
            return 1e9

    # Use Nelder-Mead
    # Options:
    # maxiter: limit iterations
    # xatol, fatol: tolerances
    
    best_params = initial_params
    best_score = -compute_intersection_count(initial_params)
    
    try:
        res = minimize(
            objective, 
            initial_params, 
            method='Nelder-Mead',
            options={
                'maxiter': 200,
                'xatol': 1e-4,
                'fatol': 1e-4,
                'adaptive': True
            }
        )
        
        if time.time() < deadline - 5:
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
            else:
                # Keep initial if it was better
                best_params = initial_params
                best_score = -compute_intersection_count(initial_params)
                
    except Exception as e:
        pass

    # Construct the final path array
    # The function must return an array of shape (n, 3)
    # Rows are (x, y, a)
    
    final_xs = best_params[:, 0]
    final_ys = best_params[:, 1]
    final_thetas = np.linspace(0, 90, N)
    
    path = np.column_stack((final_xs, final_ys, final_thetas))
    
    # Ensure finite
    if not np.all(np.isfinite(path)):
        # Fallback to analytic
        path = np.column_stack((x_init, y_init, thetas_deg_init))
        
    return path

# EVOLVE-BLOCK-END
