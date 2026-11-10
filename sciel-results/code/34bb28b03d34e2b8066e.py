# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def moving_sofa_2d() -> np.ndarray:
    """
    Moves a sofa through an L-shaped corridor.
    Strategy: Optimize the path geometry using a parametric representation 
    (polynomials for x(a) and y(a)) and Nelder-Mead optimization.
    """
    
    # 1. Setup Grid
    x_grid = np.arange(-3, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid)
    P = np.column_stack((X.ravel(), Y.ravel()))
    N_points = P.shape[0]
    wx = P[:, 0]
    wy = P[:, 1]
    
    # Start and End states
    start_state = np.array([0.0, 0.0, 0.0])
    end_state = np.array([-2.0, 0.0, 90.0])
    
    # We model the path as:
    # x(a) = c_x0 + c_x1 * a + c_x2 * a^2 + c_x3 * a^3
    # y(a) = c_y0 + c_y1 * a + c_y2 * a^2 + c_y3 * a^3
    # where a is in degrees [0, 90]
    
    # Constraints:
    # x(0) = 0, y(0) = 0  => c_x0 = 0, c_y0 = 0
    # x(90) = -2, y(90) = 0
    # So we only optimize c_x1, c_x2, c_x3 and c_y1, c_y2, c_y3.
    # Total 6 parameters.
    
    def compute_path_points(coeffs):
        """
        coeffs: [cx1, cx2, cx3, cy1, cy2, cy3]
        Returns list of [x, y, a] for a in 0..90 step 1
        """
        cx1, cx2, cx3, cy1, cy2, cy3 = coeffs
        a_vals = np.arange(0, 91, 1.0)
        
        # x(a) = cx1*a + cx2*a^2 + cx3*a^3
        x_vals = cx1 * a_vals + cx2 * a_vals**2 + cx3 * a_vals**3
        # y(a) = cy1*a + cy2*a^2 + cy3*a^3
        y_vals = cy1 * a_vals + cy2 * a_vals**2 + cy3 * a_vals**3
        
        # Enforce end point exactly (numerical stability)
        x_vals[-1] = -2.0
        y_vals[-1] = 0.0
        
        # Create path array
        path = np.column_stack((x_vals, y_vals, a_vals))
        return path

    def evaluate_area(coeffs):
        """
        Computes the area of the sofa for the path defined by coeffs.
        """
        path = compute_path_points(coeffs)
        
        mask = np.ones(N_points, dtype=bool)
        
        # We can vectorize over the path steps? No, mask accumulates.
        # But we can process all points for one step at once.
        
        for i in range(len(path)):
            x, y, a_deg = path[i]
            a_rad = np.deg2rad(a_deg)
            
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = wx - x
            dy = wy - y
            
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a
            
            in_r1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_r2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            
            in_corridor = in_r1 | in_r2
            
            mask &= in_corridor
            
            # Early exit
            if not np.any(mask):
                return 0.0
        
        return float(np.sum(mask)) * 0.0001

    # Initial guess
    # Based on current best score ~1.4, the path is roughly:
    # x goes 0 -> -2, y goes 0 -> 0 with a bump.
    # Let's start with a simple curve.
    # x(t) = -2*t, y(t) = 0.5 * sin(pi*t) where t = a/90
    # This is not a polynomial in a, but we can fit or just use random init.
    
    initial_coeffs = np.array([
        -2.0/90.0, 0, 0, # x linear
        0, 0, 0           # y linear
    ])
    
    # A better initial guess based on the "S" curve of the sofa move
    # x should stay near 0 for a while, then move to -2.
    # y should bump up and down.
    
    # Let's use a few random restarts with Nelder-Mead
    best_score = 0.0
    best_coeffs = initial_coeffs
    
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Number of restarts
    n_restarts = 20
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial guess
        if r == 0:
            coeffs = initial_coeffs
        else:
            # Random perturbation around a plausible region
            # x range: 0 to -2
            # y range: -0.5 to 0.5
            cx1 = np.random.uniform(-0.05, 0.0)
            cx2 = np.random.uniform(-0.001, 0.001)
            cx3 = np.random.uniform(-0.0001, 0.0001)
            cy1 = np.random.uniform(-0.02, 0.02)
            cy2 = np.random.uniform(-0.002, 0.002)
            cy3 = np.random.uniform(-0.0002, 0.0002)
            coeffs = np.array([cx1, cx2, cx3, cy1, cy2, cy3])
        
        # Optimize
        # Nelder-Mead is robust for small dimensional problems
        try:
            res = scipy.optimize.minimize(
                lambda c: -evaluate_area(c), 
                coeffs, 
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-8}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_coeffs = res.x
        except Exception as e:
            pass
    
    # Convert best_coeffs to the required output format
    # The evaluator expects a list of points.
    # We can return the densified path from compute_path_points
    # But we need to ensure the format matches: list of [x, y, a]
    
    final_path = compute_path_points(best_coeffs)
    
    # The evaluator adds start and end.
    # Our path includes start (0,0,0) and end (-2,0,90).
    # The problem says: "The evaluator puts (0, 0, 0) before your rows and (-2, 0, 90) after them."
    # So we should NOT include the start and end points in our output?
    # "returning a float array of shape (n, 3), n >= 1"
    # If I return the full path including start/end, the evaluator will add ANOTHER start/end.
    # Let's re-read: "The evaluator puts (0, 0, 0) before your rows and (-2, 0, 90) after them."
    # This implies my rows are the *intermediate* steps.
    # However, if I include (0,0,0) as the first row, the evaluator will have:
    # (0,0,0) [eval] -> (0,0,0) [me] -> ...
    # This is redundant but likely harmless? Or does it break interpolation?
    # Interpolation between identical points is fine.
    # But to be safe and efficient, I should return the points *between* start and end.
    # My compute_path_points returns a=0 to 90.
    # I should return rows for a=1 to 89.
    
    # Filter out a=0 and a=90
    mask_intermediate = (final_path[:, 2] > 0) & (final_path[:, 2] < 90)
    result_path = final_path[mask_intermediate]
    
    # If result is empty (shouldn't happen), return a dummy
    if len(result_path) == 0:
        result_path = np.array([[-1.0, 0.0, 45.0]])
        
    return result_path

# EVOLVE-BLOCK-END
