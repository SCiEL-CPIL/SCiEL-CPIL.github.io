import numpy as np
import time
import sys

# Grid definition
# x: -3 to 1.5 (exclusive of 1.5? np.arange(-3, 1.5, 0.01) -> 450 points)
# y: -1.5 to 1.5 (exclusive of 1.5? np.arange(-1.5, 1.5, 0.01) -> 300 points)
# Total 135,000 points.

X, Y = np.meshgrid(np.arange(-3, 1.5, 0.01), np.arange(-1.5, 1.5, 0.01))
GRID_POINTS = np.column_stack((X.ravel(), Y.ravel()))
N_POINTS = len(GRID_POINTS)

def get_corridor_mask(x, y, a_deg):
    """
    Returns a boolean mask of shape (N_POINTS,) indicating which grid points
    are inside the corridor placement (x, y, a_deg).
    """
    a_rad = np.deg2rad(a_deg)
    cos_a = np.cos(-a_rad)
    sin_a = np.sin(-a_rad)
    
    # Translate
    dx = GRID_POINTS[:, 0] - x
    dy = GRID_POINTS[:, 1] - y
    
    # Rotate
    u = cos_a * dx - sin_a * dy
    v = sin_a * dx + cos_a * dy
    
    # Base corridor conditions
    # R1: 0 <= u <= 1, -3 <= v <= 1
    mask1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    # R2: -3 <= u <= 1, 0 <= v <= 1
    mask2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    
    return mask1 | mask2

def compute_area(path):
    """
    path: np.array of shape (n, 3)
    Returns the area (count of points * 0.0001)
    """
    # Start
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([-2.0, 0.0, 90.0])
    
    # Construct full sequence
    seq = np.vstack([start, path, end])
    
    # Initialize mask with Start
    mask = get_corridor_mask(0.0, 0.0, 0.0)
    
    for i in range(len(seq) - 1):
        p1 = seq[i]
        p2 = seq[i + 1]
        
        a1 = p1[2]
        a2 = p2[2]
        
        m = max(1, int(np.ceil(abs(a2 - a1))))
        
        # Interpolate
        for k in range(1, m + 1):
            t = k / m
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            a = a1 + t * (a2 - a1)
            
            step_mask = get_corridor_mask(x, y, a)
            mask &= step_mask
            
    return np.sum(mask) * 0.0001

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 550  # 550s limit, leaving 80s margin
    
    # We will model the path with a fixed set of angles, but optimize x and y.
    # The previous program used 8 points with angles 10..80.
    # Let's try a denser set of angles to allow smoother curves, but that increases computation.
    # The evaluator adds linear interpolations between consecutive rows.
    # If we provide rows at every degree, m=1 for each step, so no extra interpolation steps.
    # This allows the path to be very flexible.
    # However, 90 rows means 90*2 = 180 parameters to optimize.
    # Nelder-Mead or Powell might struggle with 180 dimensions.
    
    # Alternative: Use a parametric curve representation.
    # The inspiration suggests a circular arc.
    # Let's try a polynomial representation for x(t) and y(t) where t is angle (0 to 90).
    # x(a) = c0 + c1*a + c2*a^2 + ...
    # y(a) = d0 + d1*a + d2*a^2 + ...
    # This reduces the number of parameters.
    
    # Let's try degree 3 polynomials for x and y.
    # 4 parameters for x, 4 for y. Total 8 parameters.
    # This is very manageable for optimization.
    
    # But we need to be careful: the path must be continuous and differentiable.
    # A polynomial ensures this.
    
    # Let's define the path by evaluating the polynomials at specific angles.
    # We can evaluate at every 10 degrees (9 points) or every 5 degrees (19 points).
    # Since the evaluator interpolates linearly between our points, if we use every 10 degrees,
    # the evaluator will add 9 intermediate points per segment (m=10).
    # If we use every 1 degree, the evaluator adds 0 intermediate points (m=1).
    # Using every 1 degree gives us more control over the shape without relying on the evaluator's interpolation.
    # But optimizing 180 parameters is hard.
    
    # Let's stick to the polynomial idea but evaluate at a moderate number of points, say every 5 degrees.
    # That's 19 points (0, 5, ..., 90).
    # Wait, the start is (0,0,0) and end is (-2,0,90).
    # So our path should connect these.
    # If we use polynomials, we can fix x(0)=0, y(0)=0, x(90)=-2, y(90)=0.
    # This reduces parameters further.
    
    # Let's try a simpler approach first: just increase the number of points in the Nelder-Mead optimization
    # but use a better initial guess and maybe Powell method which handles more dimensions better.
    
    # Actually, the previous score was 1.0644 with 8 points.
    # The cell finding mentioned 1.248 -> 1.352 with more points and bump height.
    # Let's try 18 points (angles 5, 10, ..., 90).
    # That's 36 parameters.
    
    n_points = 18
    angles = np.arange(5, 91, 5.0)  # 5, 10, ..., 90
    
    # Initial guess: Linear interpolation
    x_init = np.linspace(0, -2, n_points)
    y_init = np.linspace(0, 0, n_points)
    
    params_init = np.concatenate([x_init, y_init])
    
    def objective(params):
        n = len(angles)
        path = np.zeros((n, 3))
        path[:, 2] = angles
        path[:, 0] = params[0:n]
        path[:, 1] = params[n:2*n]
        
        area = compute_area(path)
        return -area
    
    best_params = params_init.copy()
    best_score = -objective(params_init)
    
    # Use Powell method for better handling of higher dimensions
    # Powell is derivative-free and often better than Nelder-Mead for >20 dims
    
    # Check time
    if time.time() < deadline:
        try:
            from scipy.optimize import minimize
            res = minimize(objective, params_init, method='Powell', 
                           options={'maxiter': 500, 'xtol': 1e-3, 'ftol': 1e-3, 'disp': False})
            if res.fun < -best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception as e:
            pass # Keep initial if fails
            
    # Try a second optimization with Nelder-Mead if time permits
    if time.time() < deadline - 100:
        try:
            from scipy.optimize import minimize
            # Perturb initial
            np.random.seed(42)
            params_init2 = params_init + np.random.normal(0, 0.2, len(params_init))
            res2 = minimize(objective, params_init2, method='Nelder-Mead', 
                            options={'maxiter': 1000, 'xatol': 1e-3, 'fatol': 1e-3, 'disp': False})
            if res2.fun < -best_score:
                best_score = -res2.fun
                best_params = res2.x
        except Exception as e:
            pass
            
    # Construct final path
    path = np.zeros((n_points, 3))
    path[:, 2] = angles
    path[:, 0] = best_params[0:n_points]
    path[:, 1] = best_params[n_points:2*n_points]
    
    return path

# EVOLVE-BLOCK-END
