import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds safety margin
    
    # Sofa grid
    # i = 0..449, j = 0..299
    # x = -3 + 0.01*i, y = -1.5 + 0.01*j
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    sofa_points = np.column_stack((X.ravel(), Y.ravel()))
    n_points = len(sofa_points)
    
    # Base corridor definition in local coords (u, v)
    # Union of {0 <= u <= 1, -3 <= v <= 1} and {-3 <= u <= 1, 0 <= v <= 1}
    # We can define a function to check if a point (u,v) is in the base corridor.
    # in_base = ( (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1) ) | ( (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1) )
    
    def is_in_base(u, v):
        return ((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)) | ((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1))

    def evaluate_path(path):
        """
        path: (n, 3) array of (x, y, a_deg)
        Returns area (float)
        """
        # Prepend start, append end
        start = np.array([[0.0, 0.0, 0.0]])
        end = np.array([[-2.0, 0.0, 90.0]])
        full_path = np.vstack([start, path, end])
        
        # Densify
        densified = []
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            # Linear interpolation
            ts = np.linspace(0, 1, m + 1)
            # Avoid duplicate points
            if i > 0:
                ts = ts[1:]
            for t in ts:
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = p1[2] + t * (p2[2] - p1[2])
                densified.append([x, y, a])
        
        densified = np.array(densified)
        
        # Check intersection
        # Start with all points valid
        valid = np.ones(n_points, dtype=bool)
        
        for x_c, y_c, a_deg in densified:
            a_rad = np.deg2rad(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            # Transform sofa points to local coords (u, v)
            # w = (x, y) absolute
            # w - (x_c, y_c)
            dx = sofa_points[:, 0] - x_c
            dy = sofa_points[:, 1] - y_c
            
            # R(-a) * (w - c)
            # R(-a) = [[cos, sin], [-sin, cos]]
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            # Check if in base
            in_base = is_in_base(u, v)
            
            # Update valid
            valid &= in_base
            
            # Early exit if no points valid
            if not np.any(valid):
                return 0.0
                
        area = np.sum(valid) * 0.0001
        return area

    # Parameterization of the path
    # We use a cubic polynomial for x(t) and y(t) with t in [0, 1]
    # x(t) = c_x1*t + c_x2*t^2 + c_x3*t^3  (x(0)=0, x(1)=-2 implies c_x1+c_x2+c_x3 = -2)
    # y(t) = c_y1*t + c_y2*t^2 + c_y3*t^3  (y(0)=0, y(1)=0 implies c_y1+c_y2+c_y3 = 0)
    
    # To simplify optimization, we can parameterize by the "bulge"
    # Let's use 3 parameters: A, B, C
    # x(t) = -2*t + A * t * (1-t)  # Parabolic bulge in x?
    # y(t) = B * t * (1-t) + C * t * (1-t) * (2-t) ?
    
    # Let's stick to the polynomial coefficients but constrain them.
    # Variables: [cx1, cx2, cx3, cy1, cy2, cy3]
    # Constraints:
    # cx1 + cx2 + cx3 = -2
    # cy1 + cy2 + cy3 = 0
    
    # We can eliminate cx3 and cy3:
    # cx3 = -2 - cx1 - cx2
    # cy3 = -cy1 - cy2
    
    # So variables are [cx1, cx2, cy1, cy2]
    
    def get_path_from_params(params):
        cx1, cx2, cy1, cy2 = params
        cx3 = -2.0 - cx1 - cx2
        cy3 = -cy1 - cy2
        
        # Generate waypoints
        # We need to provide a path to the evaluator.
        # The evaluator densifies linearly.
        # To approximate the curve, we provide points sampled from the curve.
        n_waypoints = 10
        ts = np.linspace(0, 1, n_waypoints)
        
        xs = cx1*ts + cx2*ts**2 + cx3*ts**3
        ys = cy1*ts + cy2*ts**2 + cy3*ts**3
        as_ = np.linspace(0, 90, n_waypoints)
        
        # The evaluator prepends (0,0,0) and appends (-2,0,90).
        # Our ts[0]=0 gives (0,0,0). ts[-1]=1 gives (-2,0,90).
        # So we should return the intermediate points?
        # The evaluator adds start and end.
        # If we return all points including start and end, the evaluator will add duplicates.
        # Duplicates are fine (m=1, step 0).
        # But to be safe, let's return points strictly between start and end?
        # Or just return the full set, the evaluator handles it.
        # "The evaluator puts (0, 0, 0) before your rows and (-2, 0, 90) after them."
        # So if I return (0,0,0) as first row, the path starts (0,0,0), (0,0,0), ...
        # This is valid.
        
        path = np.column_stack((xs, ys, as_))
        return path

    def objective(params):
        if time.time() > deadline:
            return 0.0
        path = get_path_from_params(params)
        return -evaluate_path(path) # Minimize negative area

    # Initial guess
    # A simple linear path: x = -2t, y = 0
    # cx1 = -2, cx2 = 0, cx3 = 0
    # cy1 = 0, cy2 = 0, cy3 = 0
    # params = [-2, 0, 0, 0]
    
    # A bulged path:
    # y(t) should be positive (bulging into the "outside" of the L? or inside?)
    # The L is in x>=0, y>=0 roughly?
    # Base: u in [0,1], v in [-3,1] AND u in [-3,1], v in [0,1].
    # The "empty" space is u<0, v<0.
    # The sofa must stay in the union.
    # If we bulge y(t) to be positive, we move the corner into the horizontal arm's y-range [0,1].
    # This might allow a larger intersection.
    
    x0 = np.array([-2.0, 0.0, 0.5, 0.0]) # cx1, cx2, cy1, cy2
    # Let's try a few initial guesses and run Nelder-Mead on the best one.
    
    candidates = [
        np.array([-2.0, 0.0, 0.0, 0.0]), # Linear
        np.array([-2.0, 0.0, 1.0, -1.0]), # Bulge y
        np.array([-2.0, 1.0, 0.5, -0.5]), # Bulge x and y
        np.array([-1.0, -1.0, 0.5, -0.5]), # Different x shape
    ]
    
    best_params = x0
    best_score = -1e9
    
    for cand in candidates:
        if time.time() > deadline:
            break
        res = minimize(objective, cand, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    
    # Final polish with more iterations if time allows
    if time.time() < deadline - 60:
        res = minimize(objective, best_params, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
            
    final_path = get_path_from_params(best_params)
    
    # Ensure shape and finiteness
    if not np.all(np.isfinite(final_path)):
        return np.array([[0.0, 0.0, 15.0], [0.0, 0.0, 75.0]])
        
    return final_path

# EVOLVE-BLOCK-END
