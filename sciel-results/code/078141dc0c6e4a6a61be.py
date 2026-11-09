import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = 610.0  # Safety margin before 630s kill
    
    # Grid for evaluation, matching the evaluator
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    P = np.column_stack((X.ravel(), Y.ravel()))
    n_points = P.shape[0]
    
    def get_base_mask(u, v):
        # Base corridor:
        # Arm 1: 0 <= u <= 1, -3 <= v <= 1
        # Arm 2: -3 <= u <= 1, 0 <= v <= 1
        return ((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)) | \
               ((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1))

    def evaluate_path_dense(path):
        """
        Calculate the area of the intersection of the corridor placements.
        path: array of shape (n, 3)
        """
        # Prepend (0,0,0) and append (-2,0,90)
        # The evaluator does this, so we must simulate it for accurate scoring.
        full_path = np.vstack([np.array([0.0, 0.0, 0.0]), path, np.array([-2.0, 0.0, 90.0])])
        
        # Densify path exactly as evaluator does
        dense_path = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            if m == 1:
                dense_path.append(p2)
            else:
                for k in range(1, m+1):
                    t = k / m
                    interp = p1 + t * (p2 - p1)
                    dense_path.append(interp)
        
        valid = np.ones(n_points, dtype=bool)
        
        for px, py, pa in dense_path:
            a_rad = np.deg2rad(pa)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = P[:, 0] - px
            dy = P[:, 1] - py
            
            # R(-a) rotation
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            in_base = get_base_mask(u, v)
            
            valid = valid & in_base
            
            if not np.any(valid):
                return 0.0
                
        return np.sum(valid) * 0.0001

    def evaluate_arc_params(params):
        """
        params: [R, cx, cy]
        R: radius
        cx, cy: center of the circle
        """
        if time.time() > deadline - 5:
            return 0.0
        
        R, cx, cy = params
        
        if R < 1e-6:
            return 0.0
            
        # Start point (0,0), angle 0
        # End point (-2,0), angle 90
        # The arc must pass through (0,0) and (-2,0).
        # Distance from center to (0,0) must be R.
        # Distance from center to (-2,0) must be R.
        # This implies the center lies on the perpendicular bisector of the segment from (0,0) to (-2,0).
        # Midpoint is (-1, 0). Perpendicular bisector is x = -1.
        # So cx must be -1.
        # If cx != -1, the arc cannot be a single circular arc of radius R passing through both points 
        # with the correct tangents? 
        # Actually, the tangent at (0,0) is angle 0 (horizontal). 
        # The radius vector from center to (0,0) must be perpendicular to the tangent.
        # So radius vector is vertical. Thus cx = 0.
        # Wait.
        # Tangent at start (0,0) is 0 deg. Radius is 90 deg. Center is (0, R) or (0, -R).
        # Tangent at end (-2,0) is 90 deg. Radius is 0 deg. Center is (-2+R, 0) or (-2-R, 0).
        # For a single arc, the center must be the same.
        # So (0, R) = (-2+R, 0) => R=0, impossible.
        # Or (0, R) = (-2-R, 0) => R=0, impossible.
        # This means a single circular arc CANNOT satisfy both position and tangent constraints 
        # for the specific start/end points (0,0) and (-2,0) with angles 0 and 90.
        # 
        # Let's re-read the problem.
        # "Moving sofa from a corridor path"
        # Start: (0,0,0). End: (-2,0,90).
        # Is it possible to move a sofa?
        # The sofa is width 1.
        # The corridor is L-shaped.
        # 
        # Maybe the "circular arc" strategy refers to the *center of the sofa*?
        # The sofa is the L-shaped corridor itself? No, "The sofa is the set of grid points... inside every placement".
        # The "sofa" is the object being moved. The "corridor" is the constraint?
        # "Row (x, y, a) places the width-1 L-shaped corridor with its inner corner at (x, y)"
        # So the *corridor* moves?
        # "The sofa is the set of grid points ... inside every placement"
        # This implies the sofa is the intersection of the corridors.
        # We want to maximize the area of the intersection.
        # This is equivalent to finding a path for the *inner corner* of the L-shaped corridor 
        # such that the intersection of the corridors along the path is maximized.
        
        # Since a single circular arc is geometrically impossible for the exact constraints 
        # (tangent mismatch), we must relax the model.
        # Perhaps we model the path as a circular arc that *approximates* the motion, 
        # or we use a different geometric primitive.
        
        # Let's try a **Spline** approach with geometric constraints.
        # Or, simpler: The "circular arc" idea might be flawed for the endpoints.
        # Let's stick to the **Cubic Bezier** but optimize it more aggressively, 
        # OR use a **Quadratic Bezier** which has 3 control points (P0, P1, P2).
        # P0=(0,0), P2=(-2,0). P1=(x1, y1).
        # Tangent at P0 is direction P1-P0 = (x1, y1). Angle = atan2(y1, x1).
        # We want angle 0. So y1=0, x1>0.
        # Tangent at P2 is direction P2-P1 = (-2-x1, -y1). Angle = atan2(-y1, -2-x1).
        # We want angle 90. So -2-x1=0 => x1=-2. And -y1>0 => y1<0.
        # So P1 must be (x1, 0) with x1>0 AND P1 must be (-2, y1) with y1<0.
        # Contradiction.
        
        # Conclusion: No single polynomial Bezier of low degree can satisfy the tangent constraints exactly.
        # The current program (Cubic Bezier) ignores the exact tangent constraints of the *path* 
        # and just optimizes the intersection area. The evaluator interpolates linearly.
        # The "tangent" of the *sofa's orientation* changes linearly with the interpolation steps.
        
        # Let's try a **Piecewise Linear** path with optimized vertices.
        # The evaluator densifies the path.
        # If we provide points at every 1 degree, m=1.
        # We can optimize the (x,y) for each angle 1..89.
        # This is 178 parameters. Too many for Nelder-Mead.
        
        # Let's go back to the **Circular Arc** idea but interpret it differently.
        # Maybe the *center of the sofa* follows a circular arc?
        # The sofa's orientation is fixed relative to the corner.
        # If the corner moves along a circular arc, does the orientation change?
        # The problem says "rotated by a degrees".
        # If we force the path to be a circular arc, we can't satisfy the start/end angles 
        # unless we allow the *orientation* to be independent of the *position* path?
        # No, the output is (x, y, a).
        
        # Let's try a **Hybrid** approach:
        # Use a **Cubic Spline** through a few key points, optimized via coordinate descent.
        # Or, simpler: **Simulated Annealing** on the 4 Bezier parameters.
        # The current score is 1.3762.
        # Let's try **Basin Hopping** or **Differential Evolution** on the 4 Bezier parameters.
        
        return 0.0 # Placeholder, will implement actual search below

    # Since the geometric constraint of a single arc is impossible, 
    # we will use a **Cubic Bezier** but optimize it using **Differential Evolution** 
    # which is more robust for global optimization than Nelder-Mead for this non-convex problem.
    
    def evaluate_bezier(params):
        if time.time() > deadline - 5:
            return 0.0
        x1, y1, x2, y2 = params
        
        angles = np.arange(1, 90)
        t = angles / 90.0
        one_minus_t = 1.0 - t
        
        bx = 3 * (one_minus_t**2) * t * x1 + 3 * (one_minus_t) * (t**2) * x2 + (t**3) * (-2.0)
        by = 3 * (one_minus_t**2) * t * y1 + 3 * (one_minus_t) * (t**2) * y2 + (t**3) * 0.0
        
        path = np.column_stack((bx, by, angles))
        
        return evaluate_path_dense(path)

    from scipy.optimize import differential_evolution

    # Bounds for Bezier control points
    # The path should stay within reasonable bounds, e.g., x in [-3, 1], y in [-3, 3]
    bounds = [(-3.0, 1.0), (-3.0, 3.0), (-3.0, 1.0), (-3.0, 3.0)]
    
    # Initial best guess from previous run
    x0 = np.array([-0.5, 1.0, -1.5, 1.0])
    best_score = evaluate_bezier(x0)
    best_result = x0

    try:
        # Differential Evolution is stochastic and might be slow, 
        # but with maxiter=100 and popsize=15, it should be fast enough.
        # We use a tight budget.
        result = differential_evolution(
            lambda p: -evaluate_bezier(p), # Maximize by minimizing negative
            bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-4,
            polish=False
        )
        
        if result.fun < -best_score:
            best_score = -result.fun
            best_result = result.x
            
    except Exception as e:
        pass

    # Refine with Nelder-Mead if time permits
    if time.time() < deadline - 20:
        from scipy.optimize import minimize
        try:
            result2 = minimize(
                lambda p: -evaluate_bezier(p),
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-5, 'disp': False}
            )
            if result2.fun < -best_score:
                best_score = -result2.fun
                best_result = result2.x
        except Exception as e:
            pass

    # Construct final path
    x1, y1, x2, y2 = best_result
    angles = np.arange(1, 90)
    t = angles / 90.0
    one_minus_t = 1.0 - t
    
    bx = 3 * (one_minus_t**2) * t * x1 + 3 * (one_minus_t) * (t**2) * x2 + (t**3) * (-2.0)
    by = 3 * (one_minus_t**2) * t * y1 + 3 * (one_minus_t) * (t**2) * y2 + (t**3) * 0.0
    
    final_path = np.column_stack((bx, by, angles))
    
    return final_path
