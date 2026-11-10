import numpy as np
import time
import math

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = 615.0  # Safety margin before 630s kill
    
    # Grid for evaluation, matching the evaluator
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    P = np.column_stack((X.ravel(), Y.ravel()))
    n_points = P.shape[0]
    
    def evaluate_path_fast(path):
        """
        Calculate the area of the intersection of the corridor placements.
        path: array of shape (n, 3)
        Returns: score
        """
        if not isinstance(path, np.ndarray):
            path = np.array(path)
            
        # Prepend and append fixed endpoints
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
        
        if not np.any(valid):
            return 0.0
            
        # Loop through dense_path and update valid
        for px, py, pa in dense_path:
            a_rad = np.deg2rad(pa)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = P[:, 0] - px
            dy = P[:, 1] - py
            
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            in_base = ((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)) | \
                      ((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1))
            
            valid = valid & in_base
            
            # Early exit if too few points remain
            count = np.sum(valid)
            if count < 100:
                return 0.0
                
        score = np.sum(valid) * 0.0001
        return score

    def get_path_from_arc_params(params):
        """
        Generate path from circular arc parameters.
        params: [r, cx, cy]
        The path is a circular arc from angle 0 to 90 degrees.
        The center of the circle is (cx, cy).
        The radius is r.
        The start point is (0,0), end point is (-2, 0).
        
        Wait, a single circular arc from (0,0) to (-2,0) with tangents at 0 and 90 degrees
        is not possible with a single circle unless the tangents are perpendicular to the radius.
        
        At (0,0), tangent is horizontal (angle 0). So radius is vertical. Center is (0, r) or (0, -r).
        At (-2,0), tangent is vertical (angle 90). So radius is horizontal. Center is (-2, 0+r') or (-2, 0-r').
        
        If we use a single circle, the center must be equidistant from both points.
        Distance from (0,0) to (cx,cy) is r.
        Distance from (-2,0) to (cx,cy) is r.
        So cx = -1.
        cy^2 + 1 = r^2.
        
        Tangent at (0,0) is perpendicular to radius (0, cy).
        Radius vector is (0, cy). Tangent is horizontal (1,0). Dot product is 0. OK.
        Tangent at (-2,0) is perpendicular to radius (-1, cy).
        Radius vector is (-1, cy). Tangent is vertical (0,1). Dot product is cy.
        For dot product to be 0, cy must be 0. But then r=1.
        If cy=0, r=1. Center (-1,0).
        Arc from (0,0) to (-2,0) around (-1,0).
        Angle at (0,0) is 0 deg relative to center? Vector from center to point is (1,0). Angle 0.
        Angle at (-2,0) is 180 deg relative to center? Vector from center to point is (-1,0). Angle 180.
        So the arc goes from 0 to 180 degrees. That's a semicircle.
        The rotation of the corridor would change by 180 degrees?
        But the corridor rotation is specified separately as 'a'.
        The path (x,y) is the position of the inner corner.
        The rotation 'a' is the angle of the corridor.
        
        In the current program, 'a' is linearly interpolated.
        Here, we can define 'a' to follow the tangent angle of the path if we want, or keep it linear.
        The task says: "Row (x, y, a) places ... rotated by a degrees."
        It does not require 'a' to match the path tangent.
        However, for a smooth motion, it's often better if they are related.
        
        Let's stick to the current program's strategy of linear 'a' (1..89) but optimize the (x,y) path.
        The current program uses a Cubic Bezier for (x,y).
        
        Let's try a different path parameterization:
        A path composed of 3 segments:
        1. Straight line from (0,0) to (x1, y1)
        2. Circular arc from (x1, y1) to (x2, y2)
        3. Straight line from (x2, y2) to (-2, 0)
        
        This is complex to parameterize and optimize.
        
        Let's try a simpler different strategy:
        **Grid Search over a 2-parameter family of paths.**
        Family: x(t) = -2t, y(t) = A * sin(pi * t) + B * sin(2*pi * t)
        t in [0,1]. a = 90 * t.
        Optimize A and B using a coarse grid search followed by local refinement.
        """
        A, B = params
        angles = np.arange(1, 90)
        t_vals = angles / 90.0
        
        bx = -2.0 * t_vals
        by = A * np.sin(np.pi * t_vals) + B * np.sin(2 * np.pi * t_vals)
        
        path = np.column_stack([bx, by, angles])
        return path

    # Coarse Grid Search
    best_score = -np.inf
    best_params = None
    
    # Range for A and B
    # y should be positive to go around the corner.
    # Max y for a unit corridor turning 90 deg is roughly 2.
    # Let's search A in [0, 2] and B in [-1, 1].
    
    A_range = np.linspace(0, 2.0, 20)
    B_range = np.linspace(-1.0, 1.0, 20)
    
    for A in A_range:
        for B in B_range:
            if time.time() > deadline - 30:
                break
            params = np.array([A, B])
            path = get_path_from_arc_params(params)
            score = evaluate_path_fast(path)
            if score > best_score:
                best_score = score
                best_params = params
        
        if time.time() > deadline - 30:
            break
            
    if best_params is None:
        best_params = np.array([1.0, 0.0])
        
    # Local Refinement
    # Use a simple Nelder-Mead or coordinate descent on the best region.
    # Let's do a few steps of coordinate descent.
    step = 0.1
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        improved = False
        for i in range(2):
            new_params = best_params.copy()
            new_params[i] += step
            path = get_path_from_arc_params(new_params)
            score = evaluate_path_fast(path)
            if score > best_score:
                best_score = score
                best_params = new_params
                improved = True
                
            new_params = best_params.copy()
            new_params[i] -= step
            path = get_path_from_arc_params(new_params)
            score = evaluate_path_fast(path)
            if score > best_score:
                best_score = score
                best_params = new_params
                improved = True
        
        if not improved:
            step /= 2
            if step < 0.01:
                break
                
    final_params = best_params
    final_path = get_path_from_arc_params(final_params)
    
    return final_path
