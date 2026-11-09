import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    t_start = time.time()
    time_limit = 600.0
    margin = 20.0

    # Pre-compute the sofa grid
    x_g = np.arange(-3.0, 1.5, 0.01)
    y_g = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_g, y_g)
    x_flat = X.flatten()
    y_flat = Y.flatten()
    n_points = x_flat.size

    def check_path(path_pts):
        placements = [(0.0, 0.0, 0.0)]
        prev_x, prev_y, prev_a = 0.0, 0.0, 0.0

        def add_interpolations(x1, y1, a1, x2, y2, a2):
            m = max(1, int(np.ceil(abs(a2 - a1))))
            for k in range(1, m + 1):
                t = k / m
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                a = a1 + t * (a2 - a1)
                placements.append((x, y, a))

        for p in path_pts:
            px, py, pa = p
            add_interpolations(prev_x, prev_y, prev_a, px, py, pa)
            prev_x, prev_y, prev_a = px, py, pa

        add_interpolations(prev_x, prev_y, prev_a, -2.0, 0.0, 90.0)

        remaining = np.ones(n_points, dtype=bool)

        for (cx, cy, ca) in placements:
            if not np.any(remaining):
                return 0.0

            rad = np.deg2rad(ca)
            cos_a = np.cos(rad)
            sin_a = np.sin(rad)

            idx = np.where(remaining)[0]
            if len(idx) == 0:
                return 0.0

            dx = x_flat[idx] - cx
            dy = y_flat[idx] - cy
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a

            cond1 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            cond2 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            valid = cond1 | cond2

            if not np.all(valid):
                remaining[idx[~valid]] = False
                if not np.any(remaining):
                    return 0.0

        count = np.sum(remaining)
        return count * 0.0001

    def make_circular_arc_path(params, n_pts=20):
        """
        params: [r, cy]
        Circular arc from (0,0) with horizontal tangent to (-2,0) with vertical tangent.
        Center at (cx, cy), radius r.
        At start (0,0): tangent horizontal => center directly above/below: cx = 0, cy = ±r
        At end (-2,0): tangent vertical => center directly left/right: cx = -2±r
        For a valid arc: center at (cx, cy) with distance r to both points.
        Distance to (0,0): cx^2 + cy^2 = r^2
        Distance to (-2,0): (cx+2)^2 + cy^2 = r^2
        Subtracting: (cx+2)^2 - cx^2 = 0 => 4cx + 4 = 0 => cx = -1
        So center is at (-1, cy), r^2 = 1 + cy^2, r = sqrt(1 + cy^2)
        The angle swept is 2*atan2(r, 1) = 2*atan(r)
        We need the angle to go from 0 to 90 degrees, so we interpolate linearly.
        params: [cy, n_pts_extra] where cy determines the arc shape
        """
        cy = params[0]
        cx_center = -1.0
        r = np.sqrt(1.0 + cy * cy)
        
        # Start angle: from center to (0,0)
        start_angle = np.arctan2(0.0 - cy, 0.0 - cx_center)  # angle from center to start
        # End angle: from center to (-2,0)
        end_angle = np.arctan2(0.0 - cy, -2.0 - cx_center)  # angle from center to end
        
        # We want the arc to go from start to end. The total angle change should be 90 degrees
        # for the corridor orientation. Let's just linearly interpolate the corridor angle
        # and use the circular arc for position.
        
        # Parameterize the arc by fraction t in [0,1]
        t_vals = np.linspace(0, 1, n_pts + 2)[1:-1]
        
        # Angle along the arc: interpolate from start_angle to end_angle
        # But we need to handle the direction correctly
        angle_change = end_angle - start_angle
        # Normalize to be in (-pi, pi]
        while angle_change > np.pi:
            angle_change -= 2 * np.pi
        while angle_change <= -np.pi:
            angle_change += 2 * np.pi
            
        angles = start_angle + t_vals * angle_change
        
        xs = cx_center + r * np.cos(angles)
        ys = cy + r * np.sin(angles)
        
        # Corridor angle: linear from 0 to 90
        as_ = 90.0 * t_vals
        
        return np.column_stack((xs, ys, as_))

    def make_cubic_path(params, n_pts=20):
        # params: [cx2, cx3, cy2, cy3]
        cx2, cx3, cy2, cy3 = params
        c1 = -2 - cx2 - cx3
        d1 = -cy2 - cy3

        t_vals = np.linspace(0, 1, n_pts + 2)[1:-1]
        xs = c1 * t_vals + cx2 * t_vals**2 + cx3 * t_vals**3
        ys = d1 * t_vals + cy2 * t_vals**2 + cy3 * t_vals**3
        as_ = 90.0 * t_vals
        return np.column_stack((xs, ys, as_))

    def make_bezier_path(params, n_pts=20):
        # params: [cx, cy] for quadratic Bezier
        cx, cy = params
        t_vals = np.linspace(0, 1, n_pts + 2)[1:-1]
        xs = 2 * t_vals * (1 - t_vals) * cx + t_vals**2 * (-2.0)
        ys = 2 * t_vals * (1 - t_vals) * cy
        as_ = 90.0 * t_vals
        return np.column_stack((xs, ys, as_))

    def objective(params, path_func):
        path = path_func(params)
        return -check_path(path)

    best_path = None
    best_score = -1.0

    # Phase 1: Grid search over circular arc paths (1 parameter: cy)
    cy_vals = np.linspace(-3.0, 3.0, 30)
    
    for cy in cy_vals:
        if time.time() - t_start > time_limit - margin:
            break
        params = np.array([cy])
        path = make_circular_arc_path(params)
        score = check_path(path)
        if score > best_score:
            best_score = score
            best_path = path

    # Phase 2: Grid search over quadratic Bezier
    cx_vals = np.linspace(-3, 1, 10)
    cy_vals_b = np.linspace(-2.5, 2.5, 10)

    for cx in cx_vals:
        for cy in cy_vals_b:
            if time.time() - t_start > time_limit - margin:
                break
            params = np.array([cx, cy])
            path = make_bezier_path(params)
            score = check_path(path)
            if score > best_score:
                best_score = score
                best_path = path
        if time.time() - t_start > time_limit - margin:
            break

    # Phase 3: Grid search over cubic paths
    c2_vals = np.linspace(-2, 2, 6)
    c3_vals = np.linspace(-1, 1, 4)
    d2_vals = np.linspace(-2, 2, 6)
    d3_vals = np.linspace(-1, 1, 4)

    for c2 in c2_vals:
        for d2 in d2_vals:
            for c3 in c3_vals:
                for d3 in d3_vals:
                    if time.time() - t_start > time_limit - margin - 30:
                        break
                    params = np.array([c2, c3, d2, d3])
                    path = make_cubic_path(params)
                    score = check_path(path)
                    if score > best_score:
                        best_score = score
                        best_path = path
                if time.time() - t_start > time_limit - margin - 30:
                    break
            if time.time() - t_start > time_limit - margin - 30:
                break
        if time.time() - t_start > time_limit - margin - 30:
            break

    # Phase 4: Local optimization on the best path type
    if best_path is not None and best_score > 0:
        # Determine which parameterization gave the best result by re-checking
        best_params = None
        best_path_type = None

        # Check circular arc
        for cy in cy_vals:
            if time.time() - t_start > time_limit - margin - 60:
                break
            params = np.array([cy])
            path = make_circular_arc_path(params)
            score = check_path(path)
            if score > best_score - 1e-10:
                best_score = score
                best_params = params
                best_path_type = 'arc'

        # Check Bezier
        for cx in cx_vals:
            for cy in cy_vals_b:
                if time.time() - t_start > time_limit - margin - 60:
                    break
                params = np.array([cx, cy])
                path = make_bezier_path(params)
                score = check_path(path)
                if score > best_score - 1e-10:
                    best_score = score
                    best_params = params
                    best_path_type = 'bezier'
            if time.time() - t_start > time_limit - margin - 60:
                break

        # Check cubic
        for c2 in c2_vals:
            for d2 in d2_vals:
                for c3 in c3_vals:
                    for d3 in d3_vals:
                        if time.time() - t_start > time_limit - margin - 60:
                            break
                        params = np.array([c2, c3, d2, d3])
                        path = make_cubic_path(params)
                        score = check_path(path)
                        if score > best_score - 1e-10:
                            best_score = score
                            best_params = params
                            best_path_type = 'cubic'
                    if time.time() - t_start > time_limit - margin - 60:
                        break
                if time.time() - t_start > time_limit - margin - 60:
                    break
            if time.time() - t_start > time_limit - margin - 60:
                break

        if best_params is not None:
            if best_path_type == 'arc':
                result = minimize(lambda p: objective(p, make_circular_arc_path), best_params,
                                method='Nelder-Mead',
                                options={'maxiter': 500, 'xatol': 0.001, 'fatol': 0.00001})
                if result.fun < -best_score:
                    best_score = -result.fun
                    best_path = make_circular_arc_path(result.x)
            elif best_path_type == 'bezier':
                result = minimize(lambda p: objective(p, make_bezier_path), best_params,
                                method='Nelder-Mead',
                                options={'maxiter': 500, 'xatol': 0.001, 'fatol': 0.00001})
                if result.fun < -best_score:
                    best_score = -result.fun
                    best_path = make_bezier_path(result.x)
            elif best_path_type == 'cubic':
                result = minimize(lambda p: objective(p, make_cubic_path), best_params,
                                method='Nelder-Mead',
                                options={'maxiter': 500, 'xatol': 0.001, 'fatol': 0.00001})
                if result.fun < -best_score:
                    best_score = -result.fun
                    best_path = make_cubic_path(result.x)

    # Fallback: linear path
    if best_path is None or best_score <= 0:
        t_vals = np.linspace(0, 1, 15)[1:-1]
        xs = -2.0 * t_vals
        ys = np.zeros(len(t_vals))
        as_ = 90.0 * t_vals
        best_path = np.column_stack((xs, ys, as_))

    return best_path
