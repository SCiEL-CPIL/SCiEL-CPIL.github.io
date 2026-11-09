import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    t_start = time.time()
    time_limit = 600.0
    margin = 10.0
    
    # Pre-compute the sofa grid
    i = np.arange(450)
    j = np.arange(300)
    X, Y = np.meshgrid(i, j, indexing='ij')
    x_grid = -3 + 0.01 * X
    y_grid = -1.5 + 0.01 * Y
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
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
        
        mask = np.ones(n_points, dtype=bool)
        
        for (cx, cy, ca) in placements:
            if not np.any(mask):
                return 0.0
                
            rad = np.deg2rad(ca)
            cos_a = np.cos(rad)
            sin_a = np.sin(rad)
            
            # Only compute for currently valid points
            idx = np.where(mask)[0]
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
                # Update mask: only keep points that are valid
                new_mask = np.zeros(n_points, dtype=bool)
                new_mask[idx[valid]] = True
                mask = new_mask
                if not np.any(mask):
                    return 0.0
        count = np.sum(mask)
        return count * 0.0001

    def make_path_from_params(params, n_pts=8):
        # params: [cx2, cx3, cy2, cy3] for cubic x(t)=c1*t+c2*t^2+c3*t^3, y(t)=d1*t+d2*t^2+d3*t^3
        # x(1)=-2: c1+c2+c3=-2 => c1=-2-c2-c3
        # y(1)=0: d1+d2+d3=0 => d1=-d2-d3
        cx2, cx3, cy2, cy3 = params
        c1 = -2 - cx2 - cx3
        d1 = -cy2 - cy3
        
        t_vals = np.linspace(0, 1, n_pts + 2)[1:-1]  # interior points
        xs = c1 * t_vals + cx2 * t_vals**2 + cx3 * t_vals**3
        ys = d1 * t_vals + cy2 * t_vals**2 + cy3 * t_vals**3
        as_ = 90.0 * t_vals
        return np.column_stack((xs, ys, as_))

    def objective(params):
        path = make_path_from_params(params)
        return -check_path(path)

    best_path = None
    best_score = -1.0

    # Phase 1: Coarse grid search over quadratic paths (cx3=0, cy3=0)
    c2_vals = np.linspace(-3, 3, 12)
    d2_vals = np.linspace(-3, 3, 12)
    
    for c2 in c2_vals:
        for d2 in d2_vals:
            if time.time() - t_start > time_limit - margin:
                break
            params = np.array([c2, 0.0, d2, 0.0])
            path = make_path_from_params(params)
            score = check_path(path)
            if score > best_score:
                best_score = score
                best_path = path
        if time.time() - t_start > time_limit - margin:
            break

    # Phase 2: Local optimization from best grid point
    if best_path is not None and best_score > 0:
        # Find the params that gave best_path
        # We need to track which params gave the best
        # Re-do grid search tracking best params
        best_params = None
        for c2 in c2_vals:
            for d2 in d2_vals:
                if time.time() - t_start > time_limit - margin - 30:
                    break
                params = np.array([c2, 0.0, d2, 0.0])
                path = make_path_from_params(params)
                score = check_path(path)
                if score > best_score - 1e-10:
                    best_score = score
                    best_params = params
            if time.time() - t_start > time_limit - margin - 30:
                break
        
        if best_params is not None:
            # Optimize with scipy
            result = minimize(objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.0001})
            if result.fun < -best_score:
                best_score = -result.fun
                best_path = make_path_from_params(result.x)
    
    # Fallback: linear path
    if best_path is None or best_score <= 0:
        t_vals = np.linspace(0, 1, 10)[1:-1]
        xs = -2.0 * t_vals
        ys = np.zeros(len(t_vals))
        as_ = 90.0 * t_vals
        best_path = np.column_stack((xs, ys, as_))
        
    return best_path
