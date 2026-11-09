# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _sofa_grid():
    """Create the sofa point grid once."""
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    return gx.ravel(), gy.ravel()


def _corridor_mask_batch(x, y, a_deg, px, py):
    """
    Vectorized: for each placement (x[i], y[i], a_deg[i]), compute boolean mask
    over all sofa points. Returns array of shape (n_placements, n_sofa_points).
    """
    n_pl = len(x)
    n_sp = len(px)
    
    rad = np.deg2rad(a_deg)  # (n_pl,)
    c = np.cos(rad)  # (n_pl,)
    s = np.sin(rad)  # (n_pl,)
    
    # dx[i,j] = px[j] - x[i], dy[i,j] = py[j] - y[i]
    dx = px[None, :] - x[:, None]  # (n_pl, n_sp)
    dy = py[None, :] - y[:, None]  # (n_pl, n_sp)
    
    u = c[:, None] * dx + s[:, None] * dy  # (n_pl, n_sp)
    v = -s[:, None] * dx + c[:, None] * dy  # (n_pl, n_sp)
    
    eps = 1e-9
    m1 = (u >= -eps) & (u <= 1 + eps) & (v >= -3 - eps) & (v <= 1 + eps)
    m2 = (u >= -3 - eps) & (u <= 1 + eps) & (v >= -eps) & (v <= 1 + eps)
    
    return m1 | m2  # (n_pl, n_sp)


def _compute_sofa_area(path, px, py):
    """
    Given a full densified path, compute the number of sofa points inside every placement.
    Uses batched evaluation.
    """
    n_pl = len(path)
    if n_pl == 0:
        return 0.0
    
    x = path[:, 0]
    y = path[:, 1]
    a = path[:, 2]
    
    # Process in chunks to manage memory
    chunk_size = max(1, min(n_pl, 200))
    inside = np.ones(len(px), dtype=bool)
    
    for start in range(0, n_pl, chunk_size):
        end = min(start + chunk_size, n_pl)
        mask = _corridor_mask_batch(x[start:end], y[start:end], a[start:end], px, py)
        # A point must be inside ALL placements in this chunk
        chunk_inside = np.all(mask, axis=0)
        inside = inside & chunk_inside
        if not np.any(inside):
            break
    
    return np.count_nonzero(inside) * 0.0001


def _densify_path(rows):
    """
    Densify the path: between consecutive rows, add linear interpolations
    in m = max(1, ceil(|a2-a1|)) equal steps.
    """
    if len(rows) <= 1:
        return np.array(rows)
    
    result = [rows[0]]
    for i in range(len(rows) - 1):
        x1, y1, a1 = rows[i]
        x2, y2, a2 = rows[i + 1]
        m = max(1, int(np.ceil(abs(a2 - a1))))
        for k in range(1, m + 1):
            t = k / m
            result.append([x1 + t * (x2 - x1),
                           y1 + t * (y2 - y1),
                           a1 + t * (a2 - a1)])
    return np.array(result)


def _arc_path(cx, cy, r, n_pts=20):
    """
    Generate an arc-based path from (0,0,0) to (-2,0,90).
    The path follows a circular arc centered at (cx, cy) with radius r.
    """
    # Start angle: angle from (cx,cy) to (0,0)
    start_angle = np.arctan2(0 - cy, 0 - cx)
    # End angle: angle from (cx,cy) to (-2,0)
    end_angle = np.arctan2(0 - cy, -2 - cx)
    
    # Ensure we go the "short way" or appropriate way
    # We want angle to increase from ~0 to ~90 degrees in the corridor frame
    # Let's parameterize differently: use the angle of the corridor orientation
    
    # Actually, let's use a simpler parameterization:
    # The corridor inner corner traces a path. We'll use a Bezier-like curve
    # parameterized by the arc center and a few shape parameters.
    
    # Simpler: just return control points for a smooth path
    # Use a quadratic Bezier from (0,0) to (-2,0) with control point influenced by (cx,cy)
    # The angle goes from 0 to 90
    
    t = np.linspace(0, 1, n_pts)
    # Quadratic Bezier: P(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
    P0 = np.array([0.0, 0.0])
    P2 = np.array([-2.0, 0.0])
    P1 = np.array([cx, cy])
    
    xs = (1-t)**2 * P0[0] + 2*(1-t)*t * P1[0] + t**2 * P2[0]
    ys = (1-t)**2 * P0[1] + 2*(1-t)*t * P1[1] + t**2 * P2[1]
    angles = t * 90.0  # Linear angle from 0 to 90
    
    return np.column_stack([xs, ys, angles])


def _path_from_params(params, px, py):
    """
    Convert params to a full densified path.
    params can be:
    - 2D array of shape (n, 3): control points (x, y, a)
    - 1D array of shape (3,): arc center (cx, cy, r) -> use arc path
    """
    if len(params.shape) == 1 and len(params) == 3:
        # Arc-based path
        cx, cy, r = params
        # Use the arc to generate control points
        raw = _arc_path(cx, cy, r)
        raw = np.vstack([np.array([0, 0, 0.0]), raw[1:-1], np.array([-2, 0, 90.0])])
        return _densify_path(raw)
    else:
        # Direct control points
        raw = np.vstack([np.array([0, 0, 0.0]), params, np.array([-2, 0, 90.0])])
        return _densify_path(raw)


def _objective(params, px, py):
    """Negative area (we minimize negative area = maximize area)."""
    path = _path_from_params(params, px, py)
    area = _compute_sofa_area(path, px, py)
    return -area


def moving_sofa_2d() -> np.ndarray:
    """
    Optimize the corridor path to maximize the sofa area.
    """
    deadline = time.time() + 590  # 40s safety margin
    
    px, py = _sofa_grid()
    
    # Baseline: a reasonable initial path
    baseline = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ])
    
    full_baseline = _densify_path(np.vstack([np.array([0,0,0.0]), baseline, np.array([-2,0,90.0])]))
    best_area = _compute_sofa_area(full_baseline, px, py)
    best_params = baseline.copy()
    
    # Strategy 1: Arc-based search (2D: cx, cy)
    if time.time() < deadline - 180:
        # Try arc center search
        def obj_arc(params):
            if time.time() > deadline - 20:
                return 0
            cx, cy = params
            # Use a fixed radius or derive from cx, cy
            # The arc should pass near the origin and (-2,0)
            # Radius is determined by the geometry
            r = np.sqrt(cx**2 + cy**2)  # distance from center to origin
            # Actually, let's just use the control point approach with cx, cy
            # Generate a Bezier-like path
            n_pts = 15
            t = np.linspace(0, 1, n_pts)
            P0 = np.array([0.0, 0.0])
            P2 = np.array([-2.0, 0.0])
            P1 = np.array([cx, cy])
            
            xs = (1-t)**2 * P0[0] + 2*(1-t)*t * P1[0] + t**2 * P2[0]
            ys = (1-t)**2 * P0[1] + 2*(1-t)*t * P1[1] + t**2 * P2[1]
            angles = t * 90.0
            
            ctrl_pts = np.column_stack([xs, ys, angles])
            # Remove first and last (they're the fixed endpoints)
            ctrl_pts = ctrl_pts[1:-1]
            
            return _objective(ctrl_pts, px, py)
        
        # Differential evolution on (cx, cy)
        bounds_arc = [(-1.5, 0.5), (-0.5, 1.5)]
        try:
            result = differential_evolution(
                obj_arc, bounds_arc,
                seed=42,
                maxiter=50,
                tol=0.005,
                popsize=20,
                polish=False,
                update_interval=10,
            )
            if hasattr(result, 'x'):
                cx, cy = result.x
                n_pts = 15
                t = np.linspace(0, 1, n_pts)
                P0 = np.array([0.0, 0.0])
                P2 = np.array([-2.0, 0.0])
                P1 = np.array([cx, cy])
                
                xs = (1-t)**2 * P0[0] + 2*(1-t)*t * P1[0] + t**2 * P2[0]
                ys = (1-t)**2 * P0[1] + 2*(1-t)*t * P1[1] + t**2 * P2[1]
                angles = t * 90.0
                
                ctrl_pts = np.column_stack([xs, ys, angles])[1:-1]
                area = -_objective(ctrl_pts, px, py)
                if area > best_area:
                    best_area = area
                    best_params = ctrl_pts.copy()
        except Exception:
            pass
    
    # Strategy 2: Local refinement on best_params with Nelder-Mead
    if time.time() < deadline - 120 and len(best_params) > 0:
        try:
            result = minimize(
                lambda p: _objective(p.reshape(-1, 3), px, py),
                best_params.ravel(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 0.005, 'fatol': 1e-7, 'adaptive': True}
            )
            if hasattr(result, 'x'):
                params = result.x.reshape(-1, 3)
                params[:, 2] = np.clip(params[:, 2], 0, 90)
                area = -_objective(params, px, py)
                if area > best_area:
                    best_area = area
                    best_params = params.copy()
        except Exception:
            pass
    
    # Strategy 3: Fine-tune with smaller steps
    if time.time() < deadline - 60 and len(best_params) > 0:
        try:
            # Perturbation-based local search
            rng = np.random.RandomState(42)
            current = best_params.copy()
            current_area = best_area
            step_size = 0.02
            
            for it in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Random perturbation
                perturbation = rng.normal(0, step_size, current.shape)
                # Keep angles roughly monotonic
                new_params = current + perturbation
                new_params[:, 2] = np.clip(new_params[:, 2], 0, 90)
                
                # Ensure angles are increasing
                for i in range(1, len(new_params)):
                    if new_params[i, 2] < new_params[i-1, 2]:
                        new_params[i, 2] = new_params[i-1, 2] + 0.1
                
                area = -_objective(new_params, px, py)
                if area > current_area:
                    current = new_params
                    current_area = area
                    if current_area > best_area:
                        best_area = current_area
                        best_params = current.copy()
                    # Reduce step size on success
                    step_size *= 0.9
                else:
                    step_size *= 0.95
        except Exception:
            pass
    
    # Strategy 4: Try different numbers of control points
    if time.time() < deadline - 40:
        for n_ctrl in [5, 7, 9]:
            if time.time() > deadline - 20:
                break
            
            # Initialize from best_params by interpolation
            if len(best_params) >= 2:
                t_old = np.linspace(0, 1, len(best_params))
                t_new = np.linspace(0, 1, n_ctrl)
                new_params = np.column_stack([
                    np.interp(t_new, t_old, best_params[:, 0]),
                    np.interp(t_new, t_old, best_params[:, 1]),
                    np.interp(t_new, t_old, best_params[:, 2]),
                ])
            else:
                new_params = np.column_stack([
                    np.linspace(0, -2, n_ctrl),
                    np.zeros(n_ctrl),
                    np.linspace(0, 90, n_ctrl),
                ])
            
            try:
                result = minimize(
                    lambda p: _objective(p.reshape(-1, 3), px, py),
                    new_params.ravel(),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 0.005, 'fatol': 1e-7, 'adaptive': True}
                )
                if hasattr(result, 'x'):
                    params = result.x.reshape(-1, 3)
                    params[:, 2] = np.clip(params[:, 2], 0, 90)
                    area = -_objective(params, px, py)
                    if area > best_area:
                        best_area = area
                        best_params = params.copy()
            except Exception:
                pass
    
    # Final: return best_params
    if len(best_params) == 0:
        return baseline
    
    return best_params


# EVOLVE-BLOCK-END
