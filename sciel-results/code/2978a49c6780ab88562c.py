import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    """
    Returns a path (n, 3) array for the moving sofa problem.
    """
    start_time = time.time()
    deadline = start_time + 590  # safety margin before 630s kill

    # Sofa grid points
    sofa_x = np.arange(-3, 1.5, 0.01)
    sofa_y = np.arange(-1.5, 1.5, 0.01)
    sofa_X, sofa_Y = np.meshgrid(sofa_x, sofa_y, indexing='ij')
    sofa_x_flat = sofa_X.ravel()
    sofa_y_flat = sofa_Y.ravel()
    n_sofa = len(sofa_x_flat)

    def _densify(path):
        """Add the evaluator's start/end and interpolate between consecutive rows."""
        if len(path) == 0:
            return np.array([[0.0, 0.0, 0.0], [-2.0, 0.0, 90.0]])
        # Ensure path is float64
        path = np.array(path, dtype=np.float64)
        full = np.vstack([[[0.0, 0.0, 0.0]], path, [[-2.0, 0.0, 90.0]]])
        result = [full[0].copy()]
        for i in range(len(full) - 1):
            p1, p2 = full[i], full[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(1, m):
                t = k / m
                result.append(p1 * (1 - t) + p2 * t)
            result.append(p2.copy())
        return np.array(result)

    def _count_inside(path):
        """Count sofa points inside all placements in the densified path."""
        full = _densify(path)
        inside = np.ones(n_sofa, dtype=bool)
        # Precompute sofa coords for speed
        sx = sofa_x_flat
        sy = sofa_y_flat
        
        for px, py, pa in full:
            a_rad = np.deg2rad(pa)
            c, s = np.cos(a_rad), np.sin(a_rad)
            dx = sx - px
            dy = sy - py
            u = c * dx + s * dy
            v = -s * dx + c * dy
            in1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            inside &= (in1 | in2)
        return np.sum(inside)

    def _score(path):
        """
        Score function for optimization.
        Returns negative count (to be minimized).
        """
        if not np.all(np.isfinite(path)):
            return 1e10
        return -_count_inside(path)

    def _arc_path(cx, cy, n_pts):
        """Generate a circular arc path from (0,0,0) to (-2,0,90) with center (cx, cy)."""
        R = np.sqrt(cx * cx + cy * cy)
        if R < 1e-10:
            return None
        a0 = np.arctan2(0.0 - cy, 0.0 - cx)
        a1 = np.arctan2(0.0 - cy, -2.0 - cx)
        
        sweep_direct = (a1 - a0) % (2 * np.pi)
        if sweep_direct > np.pi:
            sweep = sweep_direct - 2 * np.pi
        else:
            sweep = sweep_direct
            
        t_vals = np.linspace(0, 1, n_pts + 2)
        path = []
        for t in t_vals:
            angle = a0 + t * sweep
            x = cx + R * np.cos(angle)
            y = cy + R * np.sin(angle)
            a_deg = 90.0 * t
            path.append([x, y, a_deg])
        return np.array(path, dtype=np.float64)

    # --- Stage 1: Optimize Arc Center ---
    # The arc is defined by center (cx, cy).
    # We use a moderate number of points to evaluate the score.
    # The score is invariant to the number of points returned (due to densification),
    # but more points in our local representation might help the optimizer see the shape?
    # Actually, for the arc, the shape is fixed by (cx, cy).
    # We just need to evaluate the score of the arc.
    # We use n_pts=10 for evaluation to be reasonably smooth, but the final densification
    # will handle the rest.
    
    def objective_arc(params):
        cx, cy = params
        # Constrain center to a reasonable region
        if cx < -5 or cx > 5 or cy < -5 or cy > 5:
            return 1e10
        path = _arc_path(cx, cy, 10)
        if path is None:
            return 1e10
        return _score(path)

    # Initial guesses for arc center
    # The path goes from (0,0) to (-2,0). The center should be somewhat "above" or "below" the segment.
    # Try a few initial points.
    best_arc_score = 1e10
    best_arc_params = None
    
    initial_guesses = [
        [-1.0, 1.0],
        [-1.0, -1.0],
        [-1.0, 2.0],
        [-1.0, -2.0],
        [0.0, 1.0],
        [-2.0, 1.0]
    ]
    
    for guess in initial_guesses:
        try:
            res = minimize(objective_arc, guess, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-3, 'adaptive': True})
            if res.fun < best_arc_score:
                best_arc_score = res.fun
                best_arc_params = res.x
        except Exception:
            pass
        
        if time.time() > deadline - 100:
            break

    if best_arc_params is None:
        # Fallback to a known good path
        best_path = np.array([[0.0, 0.0, 0.0], [-1.0, 1.0, 45.0], [-2.0, 0.0, 90.0]])
    else:
        best_path = _arc_path(best_arc_params[0], best_arc_params[1], 10)
        if best_path is None:
            best_path = np.array([[0.0, 0.0, 0.0], [-1.0, 1.0, 45.0], [-2.0, 0.0, 90.0]])

    # --- Stage 2: Optimize Free Path Points ---
    # We will optimize the coordinates of the intermediate points.
    # The angles are fixed at linear interpolation.
    # We try different numbers of points K.
    
    best_overall_score = _score(best_path)
    best_overall_path = best_path

    for K in [5, 9, 15, 21]:
        if time.time() > deadline - 60:
            break
            
        # Initialize with the current best path resampled to K points
        # We need to extract the intermediate points from best_path
        # best_path has shape (N, 3). We want K intermediate points.
        # The angles of best_path are not necessarily uniform if it was optimized freely?
        # Wait, in Stage 1, the arc path has uniform angles.
        # In Stage 2, we fix the angles to be uniform: 0, 90/K, ..., 90.
        # We initialize x, y from the current best path by interpolation.
        
        # Create target angles
        target_angles = np.linspace(0, 90, K + 2)
        
        # Interpolate x, y from best_path to target_angles
        # best_path might have non-uniform angles? 
        # If best_path came from _arc_path, angles are uniform.
        # If best_path was modified in a previous Stage 2 iteration, angles are uniform.
        # So we can just index.
        
        # If best_path has different number of points, we need to interpolate.
        # Let's assume best_path has uniform angles for now.
        # If not, we use np.interp.
        
        if len(best_path) == K + 2:
            init_x = best_path[1:-1, 0]
            init_y = best_path[1:-1, 1]
        else:
            # Interpolate
            init_x = np.interp(target_angles[1:-1], best_path[:, 2], best_path[:, 0])
            init_y = np.interp(target_angles[1:-1], best_path[:, 2], best_path[:, 1])
        
        init_params = np.hstack([init_x, init_y])
        
        def objective_free(params):
            x_pts = params[:K]
            y_pts = params[K:]
            # Construct path
            path = np.empty((K + 2, 3))
            path[0] = [0.0, 0.0, 0.0]
            path[-1] = [-2.0, 0.0, 90.0]
            path[1:-1, 0] = x_pts
            path[1:-1, 1] = y_pts
            path[1:-1, 2] = target_angles[1:-1]
            
            # Basic validity check: points should not be too far away
            if np.any(np.abs(x_pts) > 10) or np.any(np.abs(y_pts) > 10):
                return 1e10
                
            return _score(path)
        
        try:
            res = minimize(objective_free, init_params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-3, 'fatol': 1e-3, 'adaptive': True})
            if res.fun < best_overall_score:
                best_overall_score = res.fun
                x_pts = res.x[:K]
                y_pts = res.x[K:]
                best_overall_path = np.empty((K + 2, 3))
                best_overall_path[0] = [0.0, 0.0, 0.0]
                best_overall_path[-1] = [-2.0, 0.0, 90.0]
                best_overall_path[1:-1, 0] = x_pts
                best_overall_path[1:-1, 1] = y_pts
                best_overall_path[1:-1, 2] = target_angles[1:-1]
        except Exception:
            pass

    # --- Stage 3: Final Polish ---
    # Do a few iterations of small perturbations if time remains
    if time.time() < deadline - 10:
        current = best_overall_path.copy()
        n = len(current)
        # Only perturb intermediate points
        for _ in range(50): # Limit iterations
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(1, n - 1):
                if time.time() > deadline - 10:
                    break
                for dx in [-0.01, 0.01]:
                    for dy in [-0.01, 0.01]:
                        trial = current.copy()
                        trial[i, 0] += dx
                        trial[i, 1] += dy
                        s = _score(trial)
                        if s < best_overall_score:
                            best_overall_score = s
                            best_overall_path = trial
                            current = trial
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    # Return the intermediate points only (evaluator adds start/end)
    # The task says: "returning a float array of shape (n, 3)... Row (x, y, a) places...
    # The evaluator puts (0, 0, 0) before your rows and (-2, 0, 90) after them."
    # So we should return the points *between* start and end.
    
    # best_overall_path includes start and end.
    if len(best_overall_path) > 2:
        result = best_overall_path[1:-1]
    else:
        # If we only have start and end, return an empty array? 
        # "n >= 1"
        # If n=1, it's one point.
        # If we return empty, evaluator adds start/end.
        # But contract says n>=1.
        # Let's return the midpoint if we have nothing else.
        result = np.array([[-1.0, 0.0, 45.0]])

    if not np.all(np.isfinite(result)):
        result = np.array([[-1.0, 0.0, 45.0]])

    return result

# EVOLVE-BLOCK-END
