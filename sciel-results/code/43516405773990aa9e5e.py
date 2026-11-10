# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 620.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _base_inside(u, v):
    """Vectorized base corridor check."""
    horiz = (u >= -1e-9) & (u <= 1.0 + 1e-9) & (v >= -3.0 - 1e-9) & (v <= 1.0 + 1e-9)
    vert = (u >= -3.0 - 1e-9) & (u <= 1.0 + 1e-9) & (v >= -1e-9) & (v <= 1.0 + 1e-9)
    return horiz | vert

def _build_grid(subsample=1):
    """Build sofa grid. subsample=1 for full, >1 for subsampled."""
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    if subsample > 1:
        xs = xs[::subsample]
        ys = ys[::subsample]
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    return X, Y

def _densify_path(p):
    """Densify path according to evaluator rules."""
    dense = [p[0]]
    for i in range(len(p) - 1):
        a1 = p[i, 2]
        a2 = p[i + 1, 2]
        m = max(1, int(np.ceil(abs(a2 - a1) - 1e-9)))
        if m > 1:
            ts = np.linspace(0.0, 1.0, m + 1)
            for t in ts[1:-1]:
                row = p[i] * (1.0 - t) + p[i + 1] * t
                dense.append(row)
        dense.append(p[i + 1])
    return np.array(dense)

def _eval_path_score_fast(path, X, Y):
    """Faster evaluation using only surviving points."""
    p = np.vstack([np.array([[0.0, 0.0, 0.0]]), path, np.array([[-2.0, 0.0, 90.0]])])
    dense = _densify_path(p)
    
    if len(dense) == 0:
        return 0.0
    
    # Track surviving indices
    alive = np.ones(X.shape, dtype=bool)
    
    for (cx, cy, a_deg) in dense:
        if not alive.any():
            break
        a = np.deg2rad(a_deg)
        c, s = np.cos(a), np.sin(a)
        dx = X - cx
        dy = Y - cy
        u = c * dx + s * dy
        v = -s * dx + c * dy
        mask = _base_inside(u, v)
        alive &= mask
        
    return np.sum(alive) * 0.0001

def _arc_path_params(r, cx, cy, n_samples=20):
    """
    Generate path from circular arc parameters.
    The arc is tangent to start (0,0,0) and end (-2,0,90).
    r: radius of arc
    cx, cy: center of arc
    n_samples: number of points along the arc
    Returns (n_samples, 3) array.
    """
    # Start point is (0,0) with angle 0, end point is (-2,0) with angle 90
    # The arc center must be at distance r from both start and end positions
    # with appropriate tangent directions
    
    # For a circle of radius r centered at (cx, cy):
    # Start tangent at (0,0) should be horizontal (angle 0)
    # End tangent at (-2,0) should be vertical (angle 90)
    
    # The tangent direction at a point on a circle is perpendicular to the radius
    # At start (0,0): radius vector is (cx-0, cy-0) = (cx, cy)
    # Tangent should be (1, 0) or (-1, 0) - perpendicular to (cx, cy)
    # So cx * 1 + cy * 0 = 0 => cx = 0, or we need the perpendicular
    
    # Actually, let's parameterize differently. 
    # The arc goes from (0,0) to (-2,0), turning 90 degrees.
    # If we use a circular arc, the center must satisfy:
    # distance from center to (0,0) = r
    # distance from center to (-2,0) = r
    # tangent at (0,0) is horizontal
    # tangent at (-2,0) is vertical
    
    # From distance constraints:
    # cx^2 + cy^2 = r^2
    # (cx+2)^2 + cy^2 = r^2
    # Subtracting: (cx+2)^2 - cx^2 = 0 => 4cx + 4 = 0 => cx = -1
    
    # So cx = -1 always! Then cy^2 = r^2 - 1, so cy = ±sqrt(r^2-1)
    # For the arc to turn the right way (counterclockwise from 0 to 90), 
    # we need cy > 0 (center above the x-axis)
    
    if r < 1.0 - 1e-6:
        # Degenerate case, return a straight-ish path
        ts = np.linspace(0, 1, n_samples)
        path = np.zeros((n_samples, 3))
        path[:, 0] = -2.0 * ts
        path[:, 1] = 0.0
        path[:, 2] = 90.0 * ts
        return path
    
    cx_val = -1.0
    cy_val = np.sqrt(r * r - 1.0)
    
    # Angle from center to start point (0,0)
    angle_start = np.arctan2(0 - cy_val, 0 - cx_val)
    # Angle from center to end point (-2,0)
    angle_end = np.arctan2(0 - cy_val, -2 - cx_val)
    
    # We need to go from angle_start to angle_end in the right direction
    # The arc should sweep 90 degrees (pi/2 radians)
    # Check which direction
    delta_angle = angle_end - angle_start
    if delta_angle < 0:
        delta_angle += 2 * np.pi
    
    # Sample angles along the arc
    angles = angle_start + np.linspace(0, delta_angle, n_samples)
    
    # Position on circle
    xs = cx_val + r * np.cos(angles)
    ys = cy_val + r * np.sin(angles)
    
    # Angle of corridor at each point (tangent direction)
    # Tangent is perpendicular to radius, rotated 90 degrees counterclockwise
    # For counterclockwise motion, tangent angle = angle + pi/2
    corridor_angles = angles + np.pi / 2
    corridor_angles_deg = np.degrees(corridor_angles)
    
    # Normalize angles to be in reasonable range
    corridor_angles_deg = np.mod(corridor_angles_deg, 360)
    # Adjust to be between 0 and 90 approximately
    # The start should be near 0, end near 90
    if corridor_angles_deg[0] > 180:
        corridor_angles_deg -= 360
    if corridor_angles_deg[-1] > 180:
        corridor_angles_deg -= 360
    
    path = np.column_stack([xs, ys, corridor_angles_deg])
    return path

def _eval_arc_score(params, X, Y):
    """Evaluate score for arc parameters [r, cx_offset, cy_offset]."""
    r = max(1.01, params[0])  # Ensure r > 1
    # cx is always -1 for the geometric constraint, but we can add small offsets
    cx_off = params[1] if len(params) > 1 else 0.0
    cy_off = params[2] if len(params) > 2 else 0.0
    
    # Generate path with n_samples based on time
    n_samples = 15
    path = _arc_path_params(r, -1.0 + cx_off, np.sqrt(r*r-1) + cy_off, n_samples)
    
    return -_eval_path_score_fast(path, X, Y)

def _initial_paths():
    """Generate multiple initial path candidates."""
    paths = []
    
    # Path 1: Arc-like path
    paths.append(np.array([
        [-0.4, 0.4, 20.0],
        [-0.9, 0.5, 45.0],
        [-1.4, 0.3, 65.0],
        [-1.8, 0.1, 80.0],
    ], dtype=np.float64))
    
    # Path 2: More gradual
    paths.append(np.array([
        [-0.3, 0.3, 15.0],
        [-0.7, 0.4, 30.0],
        [-1.1, 0.4, 50.0],
        [-1.5, 0.2, 70.0],
        [-1.8, 0.0, 85.0],
    ], dtype=np.float64))
    
    # Path 3: Keep y higher longer
    paths.append(np.array([
        [-0.5, 0.6, 25.0],
        [-1.0, 0.6, 50.0],
        [-1.5, 0.4, 70.0],
        [-1.9, 0.1, 85.0],
    ], dtype=np.float64))
    
    # Path 4: S-shaped
    paths.append(np.array([
        [-0.4, 0.2, 18.0],
        [-0.8, 0.5, 35.0],
        [-1.2, 0.5, 55.0],
        [-1.6, 0.3, 72.0],
        [-1.9, 0.0, 88.0],
    ], dtype=np.float64))
    
    # Path 5: 6 points
    paths.append(np.array([
        [-0.3, 0.3, 12.0],
        [-0.6, 0.4, 25.0],
        [-0.9, 0.5, 40.0],
        [-1.2, 0.5, 55.0],
        [-1.5, 0.3, 70.0],
        [-1.8, 0.1, 85.0],
    ], dtype=np.float64))
    
    # Path 6: 7 points - more detailed
    paths.append(np.array([
        [-0.2, 0.2, 10.0],
        [-0.5, 0.4, 22.0],
        [-0.8, 0.5, 35.0],
        [-1.1, 0.5, 48.0],
        [-1.4, 0.4, 62.0],
        [-1.7, 0.2, 76.0],
        [-1.9, 0.0, 88.0],
    ], dtype=np.float64))
    
    # Path 7: 8 points - very detailed
    paths.append(np.array([
        [-0.15, 0.2, 8.0],
        [-0.4, 0.35, 18.0],
        [-0.65, 0.5, 28.0],
        [-0.9, 0.55, 40.0],
        [-1.15, 0.5, 52.0],
        [-1.4, 0.35, 65.0],
        [-1.65, 0.15, 78.0],
        [-1.9, 0.0, 88.0],
    ], dtype=np.float64))
    
    # Path 8: 9 points - most detailed
    paths.append(np.array([
        [-0.1, 0.15, 6.0],
        [-0.3, 0.3, 14.0],
        [-0.5, 0.45, 24.0],
        [-0.7, 0.55, 34.0],
        [-0.9, 0.55, 44.0],
        [-1.1, 0.5, 54.0],
        [-1.35, 0.35, 66.0],
        [-1.6, 0.2, 78.0],
        [-1.9, 0.0, 88.0],
    ], dtype=np.float64))
    
    return paths

def _nelder_mead(f, x0, max_iter=2000, step=0.1, tol=1e-5, deadline=_DEADLINE - 30):
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        pt = x0.copy()
        pt[i] += step
        simplex[i + 1] = pt
    vals = np.array([f(simplex[i]) for i in range(n + 1)])
    
    for it in range(max_iter):
        if _elapsed() > deadline:
            break
        order = np.argsort(vals)
        simplex = simplex[order]
        vals = vals[order]
        if abs(vals[0] - vals[-1]) < tol:
            break
            
        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                vals[-1] = fe
            else:
                simplex[-1] = xr
                vals[-1] = fr
        elif fr < vals[-2]:
            simplex[-1] = xr
            vals[-1] = fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1] = xc
                vals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    vals[i] = f(simplex[i])
                    
    return simplex[int(np.argmin(vals))], np.min(vals)

def _refine_path(path, X, Y, max_iter=300, step=0.03, deadline=_DEADLINE - 30):
    """Refine a path using Nelder-Mead."""
    x0 = path.flatten()
    
    def f(flat):
        p = flat.reshape(-1, 3)
        return -_eval_path_score_fast(p, X, Y)
    
    xopt, _ = _nelder_mead(f, x0, max_iter=max_iter, step=step, deadline=deadline)
    return xopt.reshape(-1, 3)

def _optimize_arc(X, Y, deadline=_DEADLINE - 30):
    """Optimize arc parameters using Nelder-Mead."""
    # Start with reasonable initial parameters
    # r should be around 1.5-3.0 for good performance
    x0 = np.array([2.0, 0.0, 0.0])
    
    def f(params):
        if _elapsed() > deadline:
            return 0.0
        return _eval_arc_score(params, X, Y)
    
    xopt, fval = _nelder_mead(f, x0, max_iter=500, step=0.3, tol=1e-6, deadline=deadline)
    
    # Generate path from optimized parameters
    r = max(1.01, xopt[0])
    cx_off = xopt[1] if len(xopt) > 1 else 0.0
    cy_off = xopt[2] if len(xopt) > 2 else 0.0
    
    n_samples = 15
    path = _arc_path_params(r, -1.0 + cx_off, np.sqrt(r*r-1) + cy_off, n_samples)
    
    return path, -fval

def _random_restart_refine(best_path, X, Y, n_restarts=5, max_iter=200, step=0.05, deadline=_DEADLINE - 30):
    """Try random perturbations of the best path."""
    best = best_path.copy()
    best_score = _eval_path_score_fast(best, X, Y)
    
    for r in range(n_restarts):
        if _elapsed() > deadline:
            break
        # Random perturbation
        perturbed = best + np.random.randn(*best.shape) * 0.05
        # Ensure angles are in reasonable range
        perturbed[:, 2] = np.clip(perturbed[:, 2], -10, 100)
        
        refined = _refine_path(perturbed, X, Y, max_iter=max_iter, step=step, deadline=deadline)
        score = _eval_path_score_fast(refined, X, Y)
        if score > best_score:
            best = refined
            best_score = score
            
    return best, best_score

def _polish_path(path, X, Y, deadline=_DEADLINE - 10):
    """High-precision polish pass with tighter tolerance."""
    x0 = path.flatten()
    
    def f(flat):
        p = flat.reshape(-1, 3)
        return -_eval_path_score_fast(p, X, Y)
    
    xopt, _ = _nelder_mead(f, x0, max_iter=1000, step=0.01, tol=1e-7, deadline=deadline)
    return xopt.reshape(-1, 3)

def moving_sofa_2d() -> np.ndarray:
    # Use subsampled grid for optimization
    X_opt, Y_opt = _build_grid(subsample=4)
    X_full, Y_full = _build_grid(subsample=1)
    
    # Start with arc-based optimization
    best_path = None
    best_score = 0.0
    
    # Try arc optimization
    if _elapsed() < _DEADLINE - 120:
        arc_path, arc_score = _optimize_arc(X_opt, Y_opt, deadline=_DEADLINE - 100)
        best_path = arc_path
        best_score = arc_score
    
    # If arc didn't work well, try discrete paths
    if best_score < 1.0 or _elapsed() > _DEADLINE - 120:
        init_paths = _initial_paths()
        best_path = init_paths[0]
        best_score = _eval_path_score_fast(best_path, X_opt, Y_opt)
        
        # Evaluate all initial paths
        for path in init_paths[1:]:
            if _elapsed() > _DEADLINE - 120:
                break
            score = _eval_path_score_fast(path, X_opt, Y_opt)
            if score > best_score:
                best_path = path
                best_score = score
    
    # Optimize the best discrete path
    if _elapsed() < _DEADLINE - 60:
        refined = _refine_path(best_path, X_opt, Y_opt, max_iter=400, step=0.05)
        score = _eval_path_score_fast(refined, X_opt, Y_opt)
        if score > best_score:
            best_path = refined
            best_score = score
    
    # Try arc optimization with different starting points
    for r_start in [1.5, 2.0, 2.5, 3.0]:
        if _elapsed() > _DEADLINE - 45:
            break
        try:
            arc_path2, arc_score2 = _optimize_arc(X_opt, Y_opt, deadline=_DEADLINE - 30)
            # Modify initial guess by starting from different radius
            # For now, just evaluate
            if arc_score2 > best_score:
                best_path = arc_path2
                best_score = arc_score2
        except:
            pass
    
    # Random restarts
    if _elapsed() < _DEADLINE - 30:
        best_path, best_score = _random_restart_refine(best_path, X_opt, Y_opt, n_restarts=3, max_iter=150, step=0.04, deadline=_DEADLINE - 25)
    
    # Final refinement on full grid
    if _elapsed() < _DEADLINE - 20:
        refined_full = _refine_path(best_path, X_full, Y_full, max_iter=200, step=0.015, deadline=_DEADLINE - 10)
        score_full = _eval_path_score_fast(refined_full, X_full, Y_full)
        score_best_full = _eval_path_score_fast(best_path, X_full, Y_full)
        if score_full > score_best_full:
            best_path = refined_full
    
    # High-precision polish
    if _elapsed() < _DEADLINE - 15:
        polished = _polish_path(best_path, X_full, Y_full, deadline=_DEADLINE - 5)
        score_polished = _eval_path_score_fast(polished, X_full, Y_full)
        score_best = _eval_path_score_fast(best_path, X_full, Y_full)
        if score_polished > score_best:
            best_path = polished
    
    # Ensure valid output
    best_path = np.asarray(best_path, dtype=np.float64)
    if best_path.ndim != 2 or best_path.shape[1] != 3 or best_path.shape[0] < 1:
        # Fallback to a simple path
        best_path = np.array([
            [-0.5, 0.5, 30.0],
            [-1.0, 0.5, 55.0],
            [-1.5, 0.2, 75.0],
        ], dtype=np.float64)
    if not np.all(np.isfinite(best_path)):
        best_path = np.array([
            [-0.5, 0.5, 30.0],
            [-1.0, 0.5, 55.0],
            [-1.5, 0.2, 75.0],
        ], dtype=np.float64)
        
    return best_path
# EVOLVE-BLOCK-END
