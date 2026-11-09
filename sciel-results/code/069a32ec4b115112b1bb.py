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

def _eval_path_score(path, X, Y, early_stop=True):
    """
    Calculate score for a path.
    path: (n, 3) array.
    Returns score (number of points * 0.0001).
    """
    # Add evaluator endpoints
    p = np.vstack([np.array([[0.0, 0.0, 0.0]]), path, np.array([[-2.0, 0.0, 90.0]])])
    
    # Densify path
    dense = _densify_path(p)
    
    if len(dense) == 0:
        return 0.0
        
    inside = np.ones(X.shape, dtype=bool)
    
    for (cx, cy, a_deg) in dense:
        a = np.deg2rad(a_deg)
        c, s = np.cos(a), np.sin(a)
        dx = X - cx
        dy = Y - cy
        u = c * dx + s * dy
        v = -s * dx + c * dy
        mask = _base_inside(u, v)
        inside &= mask
        if early_stop and not inside.any():
            break
            
    return np.sum(inside) * 0.0001

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

def _arc_path_params(rc, cx, cy, n_pts=8):
    """Generate a path along a circular arc.
    
    The corridor inner corner follows a circular arc from (0,0) at angle 0 to (-2,0) at angle 90.
    Parameters:
    - rc: radius of the arc
    - cx, cy: center of the arc
    - n_pts: number of intermediate points
    
    The arc must be tangent to the start and end orientations.
    Start: corner at (0,0), rotation 0 deg. The corridor extends in +u (right) and -v (down) directions.
    End: corner at (-2,0), rotation 90 deg. The corridor extends in +u (up) and -v (left) directions.
    
    For a circular arc with center (cx, cy) and radius rc:
    - Start point (0,0) is on the arc: (0-cx)^2 + (0-cy)^2 = rc^2
    - End point (-2,0) is on the arc: (-2-cx)^2 + (0-cy)^2 = rc^2
    - The tangent at start should be in direction (1,0) [rotation 0 means corridor +u axis is world +x]
    - The tangent at end should be in direction (0,1) [rotation 90 means corridor +u axis is world +y]
    
    From the two distance equations:
    cx^2 + cy^2 = rc^2
    (cx+2)^2 + cy^2 = rc^2
    => cx^2 + cy^2 = cx^2 + 4cx + 4 + cy^2
    => 0 = 4cx + 4
    => cx = -1
    
    So cx = -1, and cy^2 = rc^2 - 1, so rc >= 1.
    cy = ±sqrt(rc^2 - 1)
    
    For the tangent condition:
    At start (0,0), the radius vector from center (-1, cy) to (0,0) is (1, -cy).
    The tangent should be perpendicular to this radius.
    Tangent direction options: (-(-cy), 1) = (cy, 1) normalized, or (-cy, -1) normalized.
    
    We want the corridor to move in the -x direction overall, so the tangent should have negative x component.
    If cy > 0: radius vector is (1, -cy), tangent perpendicular is (cy, 1) or (-cy, -1).
    We want tangent pointing roughly in -x direction, so (-cy, -1) if cy > 0.
    But we need the tangent to align with rotation 0, which means +u axis is world +x.
    Actually, the tangent of the path (direction of motion) should be such that the rotation increases.
    
    Let me think differently. The rotation a goes from 0 to 90 degrees.
    The position follows the arc. The angle of the corridor at each point should match the tangent direction.
    
    For a circular arc, if we parameterize by angle theta from the center:
    Position: (cx + rc*cos(theta), cy + rc*sin(theta))
    The tangent direction is (-rc*sin(theta), rc*cos(theta)) which has angle (theta + 90) degrees.
    
    At start: position (0,0), rotation 0. So the corridor +u axis (which is the tangent direction for the "forward" motion) should be at 0 degrees, i.e., pointing in +x.
    Tangent angle = theta_start + 90 = 0 => theta_start = -90 degrees.
    Position: (cx + rc*cos(-90), cy + rc*sin(-90)) = (cx, cy - rc) = (0, 0)
    So cx = 0, cy = rc. But we already found cx = -1. Contradiction.
    
    Let me reconsider. The rotation a is the angle of the corridor, not necessarily the tangent of the path.
    The path just needs to go from (0,0,0) to (-2,0,90).
    
    Let's just use a parametric form:
    t in [0, 1]
    x(t) = -2t + cx_offset
    y(t) = cy_offset * sin(pi * t)  (or some smooth function)
    a(t) = 90 * t
    
    But the directive says to use a circular arc model. Let me try a different approach.
    
    Use parameters:
    - h: height of the arc (maximum y value)
    - s: shape parameter (0 = straight line in y, 1 = parabolic)
    
    x(t) = -2t
    y(t) = h * (4*t*(1-t))  (parabolic, max at t=0.5)
    a(t) = 90*t
    
    This gives a smooth path. Parameters: h (and maybe add a cubic term).
    """
    t = np.linspace(0, 1, n_pts + 2)[1:-1]  # exclude endpoints
    # Parabolic arc: y = h * 4*t*(1-t)
    x = -2.0 * t
    y = rc * 4.0 * t * (1.0 - t)  # rc controls height
    a = 90.0 * t
    return np.column_stack([x, y, a])

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

def _optimize_arc_parameter(X, Y, h_init=0.5, max_iter=200, deadline=_DEADLINE - 30):
    """Optimize the arc height parameter for a parametric path."""
    def f(h_flat):
        h = h_flat[0]
        if h < 0:
            h = 0
        path = _arc_path_params(h, 0, 0, n_pts=8)
        return -_eval_path_score_fast(path, X, Y)
    
    xopt, _ = _nelder_mead(f, np.array([h_init]), max_iter=max_iter, step=0.1, deadline=deadline)
    h_opt = max(0, xopt[0])
    path = _arc_path_params(h_opt, 0, 0, n_pts=8)
    return path, _eval_path_score_fast(path, X, Y)

def moving_sofa_2d() -> np.ndarray:
    # Use subsampled grid for optimization
    X_opt, Y_opt = _build_grid(subsample=4)
    X_full, Y_full = _build_grid(subsample=1)
    
    # Start with best initial path
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
    
    # Try arc-based paths with different heights
    for h_init in [0.2, 0.4, 0.6, 0.8, 1.0]:
        if _elapsed() > _DEADLINE - 90:
            break
        arc_path, arc_score = _optimize_arc_parameter(X_opt, Y_opt, h_init=h_init, max_iter=100)
        if arc_score > best_score:
            best_path = arc_path
            best_score = arc_score
    
    # Optimize each initial path
    for i, path in enumerate(init_paths):
        if _elapsed() > _DEADLINE - 60:
            break
        if len(path) != len(best_path):
            # Only optimize paths with same length as current best
            continue
        
        refined = _refine_path(path, X_opt, Y_opt, max_iter=400, step=0.05)
        score = _eval_path_score_fast(refined, X_opt, Y_opt)
        if score > best_score:
            best_path = refined
            best_score = score
    
    # Try different path lengths
    for n_pts in [3, 4, 5, 6, 7, 8, 9]:
        if _elapsed() > _DEADLINE - 45:
            break
        
        # Create initial path with n_pts by interpolating best_path
        if len(best_path) != n_pts:
            ts = np.linspace(0, 1, n_pts)
            init_pts = np.zeros((n_pts, 3))
            for i in range(n_pts):
                t = ts[i] * (len(best_path) - 1)
                idx = int(t)
                frac = t - idx
                if idx >= len(best_path) - 1:
                    init_pts[i] = best_path[min(idx, len(best_path) - 1)]
                else:
                    init_pts[i] = best_path[idx] * (1 - frac) + best_path[idx + 1] * frac
        else:
            init_pts = best_path.copy()
        
        refined = _refine_path(init_pts, X_opt, Y_opt, max_iter=300, step=0.04)
        score = _eval_path_score_fast(refined, X_opt, Y_opt)
        if score > best_score:
            best_path = refined
            best_score = score
    
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
    
    # Second round of refinement if time permits
    if _elapsed() < _DEADLINE - 15:
        refined_full2 = _refine_path(best_path, X_full, Y_full, max_iter=100, step=0.008, deadline=_DEADLINE - 5)
        score_full2 = _eval_path_score_fast(refined_full2, X_full, Y_full)
        score_best_full2 = _eval_path_score_fast(best_path, X_full, Y_full)
        if score_full2 > score_best_full2:
            best_path = refined_full2
    
    # Ensure valid output
    best_path = np.asarray(best_path, dtype=np.float64)
    if best_path.ndim != 2 or best_path.shape[1] != 3 or best_path.shape[0] < 1:
        best_path = init_paths[0]
    if not np.all(np.isfinite(best_path)):
        best_path = init_paths[0]
        
    return best_path
# EVOLVE-BLOCK-END
