# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 610.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _base_inside(u, v):
    """Boolean: is (u,v) in the base L-shaped corridor?"""
    horiz = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
    vert = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    return horiz | vert


def _build_sofa_grid():
    """Return (X, Y) for the sofa grid (450 x 300)."""
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    return X, Y


def _densify_path(path):
    """Densify path per evaluator rules: add linear interpolations between consecutive rows."""
    p = np.vstack([np.array([[0.0, 0.0, 0.0]]), path, np.array([[-2.0, 0.0, 90.0]])])
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


def _eval_path_vectorized(path, X, Y):
    """Vectorized evaluation: returns number of sofa grid points inside EVERY placement."""
    dense = _densify_path(path)
    n_placements = len(dense)
    
    # Precompute for each placement: rotation and translation
    # For placement (cx, cy, a_deg): u = c*dx + s*dy, v = -s*dx + c*dy
    # where dx = X - cx, dy = Y - cy
    
    # We'll process placements in chunks to manage memory
    # X, Y have shape (300, 450) = 135000 points
    # For n placements, we need (n, 300, 450) for u and v
    
    chunk_size = max(1, min(n_placements, 200))
    inside = np.ones(X.shape, dtype=bool)
    
    for start in range(0, n_placements, chunk_size):
        end = min(start + chunk_size, n_placements)
        chunk = dense[start:end]
        n_chunk = end - start
        
        cx = chunk[:, 0]  # (n_chunk,)
        cy = chunk[:, 1]
        a_rad = np.deg2rad(chunk[:, 2])
        c = np.cos(a_rad)  # (n_chunk,)
        s = np.sin(a_rad)  # (n_chunk,)
        
        # For each placement, compute u and v for all grid points
        # dx = X - cx[None, None, :], dy = Y - cy[None, None, :]
        # u = c[None,None,:] * dx + s[None,None,:] * dy
        # v = -s[None,None,:] * dx + c[None,None,:] * dy
        
        dx = X[None, :, :] - cx[:, None, None]  # (n_chunk, 300, 450)
        dy = Y[None, :, :] - cy[:, None, None]  # (n_chunk, 300, 450)
        
        u = c[:, None, None] * dx + s[:, None, None] * dy
        v = -s[:, None, None] * dx + c[:, None, None] * dy
        
        # Check base inside for all placements at once
        horiz = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        vert = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        mask = horiz | vert  # (n_chunk, 300, 450)
        
        # A point is inside if it's inside ALL placements in this chunk
        chunk_inside = mask.all(axis=0)  # (300, 450)
        inside &= chunk_inside
        
        if not inside.any():
            break
    
    return int(inside.sum())


def _score(path, X, Y):
    return _eval_path_vectorized(path, X, Y) * 0.0001


def _circular_arc_path(c, n_points=10):
    """Generate a path along a circular arc with center (-1, c).
    
    The arc goes from (0,0) to (-2,0) with the corridor angle
    following the tangent to the arc.
    """
    cx = -1.0
    cy = c
    r = np.sqrt(1.0 + c * c)
    
    # Angles for start and end points on the circle
    theta_s = np.arctan2(0.0 - cy, 0.0 - cx)  # angle from center to start
    theta_e = np.arctan2(0.0 - cy, -2.0 - cx)  # angle from center to end
    
    # Ensure we go the short way around
    if theta_e < theta_s:
        theta_e += 2 * np.pi
    elif theta_s < theta_e - np.pi:
        theta_s -= 2 * np.pi
        if theta_e < theta_s:
            theta_e += 2 * np.pi
    
    # Sample points along the arc
    thetas = np.linspace(theta_s, theta_e, n_points)
    xs = cx + r * np.cos(thetas)
    ys = cy + r * np.sin(thetas)
    
    # Corridor angle: tangent to the arc (perpendicular to radius)
    # Tangent direction at angle theta: (-sin(theta), cos(theta))
    # Angle of tangent in degrees
    tangent_angles = np.degrees(np.arctan2(np.cos(thetas), -np.sin(thetas)))
    # Normalize to [0, 90] range
    tangent_angles = np.mod(tangent_angles, 360.0)
    # Map to [0, 90]
    tangent_angles = np.where(tangent_angles > 180, tangent_angles - 360, tangent_angles)
    tangent_angles = np.where(tangent_angles < 0, tangent_angles + 360, tangent_angles)
    # We want angles in [0, 90]
    tangent_angles = np.clip(tangent_angles, 0, 90)
    
    return np.column_stack([xs, ys, tangent_angles])


def _initial_paths():
    """Return multiple initial path candidates."""
    paths = []
    
    # Path 1: Original seed-like path
    paths.append(np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64))
    
    # Path 2: More uniform angle spacing
    angles = np.linspace(10, 80, 7)
    x = np.linspace(-0.1, -0.7, 7)
    y = np.full(7, 0.25)
    paths.append(np.column_stack([x, y, angles]))
    
    # Path 3: Curved path
    t = np.linspace(0, 1, 8)
    angles = 10 + 70 * t
    x = -0.1 - 0.6 * t
    y = 0.1 + 0.3 * np.sin(np.pi * t)
    paths.append(np.column_stack([x, y, angles]))
    
    # Path 4: Steep early, gentle late
    angles = np.array([12, 25, 40, 50, 60, 70, 80, 86])
    x = np.array([-0.1, -0.2, -0.35, -0.45, -0.55, -0.62, -0.68, -0.72])
    y = np.array([0.15, 0.2, 0.25, 0.28, 0.3, 0.28, 0.25, 0.2])
    paths.append(np.column_stack([x, y, angles]))
    
    # Path 5: S-curve
    t = np.linspace(0, 1, 9)
    angles = 10 + 75 * t
    x = -0.1 - 0.6 * t
    y = 0.2 + 0.15 * np.sin(2 * np.pi * t)
    paths.append(np.column_stack([x, y, angles]))
    
    # Path 6-10: Circular arc paths with different centers
    for c in [0.0, 0.5, 1.0, 1.5, 2.0]:
        try:
            arc_path = _circular_arc_path(c, n_points=10)
            if np.all(np.isfinite(arc_path)):
                paths.append(arc_path)
        except Exception:
            pass
    
    return paths


def _nelder_mead(f, x0, max_iter=2000, step=0.05, tol=1e-6):
    """Nelder-Mead minimization."""
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        pt = x0.copy()
        pt[i] += step
        simplex[i + 1] = pt
    vals = np.array([f(simplex[i]) for i in range(n + 1)])
    
    for it in range(max_iter):
        if _elapsed() > _DEADLINE - 20:
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
    
    best_i = int(np.argmin(vals))
    return simplex[best_i], vals[best_i]


def _optimize_angles_xy(path, X, Y, max_iter=1500):
    """Optimize both angles and (x,y) for a given path structure."""
    n = len(path)
    # Parameters: x[0..n-1], y[0..n-1], angles[0..n-1]
    x0 = np.concatenate([path[:, 0], path[:, 1], path[:, 2]])
    
    def f(flat):
        x = flat[0:n]
        y = flat[n:2*n]
        angles = flat[2*n:3*n]
        # Ensure angles are in (0, 90)
        angles = np.clip(angles, 1.0, 89.0)
        p = np.column_stack([x, y, angles])
        return -_score(p, X, Y)
    
    xopt, fopt = _nelder_mead(f, x0, max_iter=max_iter, step=0.05)
    x = xopt[0:n]
    y = xopt[n:2*n]
    angles = np.clip(xopt[2*n:3*n], 1.0, 89.0)
    return np.column_stack([x, y, angles])


def _coordinate_descent(path, X, Y, steps=3, step_size=0.02):
    """Coordinate descent refinement."""
    current = path.copy()
    n = len(current)
    
    for _ in range(steps):
        if _elapsed() > _DEADLINE - 10:
            break
        improved = False
        for i in range(n):
            for coord in range(3):
                # Try small perturbations
                best_val = _score(current, X, Y)
                best_coord = current[i, coord]
                for delta in [-step_size, -step_size/2, step_size/2, step_size]:
                    trial = current.copy()
                    trial[i, coord] = current[i, coord] + delta
                    if coord == 2:
                        trial[i, coord] = np.clip(trial[i, coord], 1.0, 89.0)
                    val = _score(trial, X, Y)
                    if val > best_val:
                        best_val = val
                        best_coord = trial[i, coord]
                current[i, coord] = best_coord
                if best_coord != path[i, coord]:
                    improved = True
        if not improved:
            break
    
    return current


def moving_sofa_2d() -> np.ndarray:
    """
    Returns a path (n,3) array of corridor inner-corner placements (x,y,a_deg).
    """
    X, Y = _build_sofa_grid()
    
    # Start with best initial path
    initial_paths = _initial_paths()
    best_path = initial_paths[0]
    best_score = _score(best_path, X, Y)
    
    for p in initial_paths[1:]:
        s = _score(p, X, Y)
        if s > best_score:
            best_score = s
            best_path = p
    
    # Optimize with Nelder-Mead on (x,y) with fixed angles
    if _elapsed() < _DEADLINE - 120:
        angles_fixed = best_path[:, 2]
        n = len(angles_fixed)
        xy0 = best_path[:, :2].flatten()
        
        def f_xy(flat):
            p = np.empty((n, 3))
            p[:, 0] = flat[0::2]
            p[:, 1] = flat[1::2]
            p[:, 2] = angles_fixed
            return -_score(p, X, Y)
        
        xopt, _ = _nelder_mead(f_xy, xy0, max_iter=1500, step=0.04)
        p_opt = np.empty((n, 3))
        p_opt[:, 0] = xopt[0::2]
        p_opt[:, 1] = xopt[1::2]
        p_opt[:, 2] = angles_fixed
        s_opt = _score(p_opt, X, Y)
        if s_opt > best_score:
            best_score = s_opt
            best_path = p_opt
    
    # Try optimizing all parameters (x, y, angles)
    if _elapsed() < _DEADLINE - 100:
        p_full = _optimize_angles_xy(best_path, X, Y, max_iter=1000)
        s_full = _score(p_full, X, Y)
        if s_full > best_score:
            best_score = s_full
            best_path = p_full
    
    # Optimize circular arc parameter c
    best_c = 0.0
    if _elapsed() < _DEADLINE - 100:
        def f_arc(c_val):
            try:
                p = _circular_arc_path(c_val, n_points=10)
                if not np.all(np.isfinite(p)):
                    return 0.0
                return -_score(p, X, Y)
            except Exception:
                return 0.0
        
        # Scan for good c values
        best_arc_score = 0.0
        for c_try in np.linspace(-2.0, 3.0, 30):
            if _elapsed() > _DEADLINE - 60:
                break
            s = _score(_circular_arc_path(c_try, n_points=10), X, Y)
            if s > best_arc_score:
                best_arc_score = s
                best_c = c_try
        
        # Refine with Nelder-Mead
        if best_arc_score > best_score:
            best_score = best_arc_score
            best_path = _circular_arc_path(best_c, n_points=10)
        
        xopt_c, fopt_c = _nelder_mead(f_arc, np.array([best_c]), max_iter=500, step=0.1)
        c_opt = float(xopt_c[0])
        p_arc = _circular_arc_path(c_opt, n_points=10)
        s_arc = _score(p_arc, X, Y)
        if s_arc > best_score:
            best_score = s_arc
            best_path = p_arc
        best_c = c_opt
    
    # Try different angle grids
    if _elapsed() < _DEADLINE - 80:
        angle_grids = [
            np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
            np.array([12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0]),
            np.array([15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0]),
            np.array([8.0, 18.0, 28.0, 38.0, 48.0, 58.0, 68.0, 78.0, 86.0]),
        ]
        
        for angles_grid in angle_grids:
            if _elapsed() > _DEADLINE - 40:
                break
            n = len(angles_grid)
            # Interpolate x,y from best_path to these angles
            x_interp = np.interp(angles_grid, best_path[:, 2], best_path[:, 0])
            y_interp = np.interp(angles_grid, best_path[:, 2], best_path[:, 1])
            p_init = np.column_stack([x_interp, y_interp, angles_grid])
            
            s_init = _score(p_init, X, Y)
            if s_init > best_score:
                best_score = s_init
                best_path = p_init
            
            # Optimize x,y for this angle grid
            if _elapsed() > _DEADLINE - 30:
                continue
            xy0 = p_init[:, :2].flatten()
            
            def f_grid(flat, ang=angles_grid, nn=n):
                p = np.empty((nn, 3))
                p[:, 0] = flat[0::2]
                p[:, 1] = flat[1::2]
                p[:, 2] = ang
                return -_score(p, X, Y)
            
            xopt, _ = _nelder_mead(f_grid, xy0, max_iter=800, step=0.03)
            p_opt = np.empty((n, 3))
            p_opt[:, 0] = xopt[0::2]
            p_opt[:, 1] = xopt[1::2]
            p_opt[:, 2] = angles_grid
            s_opt = _score(p_opt, X, Y)
            if s_opt > best_score:
                best_score = s_opt
                best_path = p_opt
    
    # Try different numbers of points on arc
    if _elapsed() < _DEADLINE - 30:
        c_default = 0.0
        if 'best_c' in locals():
            c_default = best_c
        for n_pts in [7, 8, 9, 11, 12, 15]:
            if _elapsed() > _DEADLINE - 15:
                break
            p_arc = _circular_arc_path(c_default, n_points=n_pts)
            s_arc = _score(p_arc, X, Y)
            if s_arc > best_score:
                best_score = s_arc
                best_path = p_arc
    
    # Coordinate descent refinement
    if _elapsed() < _DEADLINE - 20:
        p_cd = _coordinate_descent(best_path, X, Y, steps=2, step_size=0.015)
        s_cd = _score(p_cd, X, Y)
        if s_cd > best_score:
            best_score = s_cd
            best_path = p_cd
    
    # Final sanity check
    best_path = np.asarray(best_path, dtype=np.float64)
    if best_path.ndim != 2 or best_path.shape[1] != 3 or best_path.shape[0] < 1:
        best_path = _initial_paths()[0]
    if not np.all(np.isfinite(best_path)):
        best_path = _initial_paths()[0]
    
    return best_path


# EVOLVE-BLOCK-END
