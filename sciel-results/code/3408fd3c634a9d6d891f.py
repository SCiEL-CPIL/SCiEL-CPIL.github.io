# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 615.0  # seconds, safety margin under 630s kill
_START = time.time()


def _elapsed():
    return time.time() - _START


def _base_inside(u, v):
    """Boolean: is (u,v) in the base L-shaped corridor?
    Base = {0<=u<=1, -3<=v<=1} U {-3<=u<=1, 0<=v<=1}."""
    horiz = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
    vert = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    return horiz | vert


def _build_sofa_grid():
    """Return (X, Y) for the sofa grid (450 x 300)."""
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return X, Y


def _densify_path(path):
    """Add evaluator endpoints and densify. Returns dense (m,3) array."""
    p = np.vstack([np.array([[0.0, 0.0, 0.0]]), path, np.array([[-2.0, 0.0, 90.0]])])
    dense = [p[0]]
    for i in range(len(p) - 1):
        p1 = p[i]
        p2 = p[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        if m > 1:
            for j in range(1, m):
                t = j / m
                dense.append(p1 + t * (p2 - p1))
        dense.append(p2)
    return np.array(dense)


def _eval_path_vectorized(path, X, Y):
    """
    Evaluate the sofa area for a given path.
    path: (n, 3) array of (x, y, angle_deg)
    X, Y: meshgrid arrays
    Returns: area (float)
    """
    dense = _densify_path(path)
    
    # Initialize mask to all True
    sofa_mask = np.ones(X.shape, dtype=bool)
    
    for x, y, a_deg in dense:
        a = np.deg2rad(a_deg)
        ca, sa = np.cos(a), np.sin(a)
        
        dx = X - x
        dy = Y - y
        
        # R(-a) * (w - (x,y))
        u = ca * dx + sa * dy
        v = -sa * dx + ca * dy
        
        mask = _base_inside(u, v)
        sofa_mask &= mask
        
        # Early termination if no points remain
        if not np.any(sofa_mask):
            break
    
    return np.sum(sofa_mask) * 0.0001


def _score(path, X, Y):
    """Compute score for a path."""
    try:
        area = _eval_path_vectorized(path, X, Y)
        if np.isfinite(area) and area >= 0:
            return area
    except Exception:
        pass
    return 0.0


def _cubic_bezier(t, p0, p1, p2, p3):
    """Evaluate cubic Bezier curve at parameter t (scalar or array)."""
    t = np.asarray(t)
    b0 = (1 - t)**3
    b1 = 3 * (1 - t)**2 * t
    b2 = 3 * (1 - t) * t**2
    b3 = t**3
    
    px = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
    py = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
    return np.column_stack([px, py])


def _path_from_bezier(params, n_points=8):
    """
    Convert 8 parameters (p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y) to a path.
    Uses n_points interior points on the Bezier curve.
    """
    p0 = np.array([params[0], params[1]])
    p1 = np.array([params[2], params[3]])
    p2 = np.array([params[4], params[5]])
    p3 = np.array([params[6], params[7]])
    
    # Use interior points (exclude t=0 and t=1 since evaluator adds those)
    t_vals = np.linspace(1/(n_points+1), n_points/(n_points+1), n_points)
    positions = _cubic_bezier(t_vals, p0, p1, p2, p3)
    angles = 90.0 * t_vals
    
    return np.column_stack([positions[:, 0], positions[:, 1], angles])


def _initial_paths():
    """Generate several initial path parameterizations."""
    inits = []
    
    # Arc-like paths with varying curvature
    for y_offset in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        inits.append(np.array([0.0, 0.0, -0.5, y_offset, -1.5, y_offset, -2.0, 0.0]))
    
    # S-shaped paths
    inits.append(np.array([0.0, 0.0, -0.3, 0.5, -1.7, -0.5, -2.0, 0.0]))
    inits.append(np.array([0.0, 0.0, -0.7, -0.5, -1.3, 0.5, -2.0, 0.0]))
    
    # Straight-ish paths
    inits.append(np.array([0.0, 0.0, -0.667, 0.0, -1.333, 0.0, -2.0, 0.0]))
    
    # More curved paths
    inits.append(np.array([0.0, 0.0, -0.2, 1.2, -1.8, 1.2, -2.0, 0.0]))
    inits.append(np.array([0.0, 0.0, -0.8, 1.2, -1.2, 1.2, -2.0, 0.0]))
    
    return inits


def _nelder_mead(objective, x0, max_iter=200, tol=1e-8, timeout=None):
    """
    Nelder-Mead simplex optimization.
    objective: function to minimize
    x0: initial point
    Returns: (best_x, best_f)
    """
    n = len(x0)
    
    # Initialize simplex
    simplex = [x0.copy()]
    f_vals = [objective(x0)]
    
    for i in range(n):
        xi = x0.copy()
        xi[i] += 0.1 if xi[i] == 0 else 0.1 * xi[i]
        simplex.append(xi)
        f_vals.append(objective(xi))
    
    # Order simplex by objective value
    order = np.argsort(f_vals)
    simplex = [simplex[i].copy() for i in order]
    f_vals = [f_vals[i] for i in order]
    
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    
    best_x = simplex[0].copy()
    best_f = f_vals[0]
    
    for iteration in range(max_iter):
        if timeout is not None and _elapsed() > timeout:
            break
        
        # Check convergence
        if abs(f_vals[0] - f_vals[-1]) < tol:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = objective(x_r)
        
        if f_r < f_vals[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
        elif f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            f_c = objective(x_c)
            if f_c < f_vals[-1]:
                simplex[-1] = x_c
                f_vals[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_vals[i] = objective(simplex[i])
        
        # Reorder
        order = np.argsort(f_vals)
        simplex = [simplex[i].copy() for i in order]
        f_vals = [f_vals[i] for i in order]
        
        # Track best
        if f_vals[0] < best_f:
            best_f = f_vals[0]
            best_x = simplex[0].copy()
    
    return best_x, best_f


def _optimize_angles_xy(path, X, Y, max_iter=100, timeout=None):
    """
    Optimize the x, y coordinates of each path point while keeping angles fixed.
    Uses coordinate descent.
    """
    n = len(path)
    current_path = path.copy()
    current_score = _score(current_path, X, Y)
    
    for iteration in range(max_iter):
        if timeout is not None and _elapsed() > timeout:
            break
        
        improved = False
        for i in range(n):
            for coord in range(2):  # x and y
                for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1]:
                    test_path = current_path.copy()
                    test_path[i, coord] += delta
                    test_score = _score(test_path, X, Y)
                    if test_score > current_score:
                        current_path = test_path
                        current_score = test_score
                        improved = True
                        break
        
        if not improved:
            break
    
    return current_path, current_score


def moving_sofa_2d() -> np.ndarray:
    """
    The 2D moving-sofa problem: find the largest-area rigid shape that can be moved
    through an L-shaped corridor of width 1 that makes a single 90-degree turn.
    
    Returns:
        path: np.ndarray of shape (n, 3); each row is (x, y, angle_degrees) for the
            corridor's inner corner.
    """
    X, Y = _build_sofa_grid()
    
    # Phase 1: Evaluate initial paths and find the best
    inits = _initial_paths()
    best_params = None
    best_score = -np.inf
    
    for params in inits:
        if _elapsed() > _DEADLINE - 60:
            break
        path = _path_from_bezier(params)
        score = _score(path, X, Y)
        if score > best_score:
            best_score = score
            best_params = params.copy()
    
    if best_params is None:
        # Fallback: simple path
        return np.array([[0.0, 0.0, 0.0]])
    
    best_path = _path_from_bezier(best_params)
    
    # Phase 2: Nelder-Mead optimization on Bezier parameters
    timeout_nm = _DEADLINE - 30
    
    def objective(params):
        path = _path_from_bezier(params)
        return -_score(path, X, Y)
    
    best_params, best_obj = _nelder_mead(objective, best_params, max_iter=150, tol=1e-7, timeout=timeout_nm)
    best_path = _path_from_bezier(best_params)
    best_score = -best_obj
    
    # Phase 3: Coordinate descent on path points
    if _elapsed() < _DEADLINE - 20:
        best_path, best_score = _optimize_angles_xy(best_path, X, Y, max_iter=50, timeout=_DEADLINE - 15)
    
    # Phase 4: Final polish with more Nelder-Mead iterations
    if _elapsed() < _DEADLINE - 10:
        def objective2(params):
            path = _path_from_bezier(params)
            return -_score(path, X, Y)
        
        # Use current best path to infer Bezier-like parameters
        # For simplicity, just do a few more NM iterations from best_params
        best_params, best_obj = _nelder_mead(objective2, best_params, max_iter=50, tol=1e-9, timeout=_DEADLINE - 5)
        best_path = _path_from_bezier(best_params)
        best_score = -best_obj
    
    # Ensure valid output
    if len(best_path) < 1:
        best_path = np.array([[0.0, 0.0, 0.0]])
    
    # Check all finite
    if not np.all(np.isfinite(best_path)):
        best_path = np.nan_to_num(best_path, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return best_path


# EVOLVE-BLOCK-END
