# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
# EVOLVE-BLOCK-END

# Precompute sofa points once at module level for speed
_SOFA_U = np.arange(-3.0, 1.5, 0.01)
_SOFA_V = np.arange(-1.5, 1.5, 0.01)
_SOFA_X, _SOFA_Y = np.meshgrid(_SOFA_U, _SOFA_V, indexing='ij')
_SOFA_PTS = np.stack([_SOFA_X.ravel(), _SOFA_Y.ravel()], axis=1)  # shape (N, 2)
_N_SOFA = len(_SOFA_PTS)


def _generate_placements(waypoints):
    """
    Generate the full list of placements (including start (0,0,0) and end (-2,0,90)).
    Each placement is (x, y, a_degrees).
    """
    P0 = np.array([0.0, 0.0, 0.0])
    PN = np.array([-2.0, 0.0, 90.0])
    full = [P0] + [np.array(p, dtype=np.float64) for p in waypoints] + [PN]
    placements = [full[0]]
    for i in range(len(full) - 1):
        p1 = full[i]
        p2 = full[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da - 1e-9)))
        for k in range(1, m + 1):
            t = k / m
            placements.append((1 - t) * p1 + t * p2)
    return np.array(placements, dtype=np.float64)


def _compute_area_fast(waypoints):
    """
    Compute the area of the sofa (intersection of all corridor placements).
    Optimized for speed using vectorized operations and early exit.
    """
    placements = _generate_placements(waypoints)
    total = len(placements)
    if total == 0:
        return 0.0
    
    mask = np.ones(_N_SOFA, dtype=bool)
    for p in placements:
        xx, yy, aa = p[0], p[1], np.radians(p[2])
        ca, sa = np.cos(aa), np.sin(aa)
        dx = _SOFA_PTS[:, 0] - xx
        dy = _SOFA_PTS[:, 1] - yy
        u = ca * dx + sa * dy
        v = -sa * dx + ca * dy
        inside = ((u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)) | \
                 ((u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0))
        mask &= inside
        if not mask.any():
            return 0.0  # Early exit if no points remain
    area = mask.sum() * 0.0001
    return area


def _parametric_path(params):
    """
    Generate waypoints from a parametric model.
    Model: A smooth curve defined by a cubic Bezier or similar.
    Here we use a simple 4-waypoint model where positions are constrained to a smooth path.
    params: [cx, cy, r, theta_start, theta_end, k1, k2, k3, k4]
    Actually, let's use a simpler, more robust parametrization:
    The path is a circular arc from (0,0) to (-2,0) with a variable center.
    But a circle might not be flexible enough.
    
    Let's use a polynomial path in x, but x is not monotonic?
    The start is (0,0) and end is (-2,0).
    Let's parameterize by angle a from 0 to 90.
    x(a) and y(a) are cubic polynomials in a.
    x(a) = c0 + c1*a + c2*a^2 + c3*a^3
    y(a) = d0 + d1*a + d2*a^2 + d3*a^3
    
    Constraints:
    x(0) = 0 => c0 = 0
    x(90) = -2 => c1*90 + c2*8100 + c3*729000 = -2
    y(0) = 0 => d0 = 0
    y(90) = 0 => d1*90 + d2*8100 + d3*729000 = 0
    
    We have 6 free parameters: c1, c2, c3, d1, d2, d3.
    We can define waypoints at a = 15, 30, 45, 60, 75.
    """
    c1, c2, c3, d1, d2, d3 = params
    
    # Solve for c1 given c2, c3
    # c1*90 = -2 - c2*8100 - c3*729000
    c1 = (-2.0 - c2 * 8100.0 - c3 * 729000.0) / 90.0
    
    # Solve for d1 given d2, d3
    # d1*90 = 0 - d2*8100 - d3*729000
    d1 = (-d2 * 8100.0 - d3 * 729000.0) / 90.0
    
    angles = np.array([15.0, 30.0, 45.0, 60.0, 75.0])
    xs = c1 * angles + c2 * angles**2 + c3 * angles**3
    ys = d1 * angles + d2 * angles**2 + d3 * angles**3
    
    waypoints = np.column_stack([xs, ys, angles])
    return waypoints


def moving_sofa_2d() -> np.ndarray:
    """
    Find a path of waypoints to maximize the sofa area.
    Uses CMA-ES style optimization on a parametric curve model.
    Returns: np.ndarray of shape (n, 3), n >= 1, all finite.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_deadline = deadline - 10.0

    best_area = -1.0
    best_wp = None
    
    # Initial seed path (from previous good solution)
    seed_wp = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)
    
    seed_area = _compute_area_fast(seed_wp)
    if seed_area > best_area:
        best_area = seed_area
        best_wp = seed_wp.copy()

    # Parametric optimization
    # We will optimize the 6 parameters of the cubic polynomial path.
    # Initial guess: A straight line-ish path?
    # If we want x to go from 0 to -2, and y to stay near 0.
    # Let's try a small perturbation around a simple path.
    
    # Simple path: x = -2*a/90, y=0
    # c1 = -2/90, c2=0, c3=0, d1=0, d2=0, d3=0
    x0 = np.array([-2.0/90.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Another guess: The seed path approximated by polynomials?
    # Let's just use random restarts with CMA-ES (implemented via Nelder-Mead or simple coordinate descent)
    # Since CMA-ES is not in scipy, we will use a robust multi-start Nelder-Mead or a manual CMA-ES.
    # Given time constraints, a well-tuned Nelder-Mead with multiple restarts is safer.
    
    def neg_area(params):
        wp = _parametric_path(params)
        area = _compute_area_fast(wp)
        return -area
    
    np.random.seed(42)
    
    max_restarts = 20
    for r in range(max_restarts):
        if time.time() > safety_deadline:
            break
            
        if r == 0:
            x_start = x0
        else:
            # Random restarts in the parameter space
            x_start = np.random.normal(0, 0.1, 6)
            # Ensure the path doesn't go too crazy
            # x_start[0] (c1) should be around -0.02
            x_start[0] = -0.02 + np.random.normal(0, 0.02)
            
        # Use Nelder-Mead for optimization
        try:
            res = minimize(neg_area, x_start, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            curr_area = -res.fun
            curr_wp = _parametric_path(res.x)
            
            if curr_area > best_area:
                best_area = curr_area
                best_wp = curr_wp.copy()
        except Exception:
            pass

    # If the parametric model is too restrictive, we can also try a few random waypoints as a fallback
    # But the parametric model should be strong enough if the optimal path is smooth.
    
    if best_wp is None:
        return seed_wp

    best_wp = np.asarray(best_wp, dtype=np.float64)
    if not np.all(np.isfinite(best_wp)):
        best_wp = seed_wp

    return best_wp
# EVOLVE-BLOCK-START
# (code already defined above)
# EVOLVE-BLOCK-END
