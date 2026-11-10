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


def _compute_area(waypoints):
    """
    Compute the area of the sofa (intersection of all corridor placements).
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
            break  # Early exit if no points remain
    area = mask.sum() * 0.0001
    return area


def _params_to_waypoints(params):
    """
    Map 5 parameters to a smooth path of 8 waypoints.
    params: [x_c, y_c, x_amp, y_amp, angle_offset]
    """
    n_wp = 8
    # Base angles: linear from 0 to 90, but we want to shape the turn
    # t in [0, 1]
    t = np.linspace(0, 1, n_wp + 2)[1:-1] # 8 points
    
    # Shape the angle progression. 
    # Base is linear. We add a sinusoidal perturbation to allow "early" or "late" turns.
    # angle(t) = 90 * t + offset * sin(pi * t)
    # offset in [-20, 20] allows shifting the turn center.
    angles = 90.0 * t + params[4] * np.sin(np.pi * t)
    
    # Ensure angles are strictly increasing and within [0, 90]
    # If offset is too large, we might violate bounds. 
    # Let's clip and enforce monotonicity.
    angles = np.clip(angles, 0.0, 90.0)
    # Enforce strict increase by adding a tiny epsilon if needed
    for i in range(1, len(angles)):
        if angles[i] <= angles[i-1]:
            angles[i] = angles[i-1] + 1e-6
    
    # Path x(t), y(t)
    # We want a path from (0,0) to (-2,0) roughly.
    # But the waypoints are intermediate.
    # Let's define the center of the turn.
    # x(t) = -2 * t + x_amp * sin(pi * t)  -> 0 at t=0, -2 at t=1. Sinusoidal bulge.
    # y(t) = y_c + y_amp * sin(pi * t)     -> y_c at ends, y_c+y_amp in middle.
    
    x = -2.0 * t + params[2] * np.sin(np.pi * t)
    y = params[1] + params[3] * np.sin(np.pi * t)
    
    # Note: The start is (0,0) and end is (-2,0). 
    # The waypoints should connect smoothly.
    # At t=0, x=0, y=params[1]. But start is (0,0). 
    # So we need y(0) to be close to 0? 
    # The evaluator interpolates between (0,0,0) and the first waypoint.
    # So the first waypoint doesn't have to be at y=0.
    # However, a smooth path usually implies the waypoints follow a curve.
    
    # Let's refine the parameterization to ensure the path is physically plausible.
    # x(t) should go from 0 to -2.
    # y(t) should start near 0 and end near 0.
    # Using sin(pi*t) for y ensures y(0)=y(1)=params[1]. 
    # If we want y(0)=0 and y(1)=0, we should set params[1]=0? 
    # But the turn might need to be offset. 
    # Actually, the corridor is L-shaped. The "inner corner" moves.
    # A good path for the inner corner (x,y) of the L-shape (width 1, arms 4) 
    # turning from horizontal (pointing left? no, base is union of vertical and horizontal rects)
    # Base: {0<=u<=1, -3<=v<=1} U {-3<=u<=1, 0<=v<=1}.
    # At a=0, the "corner" is at (0,0). The corridor extends left (-u) and down (-v) and right/up.
    # Wait, u,v are local coords. 
    # R(-0)(w - (0,0)) in base.
    # So at (0,0,0), the corridor covers x in [-3, 1], y in [-3, 1] roughly (L-shape).
    # At a=90, rotated 90 deg.
    
    # The path of the corner (x,y) should stay within the "free space" of the sofa.
    # The sofa is the intersection.
    
    # Let's stick to the 5-param model:
    # x_wp = -2 * t + x_amp * sin(pi * t)
    # y_wp = y_offset + y_amp * sin(pi * t)
    # a_wp = 90 * t + a_offset * sin(pi * t)
    
    # We return 8 waypoints.
    wps = np.column_stack([x, y, angles])
    return wps


def _neg_area(params):
    """
    Objective function for minimization.
    """
    try:
        wps = _params_to_waypoints(params)
        # Check for invalid waypoints
        if not np.all(np.isfinite(wps)):
            return 1.0
        area = _compute_area(wps)
        return -area
    except Exception:
        return 1.0


def moving_sofa_2d() -> np.ndarray:
    """
    Find a path of waypoints to maximize the sofa area.
    Uses Nelder-Mead optimization on a 5-parameter analytic path family.
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
    
    seed_area = _compute_area(seed_wp)
    if seed_area > best_area:
        best_area = seed_area
        best_wp = seed_wp.copy()

    # Optimization parameters
    # x_c (unused in current param, kept for potential future), y_c, x_amp, y_amp, a_offset
    # Actually, my _params_to_waypoints uses:
    # params[0]: unused? No, I used params[1] for y_offset.
    # Let's align indices:
    # params = [y_offset, x_amp, y_amp, a_offset] -> 4 params?
    # In _params_to_waypoints:
    # x = -2 * t + params[2] * sin(...)  -> params[2] is x_amp
    # y = params[1] + params[3] * sin(...) -> params[1] is y_offset, params[3] is y_amp
    # angles = ... + params[4] * sin(...)  -> params[4] is a_offset
    # So I have 5 params: [0, 1, 2, 3, 4]. params[0] is currently unused.
    # Let's use params[0] as x_offset? 
    # x = params[0] + (-2) * t + params[2] * sin(...)
    # This allows shifting the whole path x-wise.
    
    def _params_to_waypoints_v2(params):
        n_wp = 8
        t = np.linspace(0, 1, n_wp + 2)[1:-1]
        
        # x(t) = x_start + (x_end - x_start)*t + x_amp * sin(pi * t)
        # x_start = 0, x_end = -2
        x = 0.0 + (-2.0) * t + params[2] * np.sin(np.pi * t)
        # Add x_offset
        x = x + params[0]
        
        # y(t) = y_offset + y_amp * sin(pi * t)
        y = params[1] + params[3] * np.sin(np.pi * t)
        
        # a(t) = 90 * t + a_offset * sin(pi * t)
        angles = 90.0 * t + params[4] * np.sin(np.pi * t)
        
        angles = np.clip(angles, 0.0, 90.0)
        for i in range(1, len(angles)):
            if angles[i] <= angles[i-1]:
                angles[i] = angles[i-1] + 1e-6
        
        wps = np.column_stack([x, y, angles])
        return wps

    def _neg_area_v2(params):
        try:
            wps = _params_to_waypoints_v2(params)
            if not np.all(np.isfinite(wps)):
                return 1.0
            area = _compute_area(wps)
            return -area
        except Exception:
            return 1.0

    # Bounds for params
    # x_offset: [-0.5, 0.5]
    # y_offset: [-0.5, 0.5]
    # x_amp: [-0.5, 0.5]
    # y_amp: [-0.5, 0.5]
    # a_offset: [-20, 20]
    bounds = [
        (-0.5, 0.5),
        (-0.5, 0.5),
        (-0.5, 0.5),
        (-0.5, 0.5),
        (-20.0, 20.0)
    ]
    
    np.random.seed(42)
    
    # Run multiple Nelder-Mead optimizations from different starting points
    n_restarts = 20
    
    for r in range(n_restarts):
        if time.time() > safety_deadline:
            break
            
        # Generate initial guess
        if r == 0:
            # Start near the known good solution's implied parameters
            # The seed wp has x from -0.1 to -0.7, y around 0.2-0.3, angles 15-86.
            # Let's just use a reasonable center.
            x0 = np.array([0.0, 0.2, 0.0, 0.0, 0.0])
        else:
            x0 = np.array([
                np.random.uniform(-0.3, 0.3),
                np.random.uniform(-0.3, 0.3),
                np.random.uniform(-0.3, 0.3),
                np.random.uniform(-0.3, 0.3),
                np.random.uniform(-10.0, 10.0)
            ])
        
        # Nelder-Mead
        options = {
            'maxiter': 1000,
            'xatol': 1e-4,
            'fatol': 1e-6,
            'adaptive': True
        }
        
        try:
            res = minimize(_neg_area_v2, x0, method='Nelder-Mead', bounds=bounds, options=options)
            
            if res.success or res.fun < 0.5: # 0.5 means area > 0.5
                area = -res.fun
                if area > best_area:
                    best_area = area
                    best_wp = _params_to_waypoints_v2(res.x).copy()
        except Exception:
            pass

    # Final check
    if best_wp is None:
        return seed_wp

    best_wp = np.asarray(best_wp, dtype=np.float64)
    if not np.all(np.isfinite(best_wp)):
        best_wp = seed_wp

    return best_wp

# EVOLVE-BLOCK-START
# (code already defined above)
# EVOLVE-BLOCK-END
