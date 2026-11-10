# EVOLVE-BLOCK-START
import time
import numpy as np
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

def moving_sofa_2d() -> np.ndarray:
    """
    Find a path of waypoints to maximize the sofa area.
    Uses Simulated Annealing on a dense set of free waypoints.
    Returns: np.ndarray of shape (n, 3), n >= 1, all finite.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    # Initial seed path
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
    
    best_wp = seed_wp.copy()
    best_area = _compute_area_fast(best_wp)
    
    # Simulated Annealing setup
    # We use 15 waypoints to allow for a smoother, more flexible path
    n_wp = 15
    angles = np.linspace(10, 80, n_wp)
    
    # Initialize current path
    # Interpolate seed path to get initial x, y
    seed_angles = seed_wp[:, 2]
    curr_x = np.interp(angles, seed_angles, seed_wp[:, 0])
    curr_y = np.interp(angles, seed_angles, seed_wp[:, 1])
    curr_wp = np.column_stack([curr_x, curr_y, angles])
    curr_area = _compute_area_fast(curr_wp)
    
    if curr_area > best_area:
        best_area = curr_area
        best_wp = curr_wp.copy()
        
    T = 0.05  # Initial temperature (area units)
    T_min = 1e-6
    cooling = 0.999
    alpha = 0.05 # Step size
    
    while time.time() < deadline and T > T_min:
        # Generate perturbation
        # Perturb x and y with Gaussian noise
        # Scale noise by temperature to allow larger jumps when hot
        scale = alpha * (T / 0.05) # Normalize scale relative to initial T
        
        new_x = curr_x + np.random.normal(0, scale, n_wp)
        new_y = curr_y + np.random.normal(0, scale, n_wp)
        
        # Keep angles fixed for stability, or perturb them slightly?
        # Fixed angles usually work better for this specific turn.
        new_wp = np.column_stack([new_x, new_y, angles])
        
        new_area = _compute_area_fast(new_wp)
        
        delta = new_area - curr_area
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            curr_x = new_x
            curr_y = new_y
            curr_wp = new_wp
            curr_area = new_area
            
            if curr_area > best_area:
                best_area = curr_area
                best_wp = curr_wp.copy()
                
        T *= cooling
        
        # Adaptive step size: if we've been accepting many moves, maybe increase?
        # If we've been rejecting many, decrease?
        # Simple heuristic: keep alpha constant or decay slowly.
        # Let's decay alpha slightly as T decays to ensure convergence.
        alpha *= 0.9999
        
    return best_wp
# EVOLVE-BLOCK-START
# (code already defined above)
# EVOLVE-BLOCK-END
