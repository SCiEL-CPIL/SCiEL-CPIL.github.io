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
    Uses vectorized operations over placements for each sofa point subset.
    To keep memory manageable, we process in chunks of placements if necessary, 
    but given N_sofa ~ 135k and N_placements usually < 100, 
    (N_placements, N_sofa) boolean array is ~ 13.5M bools = 13.5MB, which is fine.
    """
    placements = _generate_placements(waypoints)
    total_pl = len(placements)
    
    # Precompute rotation matrices for all placements
    # R(-a) = [[cos(a), sin(a)], [-sin(a), cos(a)]]
    a_rad = np.radians(placements[:, 2])
    ca = np.cos(a_rad)
    sa = np.sin(a_rad)
    
    # We want to check if R(-a) * (w - (x,y)) is in base.
    # Let w = (wx, wy).
    # u = ca*(wx-x) + sa*(wy-y)
    # v = -sa*(wx-x) + ca*(wy-y)
    
    # Vectorized check:
    # For each placement i, we have x_i, y_i, ca_i, sa_i.
    # For each sofa point j, we have wx_j, wy_j.
    # u[i, j] = ca_i*(wx_j - x_i) + sa_i*(wy_j - y_i)
    # v[i, j] = -sa_i*(wx_j - x_i) + ca_i*(wy_j - y_i)
    
    # This creates arrays of shape (total_pl, N_SOFA).
    # 135000 * 100 * 8 bytes * 2 (for u and v) ~ 200MB. Might be tight but okay.
    # Let's do it in chunks of placements to be safe, or just rely on NumPy efficiency.
    # Actually, 135k * 100 is 13.5 million floats. 13.5M * 8 bytes = 108 MB. 
    # We have u and v, so ~216 MB. Plus intermediate arrays.
    # This might be too much memory for some environments.
    
    # Safer approach: Iterate over placements, keep a boolean mask of sofa points.
    # This is O(N_pl * N_sofa) operations but memory efficient.
    # N_pl is small (e.g., 10-50). N_sofa is 135k.
    # 50 * 135k = 6.75 million operations. This is fast in NumPy.
    
    mask = np.ones(_N_SOFA, dtype=bool)
    
    for i in range(total_pl):
        x, y, a_deg = placements[i]
        a = np.radians(a_deg)
        ca = np.cos(a)
        sa = np.sin(a)
        
        dx = _SOFA_PTS[:, 0] - x
        dy = _SOFA_PTS[:, 1] - y
        
        u = ca * dx + sa * dy
        v = -sa * dx + ca * dy
        
        inside = ((u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)) | \
                 ((u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0))
        
        mask &= inside
        
        # Early exit if no points remain
        if not mask.any():
            return 0.0
            
    return mask.sum() * 0.0001


def moving_sofa_2d() -> np.ndarray:
    """
    Find a path of waypoints to maximize the sofa area.
    Uses Simulated Annealing with variable path length.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    # Initial state: A simple path
    # The sofa moves from (0,0,0) to (-2,0,90).
    # A good initial guess is a curve that stays near the origin but allows rotation.
    # Let's start with a few waypoints.
    
    def random_path(n_wp):
        # Angles should be sorted 0..90
        angles = np.sort(np.random.uniform(0, 90, n_wp))
        # x, y should be in a reasonable box
        # The sofa is 1x4 L-shape. 
        # x range roughly -3 to 1, y range -1.5 to 1.5 relative to corner.
        # The corner moves from (0,0) to (-2,0).
        # Let's constrain x to [-3, 1] and y to [-2, 2]
        xs = np.random.uniform(-3.0, 1.0, n_wp)
        ys = np.random.uniform(-2.0, 2.0, n_wp)
        return np.column_stack([xs, ys, angles])

    # Start with a decent heuristic path
    # The "standard" solution often involves keeping the sofa near the corner of the wall.
    # Let's try a path that rotates slowly while moving slightly.
    n_init = 5
    angles_init = np.linspace(0, 90, n_init)[1:-1] # Exclude 0 and 90 as they are fixed endpoints
    # Actually, let's just use random init to be safe, but seed it well.
    np.random.seed(42)
    
    best_wp = random_path(5)
    best_area = _compute_area_fast(best_wp)
    
    current_wp = best_wp.copy()
    current_area = best_area
    
    T = 0.01  # Initial temperature (area units)
    T_min = 0.0001
    alpha = 0.999 # Cooling rate
    
    iteration = 0
    while time.time() < deadline:
        # Perturbation
        # 1. Move a random waypoint
        # 2. Add a waypoint
        # 3. Remove a waypoint
        
        n_wp = len(current_wp)
        action = np.random.choice(['move', 'add', 'remove'], p=[0.7, 0.2, 0.1])
        
        new_wp = current_wp.copy()
        
        if action == 'move':
            idx = np.random.randint(0, n_wp)
            # Gaussian perturbation
            step_size = 0.1
            new_wp[idx, 0] += np.random.normal(0, step_size)
            new_wp[idx, 1] += np.random.normal(0, step_size)
            new_wp[idx, 2] += np.random.normal(0, step_size * 2) # Angles change slower/faster?
            # Clamp angles to [0, 90] and sort? 
            # Sorting angles is important for the path definition to make sense (monotonic rotation)
            # If we don't sort, the interpolation might twist.
            # The problem statement implies a sequence. If angles go 10, 80, 20, the sofa rotates back and forth.
            # Is that allowed? "Between consecutive rows it adds linear interpolations... in m steps".
            # Yes, it's allowed, but likely suboptimal. 
            # However, for SA, we should allow exploration. 
            # But maintaining sorted angles is a strong constraint that helps.
            # Let's clamp angles to [0, 90] but NOT sort, to allow local non-monotonicity if it helps?
            # Actually, non-monotonic rotation is very likely to reduce area.
            # Let's clamp to [0, 90].
            new_wp[:, 2] = np.clip(new_wp[:, 2], 0, 90)
            
        elif action == 'add' and n_wp < 20:
            # Insert after a random waypoint
            idx = np.random.randint(0, n_wp)
            # New point is average of neighbors + noise
            if idx == 0:
                base = current_wp[0]
            elif idx == n_wp - 1:
                base = current_wp[-1]
            else:
                base = (current_wp[idx] + current_wp[idx+1]) / 2.0
                # Wait, if we insert at idx, we shift.
                # Let's insert at position idx+1 (after idx)
                
            new_point = base + np.random.normal(0, 0.05, 3)
            new_point[2] = np.clip(new_point[2], 0, 90)
            
            # Insert into array
            # We want to insert at index idx+1
            insert_idx = idx + 1
            new_wp = np.vstack([current_wp[:insert_idx], new_point.reshape(1,3), current_wp[insert_idx:]])
            
        elif action == 'remove' and n_wp > 2:
            idx = np.random.randint(0, n_wp)
            new_wp = np.delete(current_wp, idx, axis=0)
            
        # Evaluate new state
        new_area = _compute_area_fast(new_wp)
        
        # Acceptance criterion
        delta = new_area - current_area
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_wp = new_wp
            current_area = new_area
            
            if current_area > best_area:
                best_area = current_area
                best_wp = current_wp.copy()
        
        # Cool down
        T *= alpha
        
        # Restart if stuck?
        # Simple restart logic: if no improvement for X iterations, reset T and randomize slightly
        # For now, just keep cooling. If T gets too low, we might get stuck.
        # Let's add a check: if T < T_min, restart with a perturbed best_wp
        if T < T_min:
            T = 0.005
            # Perturb best_wp
            current_wp = best_wp + np.random.normal(0, 0.1, best_wp.shape)
            current_wp[:, 2] = np.clip(current_wp[:, 2], 0, 90)
            current_area = _compute_area_fast(current_wp)
            
        iteration += 1
        
    # Ensure output is valid
    best_wp = np.asarray(best_wp, dtype=np.float64)
    if not np.all(np.isfinite(best_wp)):
        # Fallback
        return np.array([[-1.0, 0.0, 45.0]])
        
    return best_wp

# EVOLVE-BLOCK-START
# (No additional code needed here)
# EVOLVE-BLOCK-END
