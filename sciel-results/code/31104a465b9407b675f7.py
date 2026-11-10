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


def moving_sofa_2d() -> np.ndarray:
    """
    Find a path of waypoints to maximize the sofa area.
    Uses a directed random walk with adaptive step sizes and restarts.
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

    # Search parameters
    n_wp = 8
    max_restarts = 50
    max_steps_per_restart = 2000
    
    # Define bounds for waypoints
    x_bounds = np.array([-1.5, 0.5])
    y_bounds = np.array([-0.5, 0.5])
    a_bounds = np.array([0.0, 90.0])
    
    np.random.seed(42)
    
    for r in range(max_restarts):
        if time.time() > safety_deadline:
            break
            
        # Initialize current path
        if r == 0:
            curr_wp = best_wp.copy()
            curr_area = best_area
        else:
            # Random restart: generate a random valid path
            angles = np.sort(np.random.uniform(a_bounds[0], a_bounds[1], n_wp))
            # Ensure angles are strictly increasing to avoid degeneracy
            angles[0] = np.random.uniform(0, 10)
            angles[-1] = np.random.uniform(80, 90)
            for i in range(1, n_wp-1):
                angles[i] = np.random.uniform(angles[i-1] + 1, angles[i+1] - 1)
                angles[i] = np.clip(angles[i], a_bounds[0], a_bounds[1])
            
            x = np.random.uniform(x_bounds[0], x_bounds[1], n_wp)
            y = np.random.uniform(y_bounds[0], y_bounds[1], n_wp)
            curr_wp = np.column_stack([x, y, angles])
            curr_area = _compute_area(curr_wp)
            
            # If this random restart is better, update best
            if curr_area > best_area:
                best_area = curr_area
                best_wp = curr_wp.copy()

        # Local search: random walk with adaptive step size
        step_size = 0.1
        no_improve_count = 0
        max_no_improve = 100
        
        for step in range(max_steps_per_restart):
            if time.time() > safety_deadline:
                break
                
            # Perturb a random waypoint
            i = np.random.randint(n_wp)
            perturb_wp = curr_wp.copy()
            
            # Perturb x, y, a
            perturb_wp[i, 0] += np.random.normal(0, step_size)
            perturb_wp[i, 1] += np.random.normal(0, step_size)
            perturb_wp[i, 2] += np.random.normal(0, step_size)
            
            # Ensure angles are sorted (if we perturb angle, we might break order)
            # If we break order, we can either reject or sort. Sorting changes the path structure.
            # Let's just reject if order is broken to keep it simple, or re-sort.
            # Re-sorting is safer for the "path" interpretation.
            if not np.all(np.diff(perturb_wp[:, 2]) > 0):
                # Try to fix by sorting, but this might jump too far.
                # Let's just reject this move if it breaks monotonicity.
                continue
            
            # Evaluate new area
            new_area = _compute_area(perturb_wp)
            
            if new_area > curr_area:
                curr_wp = perturb_wp
                curr_area = new_area
                no_improve_count = 0
                step_size = min(step_size * 1.1, 0.5) # Increase step size slightly
                
                if curr_area > best_area:
                    best_area = curr_area
                    best_wp = curr_wp.copy()
            else:
                no_improve_count += 1
                step_size = max(step_size * 0.95, 0.01) # Decrease step size
                
                # If stuck for too long, restart local search
                if no_improve_count > max_no_improve:
                    break

    # Return the best waypoint path found
    if best_wp is None:
        return seed_wp

    best_wp = np.asarray(best_wp, dtype=np.float64)
    if not np.all(np.isfinite(best_wp)):
        best_wp = seed_wp

    return best_wp
# EVOLVE-BLOCK-START
# (code already defined above)
# EVOLVE-BLOCK-END
