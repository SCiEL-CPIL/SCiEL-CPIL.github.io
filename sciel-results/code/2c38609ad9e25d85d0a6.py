import time
import numpy as np

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
    """
    placements = _generate_placements(waypoints)
    n_plac = len(placements)
    if n_plac == 0:
        return 0.0
    
    # Process in chunks to keep memory manageable
    # Each chunk: for a set of placements, compute which sofa points are inside all of them
    mask = np.ones(_N_SOFA, dtype=bool)
    
    chunk_size = 200  # placements per chunk
    
    for start in range(0, n_plac, chunk_size):
        end = min(start + chunk_size, n_plac)
        chunk = placements[start:end]
        
        # For each placement in chunk, compute inside mask
        # Shape: (chunk_len, N_SOFA)
        xx = chunk[:, 0]  # (chunk_len,)
        yy = chunk[:, 1]
        aa = np.radians(chunk[:, 2])  # (chunk_len,)
        
        ca = np.cos(aa)  # (chunk_len,)
        sa = np.sin(aa)
        
        # dx, dy: (N_SOFA, chunk_len)
        dx = _SOFA_PTS[:, 0:1] - xx[None, :]  # (N_SOFA, chunk_len)
        dy = _SOFA_PTS[:, 1:2] - yy[None, :]
        
        # u = ca * dx + sa * dy
        # v = -sa * dx + ca * dy
        u = ca[None, :] * dx + sa[None, :] * dy  # (N_SOFA, chunk_len)
        v = -sa[None, :] * dx + ca[None, :] * dy
        
        # inside: ((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)) | ((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1))
        inside1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        inside2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        inside = inside1 | inside2  # (N_SOFA, chunk_len)
        
        # All placements in chunk must contain the point
        all_inside = inside.all(axis=1)  # (N_SOFA,)
        mask &= all_inside
        
        # Early termination if no points remain
        if not mask.any():
            return 0.0
    
    return mask.sum() * 0.0001


def _initial_path(n_wp=90):
    """Generate an initial path with n_wp waypoints, angles from 0 to 90."""
    angles = np.linspace(5, 85, n_wp)
    # Start with a reasonable curve: x goes from 0 to -2, y stays near 0
    t = angles / 90.0
    x = -1.0 * t  # linear from 0 to -1 (will be adjusted)
    y = 0.0 * np.ones(n_wp)
    wp = np.column_stack([x, y, angles])
    return wp


def moving_sofa_2d() -> np.ndarray:
    """
    Find a path of waypoints to maximize the sofa area.
    Uses coordinate descent on a dense set of waypoints.
    Returns: np.ndarray of shape (n, 3), n >= 1, all finite.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin
    
    # Start with initial path
    n_wp = 90
    wp = _initial_path(n_wp)
    
    # Evaluate initial
    best_area = _compute_area_fast(wp)
    best_wp = wp.copy()
    
    # Coordinate descent: iterate over waypoints, try small perturbations in x and y
    # Angles are fixed (evenly spaced)
    
    # Perturbation sizes (adaptive)
    base_dx = 0.05
    base_dy = 0.05
    
    # Multiple passes
    pass_num = 0
    while time.time() < deadline:
        pass_num += 1
        
        # Decrease step size as we make progress
        scale = max(0.1, 1.0 / (1 + 0.1 * pass_num))
        dx_step = base_dx * scale
        dy_step = base_dy * scale
        
        improved = False
        
        for i in range(n_wp):
            if time.time() > deadline:
                break
            
            # Try perturbing x
            for dx in [-dx_step, dx_step]:
                wp_new = wp.copy()
                wp_new[i, 0] += dx
                area = _compute_area_fast(wp_new)
                if area > best_area + 1e-10:
                    wp = wp_new
                    best_area = area
                    best_wp = wp.copy()
                    improved = True
                    break
            
            # Try perturbing y
            for dy in [-dy_step, dy_step]:
                wp_new = wp.copy()
                wp_new[i, 1] += dy
                area = _compute_area_fast(wp_new)
                if area > best_area + 1e-10:
                    wp = wp_new
                    best_area = area
                    best_wp = wp.copy()
                    improved = True
                    break
            
            # Try larger steps occasionally
            if not improved and pass_num % 5 == 0:
                for dx in [-2*dx_step, 2*dx_step]:
                    wp_new = wp.copy()
                    wp_new[i, 0] += dx
                    area = _compute_area_fast(wp_new)
                    if area > best_area + 1e-10:
                        wp = wp_new
                        best_area = area
                        best_wp = wp.copy()
                        improved = True
                        break
                if not improved:
                    for dy in [-2*dy_step, 2*dy_step]:
                        wp_new = wp.copy()
                        wp_new[i, 1] += dy
                        area = _compute_area_fast(wp_new)
                        if area > best_area + 1e-10:
                            wp = wp_new
                            best_area = area
                            best_wp = wp.copy()
                            improved = True
                            break
        
        # If no improvement in a full pass, reduce step size further or break
        if not improved and pass_num > 5:
            # Try a few random restarts of the step size
            np.random.seed(pass_num)
            restart_dx = np.random.uniform(0.01, 0.1)
            restart_dy = np.random.uniform(0.01, 0.1)
            for i in range(n_wp):
                if time.time() > deadline:
                    break
                for dx in [-restart_dx, restart_dx]:
                    wp_new = wp.copy()
                    wp_new[i, 0] += dx
                    area = _compute_area_fast(wp_new)
                    if area > best_area + 1e-10:
                        wp = wp_new
                        best_area = area
                        best_wp = wp.copy()
                        break
                for dy in [-restart_dy, restart_dy]:
                    wp_new = wp.copy()
                    wp_new[i, 1] += dy
                    area = _compute_area_fast(wp_new)
                    if area > best_area + 1e-10:
                        wp = wp_new
                        best_area = area
                        best_wp = wp.copy()
                        break
            # If still no improvement, break
            if not improved:
                break
    
    # Ensure output is valid: n >= 1, all finite
    best_wp = np.asarray(best_wp, dtype=np.float64)
    if not np.all(np.isfinite(best_wp)):
        # Fallback
        best_wp = _initial_path(n_wp)
    
    return best_wp
