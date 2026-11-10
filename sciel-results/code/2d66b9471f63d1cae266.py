import time
import numpy as np

_SOFA_U = np.arange(-3.0, 1.5, 0.01)
_SOFA_V = np.arange(-1.5, 1.5, 0.01)
_SOFA_X, _SOFA_Y = np.meshgrid(_SOFA_U, _SOFA_V, indexing='ij')
_SOFA_PTS = np.stack([_SOFA_X.ravel(), _SOFA_Y.ravel()], axis=1)
_N_SOFA = len(_SOFA_PTS)

def _generate_placements(waypoints):
    P0 = np.array([0.0, 0.0, 0.0])
    PN = np.array([-2.0, 0.0, 90.0])
    if len(waypoints.shape) == 1:
        waypoints = waypoints.reshape(1, -1)
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
    if len(waypoints.shape) == 1:
        waypoints = waypoints.reshape(1, -1)
    placements = _generate_placements(waypoints)
    total = len(placements)
    if total == 0:
        return 0.0
    x = placements[:, 0]
    y = placements[:, 1]
    a_rad = np.radians(placements[:, 2])
    sx = _SOFA_PTS[:, 0].reshape(1, -1)
    sy = _SOFA_PTS[:, 1].reshape(1, -1)
    dx = sx - x.reshape(-1, 1)
    dy = sy - y.reshape(-1, 1)
    ca = np.cos(a_rad).reshape(-1, 1)
    sa = np.sin(a_rad).reshape(-1, 1)
    u = ca * dx + sa * dy
    v = -sa * dx + ca * dy
    inside1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
    inside2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    inside = inside1 | inside2
    mask = np.all(inside, axis=0)
    return mask.sum() * 0.0001

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0
    safety_deadline = deadline - 30.0

    # Initialize with a reasonable path
    n_wp = 12
    angles = np.linspace(8, 82, n_wp)
    wp = np.zeros((n_wp, 3))
    wp[:, 2] = angles
    # Initial guess: circular arc centered at (-0.5, 0.3) with radius ~0.7
    cx, cy, r = -0.5, 0.3, 0.7
    wp[:, 0] = cx + r * np.cos(np.radians(angles) + np.pi/2)
    wp[:, 1] = cy + r * np.sin(np.radians(angles) + np.pi/2)

    best_area = _compute_area(wp)
    best_wp = wp.copy()

    # Coordinate descent with random restarts and adaptive step sizes
    rng = np.random.default_rng(42)
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    current_step_idx = 0
    no_improve_count = 0

    while time.time() < safety_deadline:
        improved = False
        # Try each coordinate of each waypoint
        for i in range(n_wp):
            for dim in range(3):
                # Try small perturbations in +/- directions
                for sign in [-1, 1]:
                    if time.time() > safety_deadline:
                        break
                    trial = best_wp.copy()
                    trial[i, dim] += sign * step_sizes[current_step_idx]
                    # Keep angle within reasonable bounds
                    if dim == 2:
                        trial[i, dim] = np.clip(trial[i, dim], 0, 90)
                    area = _compute_area(trial)
                    if area > best_area + 1e-12:
                        best_area = area
                        best_wp = trial.copy()
                        improved = True
                        no_improve_count = 0
                        break  # Break sign loop, try next dim
                if time.time() > safety_deadline:
                    break
            if time.time() > safety_deadline:
                break
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 3 and current_step_idx < len(step_sizes) - 1:
                current_step_idx += 1
                no_improve_count = 0
            elif no_improve_count >= 5:
                # Random restart with small perturbation
                if time.time() > safety_deadline - 60:
                    break
                best_wp = best_wp + rng.normal(0, 0.05, size=best_wp.shape)
                best_wp[:, 2] = np.clip(best_wp[:, 2], 0, 90)
                best_area = _compute_area(best_wp)
                no_improve_count = 0
                current_step_idx = max(0, current_step_idx - 1)

    # Final polish with very small steps
    if time.time() < safety_deadline - 10:
        for _ in range(20):
            improved = False
            for i in range(n_wp):
                for dim in range(3):
                    for sign in [-1, 1]:
                        if time.time() > safety_deadline:
                            break
                        trial = best_wp.copy()
                        trial[i, dim] += sign * 0.001
                        if dim == 2:
                            trial[i, dim] = np.clip(trial[i, dim], 0, 90)
                        area = _compute_area(trial)
                        if area > best_area + 1e-12:
                            best_area = area
                            best_wp = trial.copy()
                            improved = True
                            break
                    if time.time() > safety_deadline:
                        break
                if time.time() > safety_deadline:
                    break
            if not improved:
                break

    best_wp = np.asarray(best_wp, dtype=np.float64)
    if not np.all(np.isfinite(best_wp)):
        # Fallback
        best_wp = np.array([
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ], dtype=np.float64)

    return best_wp
