# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 620.0  # seconds (safety margin before 630s kill)


def _time_remaining() -> float:
    return _DEADLINE - (time.time() - _START_TIME)


def _check_inside(points_x, points_y, x, y, a_deg):
    """
    Check if all points (points_x, points_y) are inside the corridor placement (x, y, a_deg).
    Returns True if all points are inside, False otherwise.
    Vectorized for speed.
    """
    a_rad = np.deg2rad(a_deg)
    cos_a = np.cos(a_rad)
    sin_a = np.sin(a_rad)

    # Rotate points into corridor frame: R(-a)(w - (x, y))
    dx = points_x - x
    dy = points_y - y

    # R(-a) = [[cos(-a), -sin(-a)], [sin(-a), cos(-a)]] = [[cos_a, sin_a], [-sin_a, cos_a]]
    u = cos_a * dx + sin_a * dy
    v = -sin_a * dx + cos_a * dy

    # Base corridor: union of {0 <= u <= 1, -3 <= v <= 1} and {-3 <= u <= 1, 0 <= v <= 1}
    rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    inside = rect1 | rect2

    return np.all(inside)


def _score_path(path, sofa_x, sofa_y, verbose=False):
    """
    Compute the raw score for a given path (without endpoints).
    The evaluator prepends (0,0,0) and appends (-2,0,90), then densifies.
    We simulate that process here.
    """
    # Build full path with endpoints
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([-2.0, 0.0, 90.0])
    full_path = np.vstack([start, path, end])

    # Densify: between consecutive rows, add linear interpolations in m = max(1, ceil(|a2-a1|)) steps
    densified = [full_path[0]]
    for i in range(len(full_path) - 1):
        p1 = full_path[i]
        p2 = full_path[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        if m == 1:
            densified.append(p2)
        else:
            for k in range(1, m + 1):
                t = k / m
                interp = p1 + t * (p2 - p1)
                densified.append(interp)

    densified = np.array(densified)

    # Check each placement: all sofa points must be inside
    for (x, y, a) in densified:
        if not _check_inside(sofa_x, sofa_y, x, y, a):
            if verbose:
                print(f"  Failed at placement ({x:.4f}, {y:.4f}, {a:.2f})")
            return 0.0

    return len(sofa_x) * 0.0001


def _objective(path_flat, sofa_x, sofa_y):
    """
    Negative score for minimization. If invalid, return a large penalty.
    """
    n = len(path_flat) // 3
    path = path_flat.reshape(n, 3)
    score = _score_path(path, sofa_x, sofa_y)
    return -score


def _generate_initial_paths(n_paths, n_points, rng):
    """
    Generate diverse initial paths for optimization.
    """
    paths = []
    for i in range(n_paths):
        # Interpolate between start angle 0 and end angle 90
        angles = np.linspace(0, 90, n_points + 2)[1:-1]  # exclude endpoints
        # Add some noise to angles
        angles = angles + rng.uniform(-5, 5, size=n_points)
        # Ensure monotonic-ish increase
        for j in range(1, n_points):
            if angles[j] < angles[j - 1]:
                angles[j] = angles[j - 1] + 0.5

        # x, y positions: slide from near (0,0) to near (-2,0)
        # Use a curved path
        t = np.linspace(0, 1, n_points)
        # Curved trajectory
        x = -2.0 * t + rng.uniform(-0.2, 0.2, size=n_points)
        y = 0.5 * np.sin(np.pi * t) + rng.uniform(-0.1, 0.1, size=n_points)

        path = np.column_stack([x, y, angles])
        paths.append(path)
    return paths


def moving_sofa_2d() -> np.ndarray:
    """
    Find a path that maximizes the area of the intersection of all corridor placements.
    Returns: np.ndarray of shape (n, 3), n >= 1, all finite.
    """
    # Sofa grid points
    sofa_x = np.arange(-3, 1.5, 0.01)
    sofa_y = np.arange(-1.5, 1.5, 0.01)
    sofa_x_grid, sofa_y_grid = np.meshgrid(sofa_x, sofa_y, indexing='ij')
    sofa_x_flat = sofa_x_grid.ravel()
    sofa_y_flat = sofa_y_grid.ravel()
    n_sofa = len(sofa_x_flat)

    rng = np.random.default_rng(42)

    # Start with the baseline path
    baseline_path = np.array(
        [
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ],
        dtype=np.float64,
    )

    best_path = baseline_path.copy()
    best_score = _score_path(best_path, sofa_x_flat, sofa_y_flat)

    # Try different numbers of path points
    n_points_options = [8, 10, 12, 15]
    n_initial = 5  # number of initial random paths

    for n_points in n_points_options:
        if _time_remaining() < 30:
            break

        # Generate initial paths
        initial_paths = _generate_initial_paths(n_initial, n_points, rng)
        # Also include the baseline interpolated to n_points
        if n_points != 8:
            t = np.linspace(0, 1, n_points)
            x_base = np.interp(t, np.linspace(0, 1, len(baseline_path)), baseline_path[:, 0])
            y_base = np.interp(t, np.linspace(0, 1, len(baseline_path)), baseline_path[:, 1])
            a_base = np.interp(t, np.linspace(0, 1, len(baseline_path)), baseline_path[:, 2])
            interp_baseline = np.column_stack([x_base, y_base, a_base])
            initial_paths.append(interp_baseline)

        for init_path in initial_paths:
            if _time_remaining() < 15:
                break

            # Flatten for optimization
            x0 = init_path.ravel()

            # Use Nelder-Mead for derivative-free optimization
            try:
                result = minimize(
                    _objective,
                    x0,
                    args=(sofa_x_flat, sofa_y_flat),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 0.01,
                        'fatol': 1e-6,
                        'adaptive': True,
                    },
                )
                if result.fun < -best_score:
                    new_path = result.x.reshape(n_points, 3)
                    # Verify
                    new_score = _score_path(new_path, sofa_x_flat, sofa_y_flat)
                    if new_score > best_score:
                        best_score = new_score
                        best_path = new_path.copy()
            except Exception as e:
                pass

    # Final check
    final_score = _score_path(best_path, sofa_x_flat, sofa_y_flat)
    if final_score <= 0:
        # Fallback to baseline
        return baseline_path

    return best_path


# EVOLVE-BLOCK-END
