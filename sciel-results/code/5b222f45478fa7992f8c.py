# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Precompute sofa points at module level for fast vectorized evaluation
_SOFA_U = np.arange(-3.0, 1.5, 0.01)
_SOFA_V = np.arange(-1.5, 1.5, 0.01)
_SOFA_X, _SOFA_Y = np.meshgrid(_SOFA_U, _SOFA_V, indexing='ij')
_SOFA_PTS = np.stack([_SOFA_X.ravel(), _SOFA_Y.ravel()], axis=1)  # shape (N, 2)
_N_SOFA = _SOFA_PTS.shape[0]
_SOFA_X_1D = _SOFA_PTS[:, 0]
_SOFA_Y_1D = _SOFA_PTS[:, 1]


def _generate_placements(waypoints):
    """
    Generate the full list of placements including start (0,0,0) and end (-2,0,90).
    Each placement is (x, y, a_degrees). Linearly interpolate between consecutive
    waypoints with m = max(1, ceil(|a2 - a1|)) steps.
    """
    P0 = np.array([0.0, 0.0, 0.0])
    PN = np.array([-2.0, 0.0, 90.0])
    full = [P0] + [np.array(p, dtype=np.float64) for p in waypoints] + [PN]

    placements = [full[0].copy()]
    for i in range(len(full) - 1):
        p1 = full[i]
        p2 = full[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da - 1e-9)))
        for k in range(1, m + 1):
            t = k / m
            placements.append(p1 + t * (p2 - p1))
    return placements


def _point_in_base(u, v):
    """Check if point (u, v) is in the base L-shaped corridor."""
    in_h = (u >= -1e-12) & (u <= 1 + 1e-12) & (v >= -3 - 1e-12) & (v <= 1 + 1e-12)
    in_v = (u >= -3 - 1e-12) & (u <= 1 + 1e-12) & (v >= -1e-12) & (v <= 1 + 1e-12)
    return in_h | in_v


def _compute_sofa_mask(waypoints):
    """
    Compute which sofa points are inside ALL placements along the path.
    Returns a boolean mask of shape (N_SOFA,).
    """
    placements = _generate_placements(waypoints)

    inside_all = np.ones(_N_SOFA, dtype=bool)

    for (x, y, a_deg) in placements:
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)

        dx = _SOFA_X_1D - x
        dy = _SOFA_Y_1D - y

        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy

        in_base = _point_in_base(u, v)
        inside_all &= in_base

        if not inside_all.any():
            break

    return inside_all


def _compute_area(waypoints):
    """Compute the area of the sofa for a given path."""
    mask = _compute_sofa_mask(waypoints)
    return mask.sum() * 0.0001


def _smooth_inside_value(u, v, softness=0.05):
    """
    Compute a smooth indicator of whether (u,v) is inside the base L-corridor.
    Returns values in [0, 1], with 1 for well-inside and 0 for well-outside.
    """
    # For horizontal arm: 0 <= u <= 1, -3 <= v <= 1
    # For vertical arm: -3 <= u <= 1, 0 <= v <= 1

    mu_h1 = u
    mu_h2 = 1.0 - u
    mv_h1 = v + 3.0
    mv_h2 = 1.0 - v

    mu_v1 = u + 3.0
    mu_v2 = 1.0 - u
    mv_v1 = v
    mv_v2 = 1.0 - v

    min_h = np.minimum(np.minimum(mu_h1, mu_h2), np.minimum(mv_h1, mv_h2))
    min_v = np.minimum(np.minimum(mu_v1, mu_v2), np.minimum(mv_v1, mv_v2))

    min_union = np.maximum(min_h, min_v)

    val = np.tanh(min_union / softness)
    return (val + 1.0) / 2.0


def _compute_smooth_value(waypoints, softness=0.05):
    """
    Compute a smooth approximation of the sofa area.
    For each point, compute the minimum smooth-inside-value over all placements.
    """
    placements = _generate_placements(waypoints)

    min_vals = np.full(_N_SOFA, 1.0)

    for (x, y, a_deg) in placements:
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)

        dx = _SOFA_X_1D - x
        dy = _SOFA_Y_1D - y

        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy

        vals = _smooth_inside_value(u, v, softness)
        min_vals = np.minimum(min_vals, vals)

        if min_vals.max() < 0.01:
            break

    return min_vals.sum() * 0.0001


def _neg_smooth_area(params, softness=0.05):
    """Negative smooth area for minimization."""
    n_wp = params.shape[0] // 3
    waypoints = params.reshape(n_wp, 3)
    val = _compute_smooth_value(waypoints, softness)
    return -val


def _neg_area(params, deadline=None):
    """Negative area for minimization."""
    if deadline is not None and time.time() > deadline:
        return 1e6
    n_wp = params.shape[0] // 3
    waypoints = params.reshape(n_wp, 3)
    area = _compute_area(waypoints)
    return -area


def _generate_initial_guesses(n_wp=12):
    """Generate multiple initial guesses for the optimizer."""
    guesses = []

    ts = np.linspace(0, 1, n_wp)
    a = 90.0 * ts

    # Path 1: Classic arc - moderate height
    x = -1.2 * ts + 0.1 * np.sin(np.pi * ts)
    y = 0.4 * np.sin(np.pi * ts)
    guesses.append(np.column_stack([x, y, a]))

    # Path 2: Higher arc
    x = -1.2 * ts + 0.05 * np.sin(np.pi * ts)
    y = 0.5 * np.sin(np.pi * ts)
    guesses.append(np.column_stack([x, y, a]))

    # Path 3: Lower arc
    x = -1.2 * ts + 0.05 * np.sin(np.pi * ts)
    y = 0.2 * np.sin(np.pi * ts)
    guesses.append(np.column_stack([x, y, a]))

    # Path 4: S-curve
    x = -1.2 * ts + 0.15 * np.sin(2 * np.pi * ts)
    y = 0.3 * np.sin(np.pi * ts)
    guesses.append(np.column_stack([x, y, a]))

    # Path 5: Conservative
    x = -1.1 * ts
    y = 0.25 * np.sin(np.pi * ts)
    guesses.append(np.column_stack([x, y, a]))

    # Path 6: Aggressive turn early
    x = -1.2 * ts**1.5
    y = 0.35 * np.sin(np.pi * ts)
    guesses.append(np.column_stack([x, y, a]))

    # Path 7: Aggressive turn late
    x = -1.2 * ts**0.7
    y = 0.35 * np.sin(np.pi * ts)
    guesses.append(np.column_stack([x, y, a]))

    # Path 8: Minimal y
    x = -1.2 * ts
    y = 0.1 * np.sin(np.pi * ts)
    guesses.append(np.column_stack([x, y, a]))

    # Path 9: Quadratic x profile
    x = -1.2 * ts**2
    y = 0.3 * np.sin(np.pi * ts)
    guesses.append(np.column_stack([x, y, a]))

    # Path 10: Cubic x profile
    x = -1.2 * ts**3
    y = 0.35 * np.sin(np.pi * ts)
    guesses.append(np.column_stack([x, y, a]))

    return guesses


def moving_sofa_2d() -> np.ndarray:
    """
    Optimizes the path parameters to maximize the sofa area.
    Uses a parameterized path with 12 waypoints and optimizes their positions/angles.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 600s limit, 10s safety margin

    n_wp = 12

    # Generate initial guesses
    guesses = _generate_initial_guesses(n_wp)

    best_path = None
    best_val = 0.0

    # Evaluate all initial guesses
    for guess in guesses:
        if time.time() > deadline:
            break
        val = _compute_area(guess)
        if val > best_val:
            best_val = val
            best_path = guess.copy()

    if best_path is None:
        # Fallback to a simple path
        best_path = guesses[0]
        best_val = _compute_area(best_path)

    # Phase 1: Smooth surrogate optimization with multiple starts
    if time.time() < deadline - 120:
        for start_idx in range(min(4, len(guesses))):
            if time.time() > deadline - 60:
                break

            guess = guesses[start_idx]
            params0 = guess.ravel()

            try:
                result = minimize(
                    _neg_smooth_area,
                    params0,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-5},
                )
                if time.time() < deadline - 30:
                    if result.fun < 1e5:
                        val = -result.fun
                        if val > best_val:
                            best_val = val
                            best_path = result.x.reshape(n_wp, 3).copy()
            except Exception:
                pass

    # Phase 2: Exact optimization with Nelder-Mead on best path
    if time.time() < deadline - 60:
        try:
            params0 = best_path.ravel()

            result = minimize(
                lambda p: _neg_area(p, deadline),
                params0,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-6},
            )
            if time.time() < deadline - 20:
                if result.fun < 1e5:
                    val = -result.fun
                    if val > best_val:
                        best_val = val
                        best_path = result.x.reshape(n_wp, 3).copy()
        except Exception:
            pass

    # Phase 3: Final polish with tighter tolerances
    if time.time() < deadline - 20:
        try:
            params0 = best_path.ravel()

            result = minimize(
                lambda p: _neg_area(p, deadline),
                params0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-7},
            )
            if time.time() < deadline - 5:
                if result.fun < 1e5:
                    val = -result.fun
                    if val > best_val:
                        best_val = val
                        best_path = result.x.reshape(n_wp, 3).copy()
        except Exception:
            pass

    # Ensure all values are finite and valid
    if not np.all(np.isfinite(best_path)):
        # Fallback to the first guess
        best_path = guesses[0]

    # Ensure angles are in valid range [0, 90]
    best_path[:, 2] = np.clip(best_path[:, 2], 0.0, 90.0)

    return best_path


# EVOLVE-BLOCK-END
